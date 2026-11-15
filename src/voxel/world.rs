// Procedural world: a 16x16 grid of 32³ chunks (256 chunks total, -8..=7 on X and Z)
// generated and greedy-meshed in parallel with `rayon`.
//
// Owns:
//   • Chunk voxel data (`HashMap<IVec3, Chunk>`)
//   • Separate GPU buffers for `opaque_meshes` and `translucent_meshes` (`Water`, `Fire`)
//   • 3D item drops & PrimedTNT dynamic mesh
//   • 20 TPS industrial energy network (`EnergyNetwork`)
//   • Cellular automaton fluid simulator (`FluidSimulator`: Water, Lava, Oil)
//   • Explosions & fire system (`ExplosionSystem`: PrimedTNT, Fire, craters, shockwaves)

use std::collections::{HashMap, HashSet};

use glam::{IVec3, Vec3};
use rayon::prelude::*;
use wgpu::util::DeviceExt;

use crate::sim::energy_net::EnergyNetwork;
use crate::sim::explosions::{ExplosionSystem, TNT_FUSE_TICKS};
use crate::voxel::chunk::{
    is_fluid, is_solid_block, with_fluid_level, BlockId, Chunk, ACTIVE_MACHINE_FLAG, AIR, BEDROCK,
    CHUNK_SIZE, COAL_ORE, COPPER_ORE, DIRT, GRASS, IRON_ORE, OAK_LEAVES, STONE, TIN_ORE, TNT,
    WOOD,
};
use crate::voxel::drops::{block_center, build_drops_mesh, Drops};
use crate::voxel::fluid::FluidSimulator;
use crate::voxel::light::compute_padded_light;
use crate::voxel::mesher::{mesh_chunk, ChunkMeshData, MeshData};

/// Chunks span coordinates `CHUNK_MIN..=CHUNK_MAX` on X and Z (`-8..=7` -> 16×16 = 256 chunks).
pub const CHUNK_MIN: i32 = -8;
pub const CHUNK_MAX: i32 = 7;

struct GpuChunkMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

pub struct World {
    chunks: HashMap<IVec3, Chunk>,
    opaque_meshes: HashMap<IVec3, GpuChunkMesh>,
    translucent_meshes: HashMap<IVec3, GpuChunkMesh>,
    drops_mesh: Option<GpuChunkMesh>,
    pub drops: Drops,
    pub energy: EnergyNetwork,
    pub fluids: FluidSimulator,
    pub explosions: ExplosionSystem,
}

// ─── Procedural generation ───────────────────────────────────────────────────

fn hash3(x: i32, y: i32, z: i32, seed: u32) -> u32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343)
        ^ (y as u32).wrapping_mul(0xd816_3841)
        ^ (z as u32).wrapping_mul(0xcb1a_b31f)
        ^ seed.wrapping_mul(0x9e37_79b9);
    h ^= h >> 13;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 16;
    h
}

fn terrain_height(wx: i32, wz: i32) -> i32 {
    let (x, z) = (wx as f32, wz as f32);
    let h = 16.0
        + 5.0 * (x * 0.06).sin()
        + 4.0 * (z * 0.08).cos()
        + 2.5 * ((x + z) * 0.17).sin()
        + 1.5 * ((x - z) * 0.31).cos();
    (h as i32).clamp(3, CHUNK_SIZE as i32 - 8)
}

fn block_for(wx: i32, wy: i32, wz: i32, height: i32) -> BlockId {
    if wy == 0 {
        BEDROCK
    } else if wy > height {
        AIR
    } else if wy == height {
        GRASS
    } else if wy >= height - 3 {
        DIRT
    } else {
        // Underground: stone with clustered ore veins (Coal, Copper, Tin, Iron).
        let cell = hash3(wx.div_euclid(3), wy.div_euclid(3), wz.div_euclid(3), 1) % 1000;
        let filled = hash3(wx, wy, wz, 2) % 4 != 0;
        if filled && cell < 40 {
            COAL_ORE
        } else if filled && cell < 65 && wy < height - 5 {
            COPPER_ORE
        } else if filled && cell < 88 && wy < height - 5 {
            TIN_ORE
        } else if filled && cell < 112 && wy < height - 7 {
            IRON_ORE
        } else {
            STONE
        }
    }
}

fn generate_chunk(coord: IVec3) -> Chunk {
    let mut chunk = Chunk::new();
    let n = CHUNK_SIZE as i32;

    // 1. Base terrain & ore veins
    for lz in 0..n {
        for lx in 0..n {
            let wx = coord.x * n + lx;
            let wz = coord.z * n + lz;
            let height = terrain_height(wx, wz);
            for y in 0..=height {
                chunk.set_block(lx, y, lz, block_for(wx, y, wz, height));
            }
        }
    }

    // 2. Dome-shaped Oak Trees (4-5 block WoodLog trunk + 5x5 / 3x3 / cross OakLeaves crown)
    for lz in 2..(n - 2) {
        for lx in 2..(n - 2) {
            let wx = coord.x * n + lx;
            let wz = coord.z * n + lz;
            if wx.abs() <= 5 && wz.abs() <= 5 {
                continue; // keep spawn area clear
            }
            if hash3(wx, 0, wz, 77) % 175 != 0 {
                continue;
            }
            let base_y = terrain_height(wx, wz);
            let trunk_h = 4 + (hash3(wx, 1, wz, 99) % 2) as i32; // 4 or 5 blocks tall
            if base_y + trunk_h + 2 >= n {
                continue;
            }

            // Lower 2 canopy layers: 5x5 around trunk with rounded corners
            for ly in (base_y + trunk_h - 2)..=(base_y + trunk_h - 1) {
                for dz in -2i32..=2 {
                    for dx in -2i32..=2 {
                        if dx.abs() == 2 && dz.abs() == 2 {
                            continue; // cut extreme corners for a dome silhouette
                        }
                        if chunk.get_block(lx + dx, ly, lz + dz) == AIR {
                            chunk.set_block(lx + dx, ly, lz + dz, OAK_LEAVES);
                        }
                    }
                }
            }

            // Top trunk level: 3x3 around trunk (with corners cut on upper crown)
            let top_y = base_y + trunk_h;
            for dz in -1i32..=1 {
                for dx in -1i32..=1 {
                    if chunk.get_block(lx + dx, top_y, lz + dz) == AIR {
                        chunk.set_block(lx + dx, top_y, lz + dz, OAK_LEAVES);
                    }
                }
            }

            // Crown cap (above trunk): cross shape (`|dx| + |dz| <= 1`)
            let cap_y = top_y + 1;
            for dz in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx.abs() + dz.abs() <= 1 && chunk.get_block(lx + dx, cap_y, lz + dz) == AIR
                    {
                        chunk.set_block(lx + dx, cap_y, lz + dz, OAK_LEAVES);
                    }
                }
            }

            // Place the WoodLog trunk last so it runs cleanly up the center
            for ty in 1..=trunk_h {
                chunk.set_block(lx, base_y + ty, lz, WOOD);
            }
        }
    }

    chunk
}

// ─── World ───────────────────────────────────────────────────────────────────

fn block_in_chunks(chunks: &HashMap<IVec3, Chunk>, p: IVec3) -> BlockId {
    let n = CHUNK_SIZE as i32;
    let coord = IVec3::new(p.x.div_euclid(n), p.y.div_euclid(n), p.z.div_euclid(n));
    if coord.y < 0 {
        return BEDROCK;
    }
    match chunks.get(&coord) {
        Some(c) => c.get_block(p.x.rem_euclid(n), p.y.rem_euclid(n), p.z.rem_euclid(n)),
        None => AIR,
    }
}

impl World {
    pub fn generate(device: &wgpu::Device) -> Self {
        let mut coords = Vec::with_capacity(256);
        for cz in CHUNK_MIN..=CHUNK_MAX {
            for cx in CHUNK_MIN..=CHUNK_MAX {
                coords.push(IVec3::new(cx, 0, cz));
            }
        }

        // Parallel terrain generation across all 256 chunks via Rayon.
        let generated: Vec<(IVec3, Chunk)> = coords
            .par_iter()
            .map(|&coord| (coord, generate_chunk(coord)))
            .collect();

        let chunks: HashMap<IVec3, Chunk> = generated.into_iter().collect();

        let mut world = Self {
            chunks,
            opaque_meshes: HashMap::new(),
            translucent_meshes: HashMap::new(),
            drops_mesh: None,
            drops: Drops::new(),
            energy: EnergyNetwork::new(),
            fluids: FluidSimulator::new(),
            explosions: ExplosionSystem::new(),
        };

        // Parallel initial Greedy Meshing across all 256 chunks via Rayon.
        let built_meshes: Vec<(IVec3, ChunkMeshData)> = coords
            .par_iter()
            .map(|&coord| (coord, world.build_mesh(coord)))
            .collect();

        let (mut vertices, mut quads, mut naive_faces) = (0usize, 0usize, 0usize);
        for (coord, data) in built_meshes {
            vertices += data.opaque.vertices.len() + data.translucent.vertices.len();
            quads += (data.opaque.indices.len() + data.translucent.indices.len()) / 6;
            naive_faces += data.opaque.naive_faces;
            world.upload(device, coord, data);
        }

        log::info!(
            "World: {} chunks (16x16), {} greedy quads ({} vertices) vs {} naive faces ({} vertices)",
            coords.len(),
            quads,
            vertices,
            naive_faces,
            naive_faces * 4
        );
        world
    }

    pub fn active_chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Fixed 20 TPS logic tick:
    ///   1. Industrial Energy Network + furnaces/smelters
    ///   2. Cellular Automaton Fluids (Water, Lava, Oil + Obsidian/Cobblestone crystallization)
    ///   3. Fire & PrimedTNT explosions (crater destruction, 40% item drops, shockwave knockback)
    /// Returns any net shockwave velocity impulse to apply to the player.
    pub fn tick(&mut self, device: &wgpu::Device, player_center: Vec3) -> Vec3 {
        let mut dirty_positions: Vec<IVec3> = self.energy.tick();
        let mut total_player_impulse = Vec3::ZERO;

        // Step cellular fluid simulation (rate-limited inside FluidSimulator)
        let chunks_ref = &self.chunks;
        let fluid_edits = self.fluids.tick(|p| block_in_chunks(chunks_ref, p));
        for (pos, new_id) in fluid_edits {
            let _ = self.set_block_raw(pos, new_id);
            // Always mark dirty because even if `new_id == WATER` was already there, its fluid
            // level (`1..=8`) may have changed, changing its sloped vertex height!
            dirty_positions.push(pos);

            // If Lava flowed next to a TNT block, auto-ignite it!
            if new_id == crate::voxel::chunk::LAVA {
                for d in [
                    IVec3::X,
                    -IVec3::X,
                    IVec3::Y,
                    -IVec3::Y,
                    IVec3::Z,
                    -IVec3::Z,
                ] {
                    let np = pos + d;
                    if self.block(np) == TNT {
                        self.set_block_raw(np, AIR);
                        self.explosions.ignite_tnt(np, TNT_FUSE_TICKS);
                        dirty_positions.push(np);
                    }
                }
            }
        }

        // Step Fire & PrimedTNT explosions
        let chunks_ref = &self.chunks;
        let (fire_changes, detonations) = self
            .explosions
            .tick(player_center, |p| block_in_chunks(chunks_ref, p));
        for (pos, new_id) in fire_changes {
            if self.set_block_raw(pos, new_id) {
                dirty_positions.push(pos);
            }
        }

        let had_detonation = !detonations.is_empty();
        for det in detonations {
            log::info!(
                "[Explosion] TNT detonated at ({:.1}, {:.1}, {:.1}) — {} blocks destroyed",
                det.center.x,
                det.center.y,
                det.center.z,
                det.destroyed_blocks.len()
            );
            total_player_impulse += det.player_impulse;
            for (bp, old_id) in det.destroyed_blocks {
                if self.set_block_raw(bp, AIR) {
                    dirty_positions.push(bp);
                    // Drop any items stored inside destroyed machines/chests
                    let center = block_center(bp);
                    for (stored_id, count) in self.energy.contents(bp) {
                        self.drops.spawn(center, stored_id, count);
                    }
                    self.energy.on_block_changed(bp, old_id, AIR);
                    self.fluids.on_block_changed(bp, AIR);
                }
            }
            for (drop_pos, item_id) in det.spawned_drops {
                self.drops.spawn(drop_pos, item_id, 1);
            }
        }
        if had_detonation {
            // Immediately coalesce nearby explosion drops so they form tidy stacks
            self.drops.merge_nearby_stacks();
        }

        if !dirty_positions.is_empty() {
            self.remesh_positions(device, &dirty_positions);
        }

        total_player_impulse
    }

    /// World-space block lookup across chunk borders. Below the world is bedrock,
    /// above/outside the loaded area is air.
    pub fn block(&self, p: IVec3) -> BlockId {
        block_in_chunks(&self.chunks, p)
    }

    /// True if `p` contains a solid block that stops player/drop AABB movement.
    pub fn is_solid(&self, p: IVec3) -> bool {
        is_solid_block(self.block(p))
    }

    /// True if `p` contains a liquid (`Water`, `Lava`, or `Oil`).
    pub fn is_fluid_at(&self, p: IVec3) -> bool {
        is_fluid(self.block(p))
    }

    /// Returns the normalized fluid flow vector at `p` (`Vec3::ZERO` if stagnant or not a fluid).
    pub fn fluid_flow_at(&self, p: IVec3) -> Vec3 {
        self.fluids.flow_vector_at(p)
    }

    /// Ignites a `TNT` block at `p` into a 60-tick `PrimedTnt` entity.
    pub fn ignite_tnt_at(&mut self, device: &wgpu::Device, p: IVec3) -> bool {
        if self.block(p) != TNT {
            return false;
        }
        if self.set_block_raw(p, AIR) {
            self.explosions.ignite_tnt(p, TNT_FUSE_TICKS);
            self.remesh_positions(device, &[p]);
            return true;
        }
        false
    }

    /// Mutates a single voxel inside `self.chunks` without triggering drops or immediate remesh.
    fn set_block_raw(&mut self, p: IVec3, id: BlockId) -> bool {
        let n = CHUNK_SIZE as i32;
        let coord = IVec3::new(p.x.div_euclid(n), p.y.div_euclid(n), p.z.div_euclid(n));
        let local = IVec3::new(p.x.rem_euclid(n), p.y.rem_euclid(n), p.z.rem_euclid(n));
        let Some(chunk) = self.chunks.get_mut(&coord) else {
            return false;
        };
        let old = chunk.get_block(local.x, local.y, local.z);
        if old == id {
            return false;
        }
        chunk.set_block(local.x, local.y, local.z, id);
        true
    }

    /// Rebuilds meshes for all chunks touched by `positions` (including neighbor chunks on borders).
    fn remesh_positions(&mut self, device: &wgpu::Device, positions: &[IVec3]) {
        let n = CHUNK_SIZE as i32;
        let mut dirty_set: HashSet<IVec3> = HashSet::new();
        for &p in positions {
            let coord = IVec3::new(p.x.div_euclid(n), p.y.div_euclid(n), p.z.div_euclid(n));
            let local = IVec3::new(p.x.rem_euclid(n), p.y.rem_euclid(n), p.z.rem_euclid(n));
            if self.chunks.contains_key(&coord) {
                dirty_set.insert(coord);
            }
            for axis in 0..3 {
                if local[axis] == 0 {
                    let mut c = coord;
                    c[axis] -= 1;
                    if self.chunks.contains_key(&c) {
                        dirty_set.insert(c);
                    }
                }
                if local[axis] == n - 1 {
                    let mut c = coord;
                    c[axis] += 1;
                    if self.chunks.contains_key(&c) {
                        dirty_set.insert(c);
                    }
                }
            }
        }

        let dirty: Vec<IVec3> = dirty_set.into_iter().collect();
        if dirty.len() > 2 {
            let built: Vec<(IVec3, ChunkMeshData)> = dirty
                .par_iter()
                .map(|&c| (c, self.build_mesh(c)))
                .collect();
            for (c, data) in built {
                self.upload(device, c, data);
            }
        } else {
            for c in dirty {
                let data = self.build_mesh(c);
                self.upload(device, c, data);
            }
        }
    }

    /// Edits one block, notifies the energy/fluid/explosion systems, spawns item drops when
    /// breaking, and re-meshes the affected chunk (plus neighbours when on a border).
    /// Returns `true` if the block was actually modified.
    pub fn set_block(&mut self, device: &wgpu::Device, p: IVec3, id: BlockId) -> bool {
        let n = CHUNK_SIZE as i32;
        let coord = IVec3::new(p.x.div_euclid(n), p.y.div_euclid(n), p.z.div_euclid(n));
        let local = IVec3::new(p.x.rem_euclid(n), p.y.rem_euclid(n), p.z.rem_euclid(n));

        let Some(old) = self
            .chunks
            .get(&coord)
            .map(|c| c.get_block(local.x, local.y, local.z))
        else {
            return false;
        };
        if old == id || (old == BEDROCK && id == AIR) {
            return false;
        }

        // If placing TNT next to Fire or Lava, immediately prime it!
        if id == TNT && ExplosionSystem::should_auto_ignite(p, |q| block_in_chunks(&self.chunks, q))
        {
            self.explosions.ignite_tnt(p, TNT_FUSE_TICKS);
            return true;
        }

        if let Some(chunk) = self.chunks.get_mut(&coord) {
            chunk.set_block(local.x, local.y, local.z, id);
        }

        if id == AIR && old != AIR && !is_fluid(old) && old != crate::voxel::chunk::FIRE {
            let center = block_center(p);
            self.drops.spawn(center, old, 1);
            for (stored_id, count) in self.energy.contents(p) {
                self.drops.spawn(center, stored_id, count);
            }
        }
        self.energy.on_block_changed(p, old, id);
        self.fluids.on_block_changed(p, id);
        self.explosions.on_block_changed(p, id);

        self.remesh_positions(device, &[p]);
        true
    }

    /// Rebuilds the dynamic GPU mesh for all active item drops and flashing `PrimedTnt` entities.
    pub fn update_drops_mesh(&mut self, device: &wgpu::Device) {
        let items = self.drops.items();
        if items.is_empty() && self.explosions.primed.is_empty() {
            self.drops_mesh = None;
            return;
        }
        let (mut vertices, mut indices) = build_drops_mesh(items);
        self.explosions.append_mesh(&mut vertices, &mut indices);
        if indices.is_empty() {
            self.drops_mesh = None;
            return;
        }
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("drops_vertex_buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("drops_index_buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        self.drops_mesh = Some(GpuChunkMesh {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
        });
    }

    /// Greedy-meshes a chunk into separate opaque and translucent vertex/index buffers.
    fn build_mesh(&self, coord: IVec3) -> ChunkMeshData {
        let n = CHUNK_SIZE as i32;
        let pad = n + 2;
        let origin = coord * n;

        let mut cells = vec![AIR; (pad * pad * pad) as usize];
        for z in 0..pad {
            for y in 0..pad {
                for x in 0..pad {
                    let wp = origin + IVec3::new(x - 1, y - 1, z - 1);
                    let mut b = self.block(wp);
                    if b != AIR {
                        if is_fluid(b) {
                            let lvl = self.fluids.level_at(wp).unwrap_or(8);
                            b = with_fluid_level(b, lvl);
                        } else if self.energy.is_machine_active(wp) {
                            b |= ACTIVE_MACHINE_FLAG;
                        }
                    }
                    cells[(x + y * pad + z * pad * pad) as usize] = b;
                }
            }
        }

        let lights = compute_padded_light(&cells, pad);

        mesh_chunk(
            origin,
            |x, y, z| cells[((x + 1) + (y + 1) * pad + (z + 1) * pad * pad) as usize],
            |x, y, z| lights[((x + 1) + (y + 1) * pad + (z + 1) * pad * pad) as usize],
        )
    }

    fn upload_single(
        device: &wgpu::Device,
        map: &mut HashMap<IVec3, GpuChunkMesh>,
        coord: IVec3,
        data: MeshData,
    ) {
        if data.indices.is_empty() {
            map.remove(&coord);
            return;
        }
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk_vertex_buffer"),
            contents: bytemuck::cast_slice(&data.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk_index_buffer"),
            contents: bytemuck::cast_slice(&data.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        map.insert(
            coord,
            GpuChunkMesh {
                vertex_buffer,
                index_buffer,
                index_count: data.indices.len() as u32,
            },
        );
    }

    fn upload(&mut self, device: &wgpu::Device, coord: IVec3, data: ChunkMeshData) {
        Self::upload_single(device, &mut self.opaque_meshes, coord, data.opaque);
        Self::upload_single(
            device,
            &mut self.translucent_meshes,
            coord,
            data.translucent,
        );
    }

    pub fn draw_opaque<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        for m in self.opaque_meshes.values() {
            pass.set_vertex_buffer(0, m.vertex_buffer.slice(..));
            pass.set_index_buffer(m.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..m.index_count, 0, 0..1);
        }
        if let Some(m) = &self.drops_mesh {
            pass.set_vertex_buffer(0, m.vertex_buffer.slice(..));
            pass.set_index_buffer(m.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..m.index_count, 0, 0..1);
        }
    }

    pub fn draw_translucent<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        for m in self.translucent_meshes.values() {
            pass.set_vertex_buffer(0, m.vertex_buffer.slice(..));
            pass.set_index_buffer(m.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..m.index_count, 0, 0..1);
        }
    }
}
