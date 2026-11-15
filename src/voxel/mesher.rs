// Greedy meshing for opaque full-cube blocks + multipart sub-block meshing for cables & torches +
// proportional-height fluid meshing (`height = (level as f32) / 8.0`) + separate translucent
// mesh buffer (`Water`, `Fire`, `PurpleStainedGlass`) + RGB/Sun per-vertex `LightData`.

use glam::{IVec3, Vec3};

use crate::voxel::chunk::{
    base_block_id, block_color, block_face_color, fluid_level_from_id, is_active_block,
    is_energy_connectable, is_fluid, is_full_cube, is_translucent, BlockId, AIR, CABLE, CABLE_MAX,
    CABLE_MIN, CHUNK_SIZE, FIRE, LAVA, PURPLE_STAINED_GLASS, TORCH,
};
use crate::voxel::light::{
    blue_channel, green_channel, light_to_f32_array, pack_light, red_channel, sun_channel,
    LightData,
};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
    /// `[Red, Green, Blue, Sunlight]`, each in `0.0..=15.0`.
    pub light: [f32; 4],
}

pub struct MeshData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    /// Number of visible unit faces (what a naive mesher would emit as quads).
    pub naive_faces: usize,
}

impl MeshData {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            naive_faces: 0,
        }
    }
}

pub struct ChunkMeshData {
    pub opaque: MeshData,
    pub translucent: MeshData,
}

/// `sample` and `sample_light` take chunk-local coordinates (may lie in `-1..=32`, i.e. in neighbours).
pub fn mesh_chunk<F, L>(origin: IVec3, sample: F, sample_light: L) -> ChunkMeshData
where
    F: Fn(i32, i32, i32) -> BlockId,
    L: Fn(i32, i32, i32) -> LightData,
{
    let size = CHUNK_SIZE;
    let n = CHUNK_SIZE as i32;
    let mut opaque = MeshData::new();
    let mut translucent = MeshData::new();
    let mut mask = vec![(AIR, 0u16); size * size];

    // ── Pass 1: Greedy meshing of opaque full-cube voxels ─────────────────
    for d in 0..3usize {
        let u = (d + 1) % 3;
        let v = (d + 2) % 3;

        for positive in [true, false] {
            let step = if positive { 1 } else { -1 };

            for slice in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        let mut p = [0i32; 3];
                        p[d] = slice;
                        p[u] = k;
                        p[v] = j;
                        let mut q = p;
                        q[d] += step;

                        let b = sample(p[0], p[1], p[2]);
                        let visible = is_full_cube(b) && !is_full_cube(sample(q[0], q[1], q[2]));
                        if visible {
                            opaque.naive_faces += 1;
                            let mut fl = sample_light(q[0], q[1], q[2]);
                            if is_active_block(b) {
                                fl = pack_light(
                                    sun_channel(fl),
                                    red_channel(fl).max(14),
                                    green_channel(fl).max(7),
                                    blue_channel(fl).max(1),
                                );
                            }
                            mask[(j * n + k) as usize] = (b, fl);
                        } else {
                            mask[(j * n + k) as usize] = (AIR, 0);
                        }
                    }
                }

                // Greedy merge faces with identical (BlockId, LightData).
                for j in 0..size {
                    let mut k = 0;
                    while k < size {
                        let cell = mask[j * size + k];
                        if cell.0 == AIR {
                            k += 1;
                            continue;
                        }

                        let mut w = 1;
                        while k + w < size && mask[j * size + k + w] == cell {
                            w += 1;
                        }
                        let mut h = 1;
                        'grow: while j + h < size {
                            for x in 0..w {
                                if mask[(j + h) * size + k + x] != cell {
                                    break 'grow;
                                }
                            }
                            h += 1;
                        }
                        for dy in 0..h {
                            for dx in 0..w {
                                mask[(j + dy) * size + k + dx] = (AIR, 0);
                            }
                        }

                        emit_quad(
                            &mut opaque,
                            origin,
                            d,
                            u,
                            v,
                            positive,
                            slice,
                            k,
                            j,
                            w,
                            h,
                            cell.0,
                            light_to_f32_array(cell.1),
                        );
                        k += w;
                    }
                }
            }
        }
    }

    // ── Pass 2: Cables, Torches, variable-height Fluids, Fire & Stained Glass ──
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let raw = sample(x, y, z);
                let base = base_block_id(raw);
                if base == AIR {
                    continue;
                }
                let wp = origin + IVec3::new(x, y, z);
                let cell_light = light_to_f32_array(sample_light(x, y, z));
                if base == CABLE {
                    emit_cable(&mut opaque, wp, x, y, z, cell_light, &sample);
                } else if base == TORCH {
                    emit_torch(&mut opaque, wp);
                } else if base == PURPLE_STAINED_GLASS {
                    emit_stained_glass(&mut translucent, wp, x, y, z, cell_light, &sample);
                } else if is_fluid(base) {
                    let target_mesh = if is_translucent(base) {
                        &mut translucent
                    } else {
                        &mut opaque
                    };
                    let fl = if base == LAVA {
                        [15.0, 8.0, 2.0, 15.0]
                    } else {
                        cell_light
                    };
                    emit_fluid_voxel(target_mesh, wp, x, y, z, raw, fl, &sample);
                } else if base == FIRE {
                    emit_fire_voxel(&mut translucent, wp);
                }
            }
        }
    }

    ChunkMeshData {
        opaque,
        translucent,
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_quad(
    data: &mut MeshData,
    origin: IVec3,
    d: usize,
    u: usize,
    v: usize,
    positive: bool,
    slice: i32,
    k: usize,
    j: usize,
    w: usize,
    h: usize,
    block: BlockId,
    light: [f32; 4],
) {
    let plane = slice + if positive { 1 } else { 0 };
    let (k, j, w, h) = (k as i32, j as i32, w as i32, h as i32);

    let mut normal = [0.0f32; 3];
    normal[d] = if positive { 1.0 } else { -1.0 };
    let color = block_face_color(block, d);

    let base = data.vertices.len() as u32;
    for (cu, cv) in [(k, j), (k + w, j), (k + w, j + h), (k, j + h)] {
        let mut p = [0i32; 3];
        p[d] = plane;
        p[u] = cu;
        p[v] = cv;
        data.vertices.push(Vertex {
            position: [
                (origin.x + p[0]) as f32,
                (origin.y + p[1]) as f32,
                (origin.z + p[2]) as f32,
            ],
            normal,
            color,
            light,
        });
    }

    if positive {
        data.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    } else {
        data.indices
            .extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
    }
}

/// Computes the rendered height `0.125..=1.0` of a fluid voxel from its level `1..=8`.
#[inline]
fn fluid_height<F: Fn(i32, i32, i32) -> BlockId>(
    x: i32,
    y: i32,
    z: i32,
    raw: BlockId,
    sample: &F,
) -> f32 {
    let base = base_block_id(raw);
    if base_block_id(sample(x, y + 1, z)) == base {
        1.0
    } else {
        let level = fluid_level_from_id(raw);
        (level as f32) / 8.0
    }
}

/// Emits a fluid voxel (`Water`, `Lava`, `Oil`) with top height `(level as f32) / 8.0`.
#[allow(clippy::too_many_arguments)]
fn emit_fluid_voxel<F: Fn(i32, i32, i32) -> BlockId>(
    data: &mut MeshData,
    world_pos: IVec3,
    lx: i32,
    ly: i32,
    lz: i32,
    raw: BlockId,
    light: [f32; 4],
    sample: &F,
) {
    let base = base_block_id(raw);
    let h = fluid_height(lx, ly, lz, raw, sample);
    let color = block_face_color(raw, 1);

    let nb_visible = |nx: i32, ny: i32, nz: i32| -> bool {
        let n_raw = sample(nx, ny, nz);
        let n_base = base_block_id(n_raw);
        if is_full_cube(n_raw) {
            return false;
        }
        if n_base == base {
            // Only show side face if our fluid surface is higher than the neighbor's fluid surface.
            let nh = fluid_height(nx, ny, nz, n_raw, sample);
            return h > nh + 0.01;
        }
        true
    };

    let show_top = base_block_id(sample(lx, ly + 1, lz)) != base;
    let show_bottom = {
        let b_below = sample(lx, ly - 1, lz);
        !is_full_cube(b_below) && base_block_id(b_below) != base
    };
    let show_px = nb_visible(lx + 1, ly, lz);
    let show_nx = nb_visible(lx - 1, ly, lz);
    let show_pz = nb_visible(lx, ly, lz + 1);
    let show_nz = nb_visible(lx, ly, lz - 1);

    let bmin = world_pos.as_vec3();
    let bmax = bmin + Vec3::new(1.0, h, 1.0);
    emit_box(
        data,
        bmin,
        bmax,
        color,
        light,
        [show_px, show_nx, show_top, show_bottom, show_pz, show_nz],
    );
}

/// Emits a glowing translucent fire sub-block (`[0.12..0.88]` on X/Z, `[0.0..0.82]` on Y).
fn emit_fire_voxel(data: &mut MeshData, world_pos: IVec3) {
    let base_pos = world_pos.as_vec3();
    let color = block_face_color(FIRE, 1);
    emit_box(
        data,
        base_pos + Vec3::new(0.12, 0.0, 0.12),
        base_pos + Vec3::new(0.88, 0.82, 0.88),
        color,
        [15.0, 8.0, 2.0, 15.0],
        [true, true, true, false, true, true],
    );
}

/// Emits a Torch sub-block (`[0.42..0.58]` wooden stick + `[0.40..0.60]` glowing flame tip).
fn emit_torch(data: &mut MeshData, world_pos: IVec3) {
    let base_pos = world_pos.as_vec3();
    let light = [15.0, 12.0, 5.0, 15.0];
    // Wooden handle
    emit_box(
        data,
        base_pos + Vec3::new(0.42, 0.0, 0.42),
        base_pos + Vec3::new(0.58, 0.50, 0.58),
        [0.52, 0.36, 0.20],
        light,
        [true, true, false, true, true, true],
    );
    // Glowing flame head
    emit_box(
        data,
        base_pos + Vec3::new(0.40, 0.50, 0.40),
        base_pos + Vec3::new(0.60, 0.66, 0.60),
        block_face_color(TORCH, 1),
        light,
        [true, true, true, false, true, true],
    );
}

/// Emits a translucent `PURPLE_STAINED_GLASS` block, culling faces shared with adjacent glass or opaque cubes.
fn emit_stained_glass<F: Fn(i32, i32, i32) -> BlockId>(
    data: &mut MeshData,
    world_pos: IVec3,
    lx: i32,
    ly: i32,
    lz: i32,
    light: [f32; 4],
    sample: &F,
) {
    let vis = |nx: i32, ny: i32, nz: i32| -> bool {
        let nb = sample(nx, ny, nz);
        !is_full_cube(nb) && base_block_id(nb) != PURPLE_STAINED_GLASS
    };
    let faces = [
        vis(lx + 1, ly, lz),
        vis(lx - 1, ly, lz),
        vis(lx, ly + 1, lz),
        vis(lx, ly - 1, lz),
        vis(lx, ly, lz + 1),
        vis(lx, ly, lz - 1),
    ];
    let bmin = world_pos.as_vec3();
    emit_box(
        data,
        bmin,
        bmin + Vec3::ONE,
        block_face_color(PURPLE_STAINED_GLASS, 1),
        light,
        faces,
    );
}

/// Emits the central `[0.35..0.65]³` core of a cable and up to 6 directional branches
/// connecting to adjacent cables or energy machines.
fn emit_cable<F: Fn(i32, i32, i32) -> BlockId>(
    data: &mut MeshData,
    world_pos: IVec3,
    lx: i32,
    ly: i32,
    lz: i32,
    light: [f32; 4],
    sample: &F,
) {
    let base_pos = world_pos.as_vec3();
    let color = block_color(CABLE);
    let c0 = CABLE_MIN;
    let c1 = CABLE_MAX;

    let conn_px = is_energy_connectable(sample(lx + 1, ly, lz));
    let conn_nx = is_energy_connectable(sample(lx - 1, ly, lz));
    let conn_py = is_energy_connectable(sample(lx, ly + 1, lz));
    let conn_ny = is_energy_connectable(sample(lx, ly - 1, lz));
    let conn_pz = is_energy_connectable(sample(lx, ly, lz + 1));
    let conn_nz = is_energy_connectable(sample(lx, ly, lz - 1));

    // Central core (skip faces that are covered by a branch).
    emit_box(
        data,
        base_pos + Vec3::splat(c0),
        base_pos + Vec3::splat(c1),
        color,
        light,
        [!conn_px, !conn_nx, !conn_py, !conn_ny, !conn_pz, !conn_nz],
    );

    if conn_px {
        emit_box(
            data,
            base_pos + Vec3::new(c1, c0, c0),
            base_pos + Vec3::new(1.0, c1, c1),
            color,
            light,
            [true, false, true, true, true, true],
        );
    }
    if conn_nx {
        emit_box(
            data,
            base_pos + Vec3::new(0.0, c0, c0),
            base_pos + Vec3::new(c0, c1, c1),
            color,
            light,
            [false, true, true, true, true, true],
        );
    }
    if conn_py {
        emit_box(
            data,
            base_pos + Vec3::new(c0, c1, c0),
            base_pos + Vec3::new(c1, 1.0, c1),
            color,
            light,
            [true, true, true, false, true, true],
        );
    }
    if conn_ny {
        emit_box(
            data,
            base_pos + Vec3::new(c0, 0.0, c0),
            base_pos + Vec3::new(c1, c0, c1),
            color,
            light,
            [true, true, false, true, true, true],
        );
    }
    if conn_pz {
        emit_box(
            data,
            base_pos + Vec3::new(c0, c0, c1),
            base_pos + Vec3::new(c1, c1, 1.0),
            color,
            light,
            [true, true, true, true, true, false],
        );
    }
    if conn_nz {
        emit_box(
            data,
            base_pos + Vec3::new(c0, c0, 0.0),
            base_pos + Vec3::new(c1, c1, c0),
            color,
            light,
            [true, true, true, true, false, true],
        );
    }
}

/// Emits an axis-aligned box `[min..max]` with selective face flags `[+X, -X, +Y, -Y, +Z, -Z]`.
fn emit_box(
    data: &mut MeshData,
    min: Vec3,
    max: Vec3,
    color: [f32; 3],
    light: [f32; 4],
    faces: [bool; 6],
) {
    let (x0, y0, z0) = (min.x, min.y, min.z);
    let (x1, y1, z1) = (max.x, max.y, max.z);

    let quads: [(bool, [[f32; 3]; 4], [f32; 3]); 6] = [
        // +X
        (
            faces[0],
            [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]],
            [1.0, 0.0, 0.0],
        ),
        // -X
        (
            faces[1],
            [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]],
            [-1.0, 0.0, 0.0],
        ),
        // +Y
        (
            faces[2],
            [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]],
            [0.0, 1.0, 0.0],
        ),
        // -Y
        (
            faces[3],
            [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],
            [0.0, -1.0, 0.0],
        ),
        // +Z
        (
            faces[4],
            [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
            [0.0, 0.0, 1.0],
        ),
        // -Z
        (
            faces[5],
            [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]],
            [0.0, 0.0, -1.0],
        ),
    ];

    for (enabled, corners, normal) in quads {
        if !enabled {
            continue;
        }
        let base = data.vertices.len() as u32;
        for position in corners {
            data.vertices.push(Vertex {
                position,
                normal,
                color,
                light,
            });
        }
        data.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}
