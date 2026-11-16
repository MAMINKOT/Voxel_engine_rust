// Greedy meshing for opaque full-cube blocks + Corner Ambient Occlusion (Smooth Lighting)
// with anisotropic quad flipping + multipart sub-block meshing for cables & torches +
// proportional-height fluid meshing (`height = (level as f32) / 8.0`) + separate translucent
// mesh buffer (`Water`, `Fire`, `PurpleStainedGlass`) + RGB/Sun per-vertex `LightData`.

use glam::{IVec3, Vec3};

use crate::voxel::blocks::{
    block_face_emission, get_face_texture, texture_tint_color, BlockFace, TextureLayer,
    NO_TEXTURE,
};
use crate::voxel::chunk::{
    base_block_id, block_face_color, fluid_level_from_id, is_active_block, is_cross_foliage,
    is_energy_connectable, is_fluid, is_full_cube, is_solid_block, is_stained_glass,
    is_translucent, BlockId, AIR, CABLE, CABLE_MAX, CABLE_MIN, CHUNK_SIZE, FIRE, GRASS, LAVA,
    OAK_LEAVES, TORCH, WATER,
};
use crate::voxel::light::{
    blue_channel, green_channel, light_to_f32_array, pack_light, red_channel, sun_channel,
    LightData,
};

/// Material flag stored in `Vertex::ao_and_flags[1]`:
pub const MAT_STANDARD: f32 = 0.0;
pub const MAT_WATER: f32 = 1.0;
pub const MAT_STAINED_GLASS: f32 = 2.0;
pub const MAT_FIRE: f32 = 3.0;
pub const MAT_EMISSIVE: f32 = 4.0;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
    /// `[Red, Green, Blue, Sunlight]`, each in `0.0..=15.0`.
    pub light: [f32; 4],
    /// `[ao_level (0.0..=3.0), material_flag (0=standard, 1=water, 2=glass, 3=fire, 4=emissive)]`.
    pub ao_and_flags: [f32; 2],
    /// Tiled UV coordinates (`[0..w, 0..h]`) for `AddressMode::Repeat`.
    pub uv: [f32; 2],
    /// Layer index in the `texture_2d_array<f32>` (`NO_TEXTURE` = untextured vertex color).
    pub tex_layer: u32,
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

/// Computes the classic voxel Corner Ambient Occlusion level (`0..=3`) for one quad vertex
/// from the 3 neighboring voxels in the plane in front of the face (`side1`, `side2`, `corner`).
#[inline]
pub fn vertex_ao(side1_solid: bool, side2_solid: bool, corner_solid: bool) -> u8 {
    if side1_solid && side2_solid {
        0
    } else {
        3 - (side1_solid as u8 + side2_solid as u8 + corner_solid as u8)
    }
}

/// Computes the 4 corner AO values `[ao0, ao1, ao2, ao3]` for the unit face at `q`
/// (the air cell immediately in front of the face) spanned by tangent axes `(u, v)`.
#[inline]
fn compute_face_ao<F: Fn(i32, i32, i32) -> BlockId>(
    q: [i32; 3],
    u: usize,
    v: usize,
    sample: &F,
) -> [u8; 4] {
    let is_occluder = |du: i32, dv: i32| -> bool {
        let mut pos = q;
        pos[u] += du;
        pos[v] += dv;
        is_full_cube(sample(pos[0], pos[1], pos[2]))
    };

    // 8 neighbors around `q` in the (u, v) tangent plane
    let s_neg_u = is_occluder(-1, 0);
    let s_pos_u = is_occluder(1, 0);
    let s_neg_v = is_occluder(0, -1);
    let s_pos_v = is_occluder(0, 1);

    let c_nn = is_occluder(-1, -1);
    let c_pn = is_occluder(1, -1);
    let c_pp = is_occluder(1, 1);
    let c_np = is_occluder(-1, 1);

    // Matches the 4 quad corners emitted in order: (k, j), (k+w, j), (k+w, j+h), (k, j+h)
    [
        vertex_ao(s_neg_u, s_neg_v, c_nn),
        vertex_ao(s_pos_u, s_neg_v, c_pn),
        vertex_ao(s_pos_u, s_pos_v, c_pp),
        vertex_ao(s_neg_u, s_pos_v, c_np),
    ]
}

/// `sample` and `sample_light` take chunk-local coordinates (may lie in `-2..=33`, i.e. in neighbours).
pub fn mesh_chunk<F, L>(origin: IVec3, sample: F, sample_light: L) -> ChunkMeshData
where
    F: Fn(i32, i32, i32) -> BlockId,
    L: Fn(i32, i32, i32) -> LightData,
{
    let size = CHUNK_SIZE;
    let n = CHUNK_SIZE as i32;
    let mut opaque = MeshData::new();
    let mut translucent = MeshData::new();
    let mut mask = vec![(AIR, 0u16, [3u8; 4], NO_TEXTURE); size * size];

    // ── Pass 1: Greedy meshing of opaque full-cube voxels with Corner AO & Better Grass ──
    for d in 0..3usize {
        let u = (d + 1) % 3;
        let v = (d + 2) % 3;

        for positive in [true, false] {
            let step = if positive { 1 } else { -1 };
            let face = BlockFace::from_axis(d, positive);

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
                        let nb = sample(q[0], q[1], q[2]);
                        let visible = is_full_cube(b)
                            && (!is_full_cube(nb) || base_block_id(nb) == OAK_LEAVES);
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
                            let ao = compute_face_ao(q, u, v, &sample);

                            // Better Grass: if a vertical side face of a Grass block steps down 1 block
                            // onto another Grass block (`(x + dir.x, y - 1, z + dir.z) == GRASS`),
                            // use `GrassTop` instead of `GrassSide` for continuous green hillsides.
                            let mut tex_layer = get_face_texture(b, face, is_active_block(b));
                            if d != 1 && base_block_id(b) == GRASS {
                                let step_down = sample(q[0], q[1] - 1, q[2]);
                                if base_block_id(step_down) == GRASS {
                                    tex_layer = TextureLayer::GrassTop as u32;
                                }
                            }

                            mask[(j * n + k) as usize] = (b, fl, ao, tex_layer);
                        } else {
                            mask[(j * n + k) as usize] = (AIR, 0, [3; 4], NO_TEXTURE);
                        }
                    }
                }

                // Greedy merge faces with identical (BlockId, LightData, CornerAO, TexLayer).
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
                                mask[(j + dy) * size + k + dx] = (AIR, 0, [3; 4], NO_TEXTURE);
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
                            cell.2,
                            cell.3,
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
                } else if is_cross_foliage(base) {
                    emit_cross_foliage_voxel(&mut opaque, wp, raw, cell_light);
                } else if is_stained_glass(base) {
                    emit_stained_glass(&mut translucent, wp, x, y, z, raw, cell_light, &sample);
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
    ao: [u8; 4],
    tex_layer: u32,
) {
    let plane = slice + if positive { 1 } else { 0 };
    let (k, j, w, h) = (k as i32, j as i32, w as i32, h as i32);

    let mut normal = [0.0f32; 3];
    normal[d] = if positive { 1.0 } else { -1.0 };
    let face = BlockFace::from_axis(d, positive);
    let active = is_active_block(block);
    let color = if tex_layer != NO_TEXTURE {
        texture_tint_color(block, d)
    } else {
        block_face_color(block, d)
    };
    let mat_flag = if block_face_emission(block, face, active) > 0.0 {
        MAT_EMISSIVE
    } else {
        MAT_STANDARD
    };

    let base = data.vertices.len() as u32;
    let corners = [(k, j), (k + w, j), (k + w, j + h), (k, j + h)];
    for (idx, (cu, cv)) in corners.into_iter().enumerate() {
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
            ao_and_flags: [ao[idx] as f32, mat_flag],
            // Sentinel `-1.0` tells `voxel.wgsl` to use World-Aligned UV (`world_pos`)
            // so every 1x1x1 voxel gets a crisp unstretched 16x16 tile regardless of quad size.
            uv: [-1.0, -1.0],
            tex_layer,
        });
    }

    // AO Quad Flipping: prevent anisotropic diagonal artifacts when `ao[0] + ao[2] < ao[1] + ao[3]`.
    let flip = (ao[0] as u16 + ao[2] as u16) < (ao[1] as u16 + ao[3] as u16);
    match (positive, flip) {
        (true, false) => data
            .indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]),
        (true, true) => data
            .indices
            .extend_from_slice(&[base + 1, base + 2, base + 3, base + 1, base + 3, base]),
        (false, false) => data
            .indices
            .extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]),
        (false, true) => data
            .indices
            .extend_from_slice(&[base + 1, base + 3, base + 2, base + 1, base, base + 3]),
    }
}

/// Computes the smooth corner height in `0.0..=1.0` (`corner_height / 8.0`) for fluid `base`
/// at the grid vertex `(vx, ly, vz)` by averaging the fluid levels of the 4 columns
/// `(vx, vz)`, `(vx - 1, vz)`, `(vx, vz - 1)`, `(vx - 1, vz - 1)` meeting at that corner.
#[inline]
fn fluid_corner_height<F: Fn(i32, i32, i32) -> BlockId>(
    vx: i32,
    ly: i32,
    vz: i32,
    base: BlockId,
    sample: &F,
) -> f32 {
    let offsets = [(0, 0), (-1, 0), (0, -1), (-1, -1)];

    // If any of the 4 columns touching this corner has the same fluid above it (`ly + 1`),
    // the corner reaches full voxel height `1.0` (`8.0 / 8.0`) to join the upper fluid seamlessly.
    for (dx, dz) in offsets {
        let cx = vx + dx;
        let cz = vz + dz;
        if base_block_id(sample(cx, ly, cz)) == base
            && base_block_id(sample(cx, ly + 1, cz)) == base
        {
            return 1.0;
        }
    }

    let mut sum_levels = 0.0f32;
    let mut weight = 0.0f32;
    for (dx, dz) in offsets {
        let cell = sample(vx + dx, ly, vz + dz);
        let cell_base = base_block_id(cell);
        if cell_base == base {
            sum_levels += fluid_level_from_id(cell) as f32;
            weight += 1.0;
        } else if !is_solid_block(cell) {
            // Air / non-solid neighbor contributes level 0 so flowing streams taper smoothly.
            weight += 1.0;
        }
    }

    if weight > 0.0 {
        let corner_height = sum_levels / weight;
        (corner_height / 8.0).clamp(0.05, 1.0)
    } else {
        1.0
    }
}

/// Computes the 4 top corner heights `[h00, h10, h11, h01]` in `0.0..=1.0` for fluid voxel `(lx, ly, lz)`.
#[inline]
fn fluid_voxel_corner_heights<F: Fn(i32, i32, i32) -> BlockId>(
    lx: i32,
    ly: i32,
    lz: i32,
    base: BlockId,
    sample: &F,
) -> [f32; 4] {
    if base_block_id(sample(lx, ly + 1, lz)) == base {
        [1.0, 1.0, 1.0, 1.0]
    } else {
        [
            fluid_corner_height(lx, ly, lz, base, sample),
            fluid_corner_height(lx + 1, ly, lz, base, sample),
            fluid_corner_height(lx + 1, ly, lz + 1, base, sample),
            fluid_corner_height(lx, ly, lz + 1, base, sample),
        ]
    }
}

/// Emits a smooth corner-interpolated fluid voxel (`Water`, `Lava`, `Oil`) with sloped top surface
/// (`y_pos = base_y + corner_height / 8.0`) and internal vertical face culling.
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
    let tex_layer = if base == LAVA {
        TextureLayer::LavaStill as u32
    } else {
        NO_TEXTURE
    };
    let color = if tex_layer != NO_TEXTURE {
        [1.0, 1.0, 1.0]
    } else {
        block_face_color(raw, 1)
    };
    let mat_flag = if base == WATER {
        MAT_WATER
    } else if base == LAVA {
        MAT_EMISSIVE
    } else {
        MAT_STANDARD
    };

    let submerged = base_block_id(sample(lx, ly + 1, lz)) == base;
    let [h00, h10, h11, h01] = fluid_voxel_corner_heights(lx, ly, lz, base, sample);
    let top_wave_flag = if submerged { 0.0 } else { 1.0 };

    let bmin = world_pos.as_vec3();
    let (x0, y0, z0) = (bmin.x, bmin.y, bmin.z);
    let (x1, z1) = (x0 + 1.0, z0 + 1.0);

    let push_quad = |data: &mut MeshData,
                     positions: [[f32; 3]; 4],
                     normal: [f32; 3],
                     top_flags: [f32; 4]| {
        const QUAD_UVS: [[f32; 2]; 4] = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
        let base_idx = data.vertices.len() as u32;
        for i in 0..4 {
            data.vertices.push(Vertex {
                position: positions[i],
                normal,
                color,
                light,
                ao_and_flags: [top_flags[i], mat_flag],
                uv: QUAD_UVS[i],
                tex_layer,
            });
        }
        data.indices.extend_from_slice(&[
            base_idx,
            base_idx + 1,
            base_idx + 2,
            base_idx,
            base_idx + 2,
            base_idx + 3,
        ]);
    };

    // 1. Top sloped surface (`+Y`): emitted whenever the block above is not the same fluid.
    if !submerged {
        let tx = Vec3::new(1.0, ((h10 - h00) + (h11 - h01)) * 0.5, 0.0);
        let tz = Vec3::new(0.0, ((h01 - h00) + (h11 - h10)) * 0.5, 1.0);
        let top_n = tz.cross(tx).normalize_or_zero();
        let normal = if top_n.y > 0.01 {
            [top_n.x, top_n.y, top_n.z]
        } else {
            [0.0, 1.0, 0.0]
        };
        push_quad(
            data,
            [
                [x0, y0 + h01, z1],
                [x1, y0 + h11, z1],
                [x1, y0 + h10, z0],
                [x0, y0 + h00, z0],
            ],
            normal,
            [1.0, 1.0, 1.0, 1.0],
        );
    }

    // 2. Bottom surface (`-Y`): only when the block below is neither a full cube nor the same fluid.
    let b_below = sample(lx, ly - 1, lz);
    if !is_full_cube(b_below) && base_block_id(b_below) != base {
        push_quad(
            data,
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y0, z1],
                [x0, y0, z1],
            ],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        );
    }

    // 3. Side faces (`+X`, `-X`, `+Z`, `-Z`) with strict internal face culling:
    //    - Against a full solid cube: culled.
    //    - Against another voxel of the same fluid: drawn ONLY if our corner heights exceed the neighbor's
    //      (from the lower fluid height to the higher fluid height; zero internal walls inside fluid!).
    //    - Against air / non-fluid: drawn from `y0` (`0.0`) up to our sloped corner heights.
    let mut emit_side = |nx: i32,
                         nz: i32,
                         xa: f32,
                         za: f32,
                         xb: f32,
                         zb: f32,
                         ha: f32,
                         hb: f32,
                         nb_corner_indices: (usize, usize),
                         normal: [f32; 3]| {
        let n_raw = sample(nx, ly, nz);
        if is_full_cube(n_raw) {
            return;
        }
        if base_block_id(n_raw) == base {
            let nh = fluid_voxel_corner_heights(nx, ly, nz, base, sample);
            let nha = nh[nb_corner_indices.0];
            let nhb = nh[nb_corner_indices.1];
            if ha > nha + 1e-4 || hb > nhb + 1e-4 {
                let bot_a = y0 + nha.min(ha);
                let bot_b = y0 + nhb.min(hb);
                let bot_wave = if base_block_id(sample(nx, ly + 1, nz)) == base {
                    0.0
                } else {
                    1.0
                };
                push_quad(
                    data,
                    [
                        [xa, bot_a, za],
                        [xb, bot_b, zb],
                        [xb, y0 + hb, zb],
                        [xa, y0 + ha, za],
                    ],
                    normal,
                    [bot_wave, bot_wave, top_wave_flag, top_wave_flag],
                );
            }
        } else {
            push_quad(
                data,
                [
                    [xa, y0, za],
                    [xb, y0, zb],
                    [xb, y0 + hb, zb],
                    [xa, y0 + ha, za],
                ],
                normal,
                [0.0, 0.0, top_wave_flag, top_wave_flag],
            );
        }
    };

    // +X side: corners (x1, z1) [h11] -> (x1, z0) [h10]; neighbor (lx+1, lz) corners (0,1)=idx 3 and (0,0)=idx 0
    emit_side(lx + 1, lz, x1, z1, x1, z0, h11, h10, (3, 0), [1.0, 0.0, 0.0]);
    // -X side: corners (x0, z0) [h00] -> (x0, z1) [h01]; neighbor (lx-1, lz) corners (1,0)=idx 1 and (1,1)=idx 2
    emit_side(lx - 1, lz, x0, z0, x0, z1, h00, h01, (1, 2), [-1.0, 0.0, 0.0]);
    // +Z side: corners (x0, z1) [h01] -> (x1, z1) [h11]; neighbor (lx, lz+1) corners (0,0)=idx 0 and (1,0)=idx 1
    emit_side(lx, lz + 1, x0, z1, x1, z1, h01, h11, (0, 1), [0.0, 0.0, 1.0]);
    // -Z side: corners (x1, z0) [h10] -> (x0, z0) [h00]; neighbor (lx, lz-1) corners (1,1)=idx 2 and (0,1)=idx 3
    emit_side(lx, lz - 1, x1, z0, x0, z0, h10, h00, (2, 3), [0.0, 0.0, -1.0]);
}

/// Emits Fire (`BlockId::FIRE`) as two crossed vertical diagonal planes (X-shape)
/// from `(0,0,0)-(1,1,1)` and `(0,0,1)-(1,1,0)`, visible from both sides (`cull_mode: None`).
fn emit_fire_voxel(data: &mut MeshData, world_pos: IVec3) {
    let b = world_pos.as_vec3();
    let (x0, y0, z0) = (b.x, b.y, b.z);
    let (x1, y1, z1) = (b.x + 1.0, b.y + 1.0, b.z + 1.0);
    const FIRE_UVS: [[f32; 2]; 4] = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
    let light = [15.0, 8.0, 2.0, 15.0];

    let planes: [([[f32; 3]; 4], [f32; 3]); 2] = [
        // Diagonal (0,0,0) -> (1,1,1)
        (
            [[x0, y0, z0], [x1, y0, z1], [x1, y1, z1], [x0, y1, z0]],
            [0.707, 0.0, -0.707],
        ),
        // Diagonal (0,0,1) -> (1,1,0)
        (
            [[x0, y0, z1], [x1, y0, z0], [x1, y1, z0], [x0, y1, z1]],
            [0.707, 0.0, 0.707],
        ),
    ];

    for (corners, normal) in planes {
        let base = data.vertices.len() as u32;
        for (idx, position) in corners.into_iter().enumerate() {
            data.vertices.push(Vertex {
                position,
                normal,
                color: [1.0, 1.0, 1.0],
                light,
                ao_and_flags: [3.0, MAT_FIRE],
                uv: FIRE_UVS[idx],
                tex_layer: TextureLayer::FireLayer0 as u32,
            });
        }
        data.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

/// Emits a 2x2x10 pixel Torch post (`x: 0.4375..0.5625, y: 0.0..0.625, z: 0.4375..0.5625`)
/// textured with `torch_on.png` (`TextureLayer::TorchOn`) and `MAT_EMISSIVE` (`emission = 2.0`).
fn emit_torch(data: &mut MeshData, world_pos: IVec3) {
    let base_pos = world_pos.as_vec3();
    let light = [15.0, 12.0, 5.0, 15.0];
    let min = base_pos + Vec3::new(0.4375, 0.0, 0.4375);
    let max = base_pos + Vec3::new(0.5625, 0.625, 0.5625);
    let (x0, y0, z0) = (min.x, min.y, min.z);
    let (x1, y1, z1) = (max.x, max.y, max.z);

    // Map the center 2x10 pixel column of `torch_on.png` (`u: 7/16..9/16`, `v: 6/16..1.0`)
    // onto the 4 sides, and the glowing tip (`u: 7/16..9/16`, `v: 6/16..8/16`) onto the top cap.
    const U0: f32 = 7.0 / 16.0;
    const U1: f32 = 9.0 / 16.0;
    const V_TOP: f32 = 6.0 / 16.0;
    const V_MID: f32 = 8.0 / 16.0;
    const V_BOT: f32 = 1.0;

    let side_uvs = [[U0, V_BOT], [U1, V_BOT], [U1, V_MID], [U0, V_MID]];
    let top_uvs = [[U0, V_MID], [U1, V_MID], [U1, V_TOP], [U0, V_TOP]];

    let quads: [([[f32; 3]; 4], [f32; 3], [[f32; 2]; 4], f32); 5] = [
        // +X
        ([[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]], [1.0, 0.0, 0.0], side_uvs, MAT_STANDARD),
        // -X
        ([[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]], [-1.0, 0.0, 0.0], side_uvs, MAT_STANDARD),
        // +Y (glowing flame tip -> triggers soft Bloom!)
        ([[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]], [0.0, 1.0, 0.0], top_uvs, MAT_EMISSIVE),
        // +Z
        ([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], [0.0, 0.0, 1.0], side_uvs, MAT_STANDARD),
        // -Z
        ([[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]], [0.0, 0.0, -1.0], side_uvs, MAT_STANDARD),
    ];

    for (corners, normal, uvs, mat_flag) in quads {
        let base = data.vertices.len() as u32;
        for (idx, position) in corners.into_iter().enumerate() {
            data.vertices.push(Vertex {
                position,
                normal,
                color: [1.0, 1.0, 1.0],
                light,
                ao_and_flags: [3.0, mat_flag],
                uv: uvs[idx],
                tex_layer: TextureLayer::TorchOn as u32,
            });
        }
        data.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

/// Emits Cross-Mesh Foliage (`TallGrass`, `Rose`, `Dandelion`, `DeadBush`) as two crossed
/// diagonal planes inside the voxel, emitted double-sided so both faces render cleanly.
fn emit_cross_foliage_voxel(
    data: &mut MeshData,
    world_pos: IVec3,
    raw: BlockId,
    light: [f32; 4],
) {
    let b = world_pos.as_vec3();
    let (x0, y0, z0) = (b.x + 0.08, b.y, b.z + 0.08);
    let (x1, y1, z1) = (b.x + 0.92, b.y + 0.88, b.z + 0.92);
    const FOLIAGE_UVS: [[f32; 2]; 4] = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
    let tex_layer = get_face_texture(raw, BlockFace::PosZ, false);

    let planes: [([[f32; 3]; 4], [f32; 3]); 4] = [
        // Diagonal A front & back
        (
            [[x0, y0, z0], [x1, y0, z1], [x1, y1, z1], [x0, y1, z0]],
            [0.707, 0.0, -0.707],
        ),
        (
            [[x1, y0, z1], [x0, y0, z0], [x0, y1, z0], [x1, y1, z1]],
            [-0.707, 0.0, 0.707],
        ),
        // Diagonal B front & back
        (
            [[x0, y0, z1], [x1, y0, z0], [x1, y1, z0], [x0, y1, z1]],
            [0.707, 0.0, 0.707],
        ),
        (
            [[x1, y0, z0], [x0, y0, z1], [x0, y1, z1], [x1, y1, z0]],
            [-0.707, 0.0, -0.707],
        ),
    ];

    for (corners, normal) in planes {
        let base = data.vertices.len() as u32;
        for (idx, position) in corners.into_iter().enumerate() {
            data.vertices.push(Vertex {
                position,
                normal,
                color: [1.0, 1.0, 1.0],
                light,
                ao_and_flags: [3.0, MAT_STANDARD],
                uv: FOLIAGE_UVS[idx],
                tex_layer,
            });
        }
        data.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

/// Emits a translucent stained-glass block, culling faces shared with adjacent same-color glass or opaque cubes.
#[allow(clippy::too_many_arguments)]
fn emit_stained_glass<F: Fn(i32, i32, i32) -> BlockId>(
    data: &mut MeshData,
    world_pos: IVec3,
    lx: i32,
    ly: i32,
    lz: i32,
    raw: BlockId,
    light: [f32; 4],
    sample: &F,
) {
    let self_base = base_block_id(raw);
    let vis = |nx: i32, ny: i32, nz: i32| -> bool {
        let nb = sample(nx, ny, nz);
        !is_full_cube(nb) && base_block_id(nb) != self_base
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
    let tex_layer = get_face_texture(raw, BlockFace::PosZ, false);
    emit_box_with_custom_top_ao(
        data,
        bmin,
        bmin + Vec3::ONE,
        [1.0, 1.0, 1.0],
        light,
        3.0,
        MAT_STAINED_GLASS,
        tex_layer,
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
    let color = crate::voxel::chunk::block_color(CABLE);
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

fn emit_box(
    data: &mut MeshData,
    min: Vec3,
    max: Vec3,
    color: [f32; 3],
    light: [f32; 4],
    faces: [bool; 6],
) {
    emit_box_with_flag(data, min, max, color, light, MAT_STANDARD, faces);
}

/// Emits an axis-aligned box `[min..max]` with selective face flags `[+X, -X, +Y, -Y, +Z, -Z]`.
fn emit_box_with_flag(
    data: &mut MeshData,
    min: Vec3,
    max: Vec3,
    color: [f32; 3],
    light: [f32; 4],
    mat_flag: f32,
    faces: [bool; 6],
) {
    emit_box_with_custom_top_ao(
        data,
        min,
        max,
        color,
        light,
        3.0,
        mat_flag,
        NO_TEXTURE,
        faces,
    );
}

#[allow(clippy::too_many_arguments)]
fn emit_box_with_custom_top_ao(
    data: &mut MeshData,
    min: Vec3,
    max: Vec3,
    color: [f32; 3],
    light: [f32; 4],
    top_ao_or_mask: f32,
    mat_flag: f32,
    tex_layer: u32,
    faces: [bool; 6],
) {
    let (x0, y0, z0) = (min.x, min.y, min.z);
    let (x1, y1, z1) = (max.x, max.y, max.z);

    let quads: [(bool, [[f32; 3]; 4], [f32; 3], f32); 6] = [
        // +X
        (
            faces[0],
            [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]],
            [1.0, 0.0, 0.0],
            3.0,
        ),
        // -X
        (
            faces[1],
            [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]],
            [-1.0, 0.0, 0.0],
            3.0,
        ),
        // +Y
        (
            faces[2],
            [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]],
            [0.0, 1.0, 0.0],
            top_ao_or_mask,
        ),
        // -Y
        (
            faces[3],
            [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],
            [0.0, -1.0, 0.0],
            3.0,
        ),
        // +Z
        (
            faces[4],
            [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
            [0.0, 0.0, 1.0],
            3.0,
        ),
        // -Z
        (
            faces[5],
            [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]],
            [0.0, 0.0, -1.0],
            3.0,
        ),
    ];

    const UNIT_UVS: [[f32; 2]; 4] = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
    for (enabled, corners, normal, ao_val) in quads {
        if !enabled {
            continue;
        }
        let base = data.vertices.len() as u32;
        for (idx, position) in corners.into_iter().enumerate() {
            data.vertices.push(Vertex {
                position,
                normal,
                color,
                light,
                ao_and_flags: [ao_val, mat_flag],
                uv: UNIT_UVS[idx],
                tex_layer,
            });
        }
        data.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::chunk::{with_fluid_level, STONE};

    #[test]
    fn corner_ao_follows_voxel_occlusion_table() {
        // Both sides solid -> full occlusion (0) regardless of corner
        assert_eq!(vertex_ao(true, true, false), 0);
        assert_eq!(vertex_ao(true, true, true), 0);
        // No neighbors solid -> full brightness (3)
        assert_eq!(vertex_ao(false, false, false), 3);
        // One side or corner solid -> 2
        assert_eq!(vertex_ao(true, false, false), 2);
        assert_eq!(vertex_ao(false, false, true), 2);
        // One side + corner solid -> 1
        assert_eq!(vertex_ao(true, false, true), 1);
    }

    #[test]
    fn smooth_water_corner_averaging_and_internal_face_culling() {
        // Two adjacent water blocks in a stone channel: level 8 at x=0 and level 6 at x=1.
        let sample = |x: i32, y: i32, z: i32| -> BlockId {
            if y == 0 || z != 0 || x < 0 || x > 1 {
                if y <= 1 {
                    STONE
                } else {
                    AIR
                }
            } else if y == 1 && x == 0 {
                with_fluid_level(WATER, 8)
            } else if y == 1 && x == 1 {
                with_fluid_level(WATER, 6)
            } else {
                AIR
            }
        };

        let c0 = fluid_voxel_corner_heights(0, 1, 0, WATER, &sample);
        let c1 = fluid_voxel_corner_heights(1, 1, 0, WATER, &sample);

        // Shared edge at x = 1 has height (8 + 6) / 2 / 8 = 7.0 / 8.0 = 0.875 on both blocks!
        assert!((c0[1] - 0.875).abs() < 1e-5);
        assert!((c0[2] - 0.875).abs() < 1e-5);
        assert!((c1[0] - 0.875).abs() < 1e-5);
        assert!((c1[3] - 0.875).abs() < 1e-5);

        // Emit both water voxels and verify ONLY the 2 top sloped quads (8 vertices) are emitted
        // (zero internal vertical partition between x=0 and x=1!).
        let mut mesh = MeshData::new();
        emit_fluid_voxel(
            &mut mesh,
            IVec3::new(0, 1, 0),
            0,
            1,
            0,
            with_fluid_level(WATER, 8),
            [15.0; 4],
            &sample,
        );
        emit_fluid_voxel(
            &mut mesh,
            IVec3::new(1, 1, 0),
            1,
            1,
            0,
            with_fluid_level(WATER, 6),
            [15.0; 4],
            &sample,
        );
        assert_eq!(mesh.vertices.len(), 8);
    }

    #[test]
    fn water_at_all_chunk_corners_and_borders_meshes_without_panic() {
        let n = CHUNK_SIZE as i32;
        let pad = n + 4;
        let pad_u = pad as usize;
        let mut cells = vec![AIR; pad_u * pad_u * pad_u];

        // Place water at every extreme corner/border of the chunk (0, 31) including y=0 and y=31.
        let water_coords = [
            (0, 0, 0),
            (31, 0, 0),
            (0, 0, 31),
            (31, 0, 31),
            (0, 31, 0),
            (31, 31, 31),
        ];
        for (x, y, z) in water_coords {
            let idx = (x + 2) as usize
                + (y + 2) as usize * pad_u
                + (z + 2) as usize * pad_u * pad_u;
            cells[idx] = with_fluid_level(WATER, 8);
        }

        let mesh = mesh_chunk(
            IVec3::ZERO,
            |x, y, z| {
                let (Some(ix), Some(iy), Some(iz)) = (
                    usize::try_from(x + 2).ok().filter(|&v| v < pad_u),
                    usize::try_from(y + 2).ok().filter(|&v| v < pad_u),
                    usize::try_from(z + 2).ok().filter(|&v| v < pad_u),
                ) else {
                    return AIR;
                };
                cells.get(ix + iy * pad_u + iz * pad_u * pad_u).copied().unwrap_or(AIR)
            },
            |_x, _y, _z| 0,
        );
        assert!(!mesh.translucent.vertices.is_empty());
        assert!(!mesh.translucent.indices.is_empty());
    }

    #[test]
    fn better_grass_uses_grass_top_on_1_block_step_and_grass_side_on_cliffs() {
        // Grass block at (5, 5, 5):
        //   +X step-down (6, 4, 5) = GRASS -> +X face must use GrassTop (2)
        //   -X step-down (4, 4, 5) = AIR   -> -X face must use GrassSide (3)
        let mesh = mesh_chunk(
            IVec3::ZERO,
            |x, y, z| {
                if (x, y, z) == (5, 5, 5) || (x, y, z) == (6, 4, 5) {
                    GRASS
                } else {
                    AIR
                }
            },
            |_x, _y, _z| 15,
        );

        let px_layers: Vec<u32> = mesh
            .opaque
            .vertices
            .iter()
            .filter(|v| v.normal[0] > 0.5 && (v.position[0] - 6.0).abs() < 0.01)
            .map(|v| v.tex_layer)
            .collect();
        let nx_layers: Vec<u32> = mesh
            .opaque
            .vertices
            .iter()
            .filter(|v| v.normal[0] < -0.5 && (v.position[0] - 5.0).abs() < 0.01)
            .map(|v| v.tex_layer)
            .collect();

        assert!(!px_layers.is_empty());
        assert!(px_layers.iter().all(|&l| l == TextureLayer::GrassTop as u32));
        assert!(!nx_layers.is_empty());
        assert!(nx_layers.iter().all(|&l| l == TextureLayer::GrassSide as u32));
    }
}
