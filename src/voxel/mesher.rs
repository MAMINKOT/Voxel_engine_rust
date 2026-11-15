// Greedy meshing for opaque full-cube blocks + Corner Ambient Occlusion (Smooth Lighting)
// with anisotropic quad flipping + multipart sub-block meshing for cables & torches +
// proportional-height fluid meshing (`height = (level as f32) / 8.0`) + separate translucent
// mesh buffer (`Water`, `Fire`, `PurpleStainedGlass`) + RGB/Sun per-vertex `LightData`.

use glam::{IVec3, Vec3};

use crate::voxel::chunk::{
    base_block_id, block_color, block_face_color, fluid_level_from_id, is_active_block,
    is_energy_connectable, is_fluid, is_full_cube, is_solid_block, is_translucent, BlockId, AIR,
    CABLE, CABLE_MAX, CABLE_MIN, CHUNK_SIZE, FIRE, LAVA, PURPLE_STAINED_GLASS, TORCH, WATER,
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
    /// `[ao_level (0.0..=3.0), material_flag (0=standard, 1=water, 2=glass, 3=fire)]`.
    pub ao_and_flags: [f32; 2],
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
    let mut mask = vec![(AIR, 0u16, [3u8; 4]); size * size];

    // ── Pass 1: Greedy meshing of opaque full-cube voxels with Corner AO ──
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
                            let ao = compute_face_ao(q, u, v, &sample);
                            mask[(j * n + k) as usize] = (b, fl, ao);
                        } else {
                            mask[(j * n + k) as usize] = (AIR, 0, [3; 4]);
                        }
                    }
                }

                // Greedy merge faces with identical (BlockId, LightData, CornerAO).
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
                                mask[(j + dy) * size + k + dx] = (AIR, 0, [3; 4]);
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
    ao: [u8; 4],
) {
    let plane = slice + if positive { 1 } else { 0 };
    let (k, j, w, h) = (k as i32, j as i32, w as i32, h as i32);

    let mut normal = [0.0f32; 3];
    normal[d] = if positive { 1.0 } else { -1.0 };
    let color = block_face_color(block, d);
    let mat_flag = if is_active_block(block) {
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
        if is_full_cube(n_raw) || n_base == base {
            // Never emit internal vertical faces between two adjacent blocks of the same fluid
            // so underwater riverbeds/pools have zero internal grid lines!
            return false;
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

    let mat_flag = if base == WATER {
        MAT_WATER
    } else if base == LAVA {
        MAT_EMISSIVE
    } else {
        MAT_STANDARD
    };

    // Shoreline contact mask (bits 0..3 = +X, -X, +Z, -Z): set ONLY when the top water surface
    // (`show_top`) directly touches a solid land block at `ly` whose top at `ly + 1` is not submerged.
    let mut shore_mask: u8 = 0;
    if base == WATER && show_top {
        let is_dry_shore = |nx: i32, nz: i32| -> bool {
            let side = sample(nx, ly, nz);
            let above_side = sample(nx, ly + 1, nz);
            is_solid_block(side)
                && base_block_id(side) != PURPLE_STAINED_GLASS
                && !is_fluid(above_side)
        };
        if is_dry_shore(lx + 1, lz) {
            shore_mask |= 1;
        }
        if is_dry_shore(lx - 1, lz) {
            shore_mask |= 2;
        }
        if is_dry_shore(lx, lz + 1) {
            shore_mask |= 4;
        }
        if is_dry_shore(lx, lz - 1) {
            shore_mask |= 8;
        }
    }

    let bmin = world_pos.as_vec3();
    let bmax = bmin + Vec3::new(1.0, h, 1.0);
    emit_box_with_custom_top_ao(
        data,
        bmin,
        bmax,
        color,
        light,
        shore_mask as f32,
        mat_flag,
        [show_px, show_nx, show_top, show_bottom, show_pz, show_nz],
    );
}

/// Emits a glowing translucent fire sub-block (`[0.12..0.88]` on X/Z, `[0.0..0.82]` on Y).
fn emit_fire_voxel(data: &mut MeshData, world_pos: IVec3) {
    let base_pos = world_pos.as_vec3();
    let color = block_face_color(FIRE, 1);
    emit_box_with_flag(
        data,
        base_pos + Vec3::new(0.12, 0.0, 0.12),
        base_pos + Vec3::new(0.88, 0.82, 0.88),
        color,
        [15.0, 8.0, 2.0, 15.0],
        MAT_FIRE,
        [true, true, true, false, true, true],
    );
}

/// Emits a Torch sub-block (`[0.42..0.58]` wooden stick + `[0.40..0.60]` glowing flame tip).
fn emit_torch(data: &mut MeshData, world_pos: IVec3) {
    let base_pos = world_pos.as_vec3();
    let light = [15.0, 12.0, 5.0, 15.0];
    // Wooden handle (non-emissive)
    emit_box(
        data,
        base_pos + Vec3::new(0.42, 0.0, 0.42),
        base_pos + Vec3::new(0.58, 0.50, 0.58),
        [0.52, 0.36, 0.20],
        light,
        [true, true, false, true, true, true],
    );
    // Glowing flame head (emissive -> triggers Bloom!)
    emit_box_with_flag(
        data,
        base_pos + Vec3::new(0.40, 0.50, 0.40),
        base_pos + Vec3::new(0.60, 0.66, 0.60),
        block_face_color(TORCH, 1),
        light,
        MAT_EMISSIVE,
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
    emit_box_with_flag(
        data,
        bmin,
        bmin + Vec3::ONE,
        block_face_color(PURPLE_STAINED_GLASS, 1),
        light,
        MAT_STAINED_GLASS,
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
    emit_box_with_custom_top_ao(data, min, max, color, light, 3.0, mat_flag, faces);
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

    for (enabled, corners, normal, ao_val) in quads {
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
                ao_and_flags: [ao_val, mat_flag],
            });
        }
        data.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
