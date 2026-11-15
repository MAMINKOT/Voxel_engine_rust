// Voxel raycasting: Amanatides & Woo "A Fast Voxel Traversal Algorithm" with sub-block AABB
// intersection for thin cables.

use glam::{IVec3, Vec3};

use crate::voxel::chunk::{
    is_energy_connectable, is_fluid, BlockId, AIR, CABLE, CABLE_MAX, CABLE_MIN, TORCH,
};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RaycastHit {
    /// The solid voxel that was hit.
    pub block_pos: IVec3,
    /// Normal of the face that was entered (points back towards the ray origin).
    /// `block_pos + face_normal` is the empty cell where a new block would be placed.
    pub face_normal: IVec3,
    /// World-space bounding box of the hit shape (for wireframe highlight).
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
}

/// Casts a ray through full-cube voxels and returns the first hit within `max_dist`.
#[cfg(test)]
pub fn raycast<F: Fn(IVec3) -> bool>(
    origin: Vec3,
    dir: Vec3,
    max_dist: f32,
    is_solid: F,
) -> Option<RaycastHit> {
    raycast_blocks(origin, dir, max_dist, |p| if is_solid(p) { 1 } else { AIR })
}

/// Casts a ray that tests full-cube blocks as 1x1x1 and `CABLE` blocks against their
/// sub-block core (`0.35..0.65`) + connected branch arms.
pub fn raycast_blocks<F: Fn(IVec3) -> BlockId>(
    origin: Vec3,
    dir: Vec3,
    max_dist: f32,
    get_block: F,
) -> Option<RaycastHit> {
    let dir = dir.normalize_or_zero();
    if dir == Vec3::ZERO {
        return None;
    }

    let mut pos = origin.floor().as_ivec3();
    let mut step = IVec3::ZERO;
    let mut t_max = Vec3::splat(f32::INFINITY);
    let mut t_delta = Vec3::splat(f32::INFINITY);

    for i in 0..3 {
        if dir[i] > 0.0 {
            step[i] = 1;
            t_delta[i] = 1.0 / dir[i];
            t_max[i] = (pos[i] as f32 + 1.0 - origin[i]) / dir[i];
        } else if dir[i] < 0.0 {
            step[i] = -1;
            t_delta[i] = -1.0 / dir[i];
            t_max[i] = (origin[i] - pos[i] as f32) / -dir[i];
        }
    }

    // Also test the starting voxel if it contains a cable and the eye is outside the core.
    if get_block(pos) == CABLE {
        if let Some(hit) = intersect_cable(origin, dir, max_dist, pos, &get_block) {
            return Some(hit);
        }
    }

    loop {
        let axis = if t_max.x < t_max.y {
            if t_max.x < t_max.z { 0 } else { 2 }
        } else if t_max.y < t_max.z {
            1
        } else {
            2
        };

        if t_max[axis] > max_dist {
            return None;
        }
        pos[axis] += step[axis];
        t_max[axis] += t_delta[axis];

        let id = get_block(pos);
        if id == AIR || is_fluid(id) {
            continue;
        }
        if id == CABLE {
            if let Some(hit) = intersect_cable(origin, dir, max_dist, pos, &get_block) {
                return Some(hit);
            }
            continue;
        }
        if id == TORCH {
            let base = pos.as_vec3();
            let bmin = base + Vec3::new(0.40, 0.0, 0.40);
            let bmax = base + Vec3::new(0.60, 0.66, 0.60);
            if let Some((t, face_normal)) = ray_aabb(origin, dir, bmin, bmax) {
                if t <= max_dist {
                    return Some(RaycastHit {
                        block_pos: pos,
                        face_normal,
                        bounds_min: bmin,
                        bounds_max: bmax,
                    });
                }
            }
            continue;
        }

        let mut face_normal = IVec3::ZERO;
        face_normal[axis] = -step[axis];
        let bounds_min = pos.as_vec3();
        let bounds_max = bounds_min + Vec3::ONE;
        return Some(RaycastHit { block_pos: pos, face_normal, bounds_min, bounds_max });
    }
}

/// Tests a ray against the cable's central core `[0.35..0.65]³` and any connected arms.
fn intersect_cable<F: Fn(IVec3) -> BlockId>(
    origin: Vec3,
    dir: Vec3,
    max_dist: f32,
    pos: IVec3,
    get_block: &F,
) -> Option<RaycastHit> {
    let base = pos.as_vec3();
    let c0 = CABLE_MIN;
    let c1 = CABLE_MAX;

    let mut boxes: Vec<(Vec3, Vec3)> = Vec::with_capacity(7);
    // Central core
    boxes.push((base + Vec3::splat(c0), base + Vec3::splat(c1)));

    let mut env_min = Vec3::splat(c0);
    let mut env_max = Vec3::splat(c1);

    if is_energy_connectable(get_block(pos + IVec3::X)) {
        boxes.push((base + Vec3::new(c1, c0, c0), base + Vec3::new(1.0, c1, c1)));
        env_max.x = 1.0;
    }
    if is_energy_connectable(get_block(pos - IVec3::X)) {
        boxes.push((base + Vec3::new(0.0, c0, c0), base + Vec3::new(c0, c1, c1)));
        env_min.x = 0.0;
    }
    if is_energy_connectable(get_block(pos + IVec3::Y)) {
        boxes.push((base + Vec3::new(c0, c1, c0), base + Vec3::new(c1, 1.0, c1)));
        env_max.y = 1.0;
    }
    if is_energy_connectable(get_block(pos - IVec3::Y)) {
        boxes.push((base + Vec3::new(c0, 0.0, c0), base + Vec3::new(c1, c0, c1)));
        env_min.y = 0.0;
    }
    if is_energy_connectable(get_block(pos + IVec3::Z)) {
        boxes.push((base + Vec3::new(c0, c0, c1), base + Vec3::new(c1, c1, 1.0)));
        env_max.z = 1.0;
    }
    if is_energy_connectable(get_block(pos - IVec3::Z)) {
        boxes.push((base + Vec3::new(c0, c0, 0.0), base + Vec3::new(c1, c1, c0)));
        env_min.z = 0.0;
    }

    let mut best_t = max_dist;
    let mut best_normal = None;
    for (bmin, bmax) in boxes {
        if let Some((t, normal)) = ray_aabb(origin, dir, bmin, bmax) {
            if t <= best_t {
                best_t = t;
                best_normal = Some(normal);
            }
        }
    }

    best_normal.map(|face_normal| RaycastHit {
        block_pos: pos,
        face_normal,
        bounds_min: base + env_min,
        bounds_max: base + env_max,
    })
}

/// Slab-method Ray-AABB intersection. Returns `(t_enter, outward_face_normal)`.
fn ray_aabb(origin: Vec3, dir: Vec3, min: Vec3, max: Vec3) -> Option<(f32, IVec3)> {
    let mut t_min = 0.0_f32;
    let mut t_max = f32::INFINITY;
    let mut hit_axis = 0usize;
    let mut hit_sign = -1i32;

    for i in 0..3 {
        if dir[i].abs() < 1e-6 {
            if origin[i] < min[i] || origin[i] > max[i] {
                return None;
            }
        } else {
            let inv = 1.0 / dir[i];
            let (mut t1, mut t2, sign) = if dir[i] >= 0.0 {
                ((min[i] - origin[i]) * inv, (max[i] - origin[i]) * inv, -1)
            } else {
                ((max[i] - origin[i]) * inv, (min[i] - origin[i]) * inv, 1)
            };
            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
            }
            if t1 > t_min {
                t_min = t1;
                hit_axis = i;
                hit_sign = sign;
            }
            t_max = t_max.min(t2);
            if t_min > t_max {
                return None;
            }
        }
    }

    let mut normal = IVec3::ZERO;
    normal[hit_axis] = hit_sign;
    Some((t_min, normal))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hits_block_along_x_with_correct_face() {
        let hit = raycast(Vec3::new(0.5, 0.5, 0.5), Vec3::X, 6.0, |p| p == IVec3::new(3, 0, 0)).unwrap();
        assert_eq!(hit.block_pos, IVec3::new(3, 0, 0));
        assert_eq!(hit.face_normal, IVec3::new(-1, 0, 0));
    }

    #[test]
    fn hits_floor_looking_down_and_negative_directions() {
        let hit = raycast(Vec3::new(0.5, 3.5, 0.5), -Vec3::Y, 6.0, |p| p.y <= 0).unwrap();
        assert_eq!(hit.block_pos, IVec3::new(0, 0, 0));
        assert_eq!(hit.face_normal, IVec3::new(0, 1, 0));

        let hit = raycast(Vec3::new(-0.5, 0.5, -0.5), Vec3::new(-1.0, 0.0, -1.0), 6.0, |p| p.x <= -3).unwrap();
        assert_eq!(hit.face_normal, IVec3::new(1, 0, 0));
    }

    #[test]
    fn respects_max_distance() {
        assert!(raycast(Vec3::new(0.5, 0.5, 0.5), Vec3::X, 6.0, |p| p.x == 8).is_none());
        assert!(raycast(Vec3::new(0.5, 0.5, 0.5), Vec3::X, 6.0, |p| p.x == 6).is_some());
    }

    #[test]
    fn cable_subblock_raycast_hits_core_and_skips_empty_corner() {
        let get = |p: IVec3| if p == IVec3::new(2, 0, 0) { CABLE } else { AIR };
        // Center ray hits the [0.35..0.65] cable core.
        let hit = raycast_blocks(Vec3::new(0.5, 0.5, 0.5), Vec3::X, 6.0, get).unwrap();
        assert_eq!(hit.block_pos, IVec3::new(2, 0, 0));
        assert_eq!(hit.face_normal, IVec3::new(-1, 0, 0));
        // Corner ray (y = 0.1, z = 0.1) passes through the empty space around the cable.
        assert!(raycast_blocks(Vec3::new(0.5, 0.1, 0.1), Vec3::X, 6.0, get).is_none());
    }
}
