// Explosions & Fire simulation (20 TPS):
//   • `PrimedTnt`: ignited TNT entity that flashes white for 60 ticks (3.0 s) before detonating.
//   • Detonation: sphere of radius R = 4.5 blocks destroys all voxels except `BEDROCK` and
//     `OBSIDIAN`, drops 40% of destroyed blocks as `ItemDrop`s, chain-ignites nearby `TNT`,
//     and applies a radial shockwave impulse to the player.
//   • `FIRE`: burns for ~100 ticks, spreads to adjacent flammable blocks (`WOOD`, `OIL`),
//     and ignites adjacent `TNT` blocks on contact.

use std::collections::HashMap;

use glam::{IVec3, Vec3};

use crate::voxel::chunk::{
    is_blast_resistant, is_flammable, is_fluid, is_solid_block, BlockId, AIR, FIRE, LAVA, TNT,
};
use crate::voxel::mesher::Vertex;

pub const TNT_FUSE_TICKS: u16 = 60; // 3.0 seconds at 20 TPS
pub const EXPLOSION_RADIUS: f32 = 4.5;
pub const SHOCKWAVE_RADIUS: f32 = 8.0;
pub const FIRE_LIFETIME_TICKS: u16 = 100;

const NEIGHBORS_6: [IVec3; 6] = [
    IVec3::new(1, 0, 0),
    IVec3::new(-1, 0, 0),
    IVec3::new(0, 1, 0),
    IVec3::new(0, -1, 0),
    IVec3::new(0, 0, 1),
    IVec3::new(0, 0, -1),
];

#[derive(Clone, Debug)]
pub struct PrimedTnt {
    pub position: Vec3,
    pub velocity: Vec3,
    pub fuse_ticks: u16,
}

pub struct DetonationEvent {
    pub center: Vec3,
    pub destroyed_blocks: Vec<(IVec3, BlockId)>,
    pub spawned_drops: Vec<(Vec3, BlockId)>,
    pub player_impulse: Vec3,
}

pub struct ExplosionSystem {
    pub primed: Vec<PrimedTnt>,
    fires: HashMap<IVec3, u16>,
    rng_state: u32,
}

impl ExplosionSystem {
    pub fn new() -> Self {
        Self {
            primed: Vec::new(),
            fires: HashMap::new(),
            rng_state: 0x7a4f_91c3,
        }
    }

    pub fn primed_count(&self) -> usize {
        self.primed.len()
    }

    /// Ignites a TNT block at `block_pos` with the standard 60-tick (3-second) fuse.
    pub fn ignite_tnt(&mut self, block_pos: IVec3, fuse_ticks: u16) {
        let center = block_pos.as_vec3() + Vec3::splat(0.5);
        self.primed.push(PrimedTnt {
            position: center,
            velocity: Vec3::new(0.0, 2.6, 0.0),
            fuse_ticks,
        });
    }

    /// Notifies the fire tracker when a block in the world changes.
    pub fn on_block_changed(&mut self, pos: IVec3, new_id: BlockId) {
        if new_id == FIRE {
            self.fires.insert(pos, FIRE_LIFETIME_TICKS);
        } else {
            self.fires.remove(&pos);
        }
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.rng_state = x;
        x
    }

    /// Physics update for `PrimedTnt` (3D arc trajectory, gravity, wall bounce & floor slide) run every frame.
    pub fn update_physics<F: Fn(IVec3) -> BlockId>(&mut self, dt: f32, get_block: F) {
        const HALF: f32 = 0.48;
        for tnt in &mut self.primed {
            tnt.velocity.y = (tnt.velocity.y - 22.0 * dt).max(-35.0);

            // Axis X
            let dx = tnt.velocity.x * dt;
            if dx != 0.0 {
                let probe_x = tnt.position.x + dx + HALF * dx.signum();
                let cell = Vec3::new(probe_x, tnt.position.y, tnt.position.z)
                    .floor()
                    .as_ivec3();
                if is_solid_block(get_block(cell)) {
                    tnt.velocity.x = -tnt.velocity.x * 0.45;
                } else {
                    tnt.position.x += dx;
                }
            }

            // Axis Y
            let dy = tnt.velocity.y * dt;
            if dy != 0.0 {
                let probe_y = tnt.position.y + dy + HALF * dy.signum();
                let cell = Vec3::new(tnt.position.x, probe_y, tnt.position.z)
                    .floor()
                    .as_ivec3();
                if is_solid_block(get_block(cell)) {
                    if dy < 0.0 {
                        tnt.position.y = cell.y as f32 + 1.0 + HALF;
                        tnt.velocity.y = 0.0;
                        tnt.velocity.x *= 0.75;
                        tnt.velocity.z *= 0.75;
                    } else {
                        tnt.position.y = cell.y as f32 - HALF - 1e-3;
                        tnt.velocity.y = 0.0;
                    }
                } else {
                    tnt.position.y += dy;
                }
            }

            // Axis Z
            let dz = tnt.velocity.z * dt;
            if dz != 0.0 {
                let probe_z = tnt.position.z + dz + HALF * dz.signum();
                let cell = Vec3::new(tnt.position.x, tnt.position.y, probe_z)
                    .floor()
                    .as_ivec3();
                if is_solid_block(get_block(cell)) {
                    tnt.velocity.z = -tnt.velocity.z * 0.45;
                } else {
                    tnt.position.z += dz;
                }
            }
        }
    }

    /// Fixed 20 TPS tick:
    ///   • Steps fire blocks (spreading to wood/oil, igniting adjacent TNT, or burning out).
    ///   • Decrements fuse on all `PrimedTnt` and computes `DetonationEvent`s when `fuse_ticks == 0`.
    pub fn tick<F: Fn(IVec3) -> BlockId>(
        &mut self,
        player_center: Vec3,
        get_block: F,
    ) -> (Vec<(IVec3, BlockId)>, Vec<DetonationEvent>) {
        let mut block_changes: Vec<(IVec3, BlockId)> = Vec::new();
        let mut newly_ignited_tnt: Vec<IVec3> = Vec::new();

        // 1. Step active Fire blocks.
        let fire_positions: Vec<IVec3> = self.fires.keys().copied().collect();
        for pos in fire_positions {
            if get_block(pos) != FIRE {
                self.fires.remove(&pos);
                continue;
            }
            // Check neighbors for TNT or flammable Wood/Oil.
            for d in NEIGHBORS_6 {
                let np = pos + d;
                let nb = get_block(np);
                if nb == TNT && !newly_ignited_tnt.contains(&np) {
                    newly_ignited_tnt.push(np);
                } else if is_flammable(nb) && (self.next_u32() % 100) < 12 {
                    // Consume flammable wood/oil into fire!
                    block_changes.push((np, FIRE));
                    self.fires.insert(np, FIRE_LIFETIME_TICKS);
                }
            }

            if let Some(rem) = self.fires.get_mut(&pos) {
                *rem = rem.saturating_sub(1);
                if *rem == 0 {
                    self.fires.remove(&pos);
                    block_changes.push((pos, AIR));
                }
            }
        }

        for tnt_pos in newly_ignited_tnt {
            block_changes.push((tnt_pos, AIR));
            self.ignite_tnt(tnt_pos, TNT_FUSE_TICKS);
        }

        // 2. Step PrimedTNT fuses & detonate when fuse reaches 0.
        let mut detonations = Vec::new();
        let mut i = 0;
        while i < self.primed.len() {
            self.primed[i].fuse_ticks = self.primed[i].fuse_ticks.saturating_sub(1);
            if self.primed[i].fuse_ticks == 0 {
                let tnt = self.primed.swap_remove(i);
                let mut chain_tnt: Vec<(IVec3, u16)> = Vec::new();
                let event = self.compute_detonation(
                    tnt.position,
                    player_center,
                    &get_block,
                    &mut chain_tnt,
                );
                // Convert unignited TNT blocks within shockwave radius into PrimedTnt entities
                for (pos, fuse) in chain_tnt {
                    self.primed.push(PrimedTnt {
                        position: pos.as_vec3() + Vec3::splat(0.5),
                        velocity: Vec3::ZERO,
                        fuse_ticks: fuse,
                    });
                }
                // Apply radial + upward shockwave impulse to all PrimedTnt within R = 8.0 m
                for other in &mut self.primed {
                    let diff = other.position - event.center;
                    let dist = diff.length();
                    if dist < SHOCKWAVE_RADIUS {
                        let dir = if dist > 1e-4 { diff / dist } else { Vec3::Y };
                        let force = (1.0 - dist / SHOCKWAVE_RADIUS).max(0.0) * 15.0;
                        other.velocity += dir * force + Vec3::new(0.0, 4.0, 0.0);
                    }
                }
                detonations.push(event);
            } else {
                i += 1;
            }
        }

        (block_changes, detonations)
    }

    fn compute_detonation<F: Fn(IVec3) -> BlockId>(
        &mut self,
        center: Vec3,
        player_center: Vec3,
        get_block: &F,
        chain_tnt: &mut Vec<(IVec3, u16)>,
    ) -> DetonationEvent {
        let r = EXPLOSION_RADIUS;
        let r_sq = r * r;
        let sw_r = SHOCKWAVE_RADIUS;
        let sw_r_sq = sw_r * sw_r;
        let min_p = (center - Vec3::splat(sw_r)).floor().as_ivec3();
        let max_p = (center + Vec3::splat(sw_r)).ceil().as_ivec3();

        let mut destroyed_blocks = Vec::new();
        let mut spawned_drops = Vec::new();

        for z in min_p.z..=max_p.z {
            for y in min_p.y.max(1)..=max_p.y {
                for x in min_p.x..=max_p.x {
                    let bp = IVec3::new(x, y, z);
                    let block_center = bp.as_vec3() + Vec3::splat(0.5);
                    let dist_sq = block_center.distance_squared(center);
                    if dist_sq > sw_r_sq {
                        continue;
                    }
                    let id = get_block(bp);
                    if id == TNT {
                        // Any unignited TNT within SHOCKWAVE_RADIUS (8.0m) converts to PrimedTnt
                        // with a random short fuse (10..=30 ticks) and gets launched by the blast!
                        destroyed_blocks.push((bp, id));
                        let short_fuse = 10 + (self.next_u32() % 21) as u16;
                        chain_tnt.push((bp, short_fuse));
                        continue;
                    }
                    if dist_sq > r_sq || id == AIR || is_blast_resistant(id) {
                        continue;
                    }
                    destroyed_blocks.push((bp, id));
                    if !is_fluid(id) && id != FIRE {
                        // 40% chance to spawn an ItemDrop
                        if (self.next_u32() % 100) < 40 {
                            spawned_drops.push((block_center, id));
                        }
                    }
                }
            }
        }

        // Player shockwave impulse if within SHOCKWAVE_RADIUS
        let to_player = player_center - center;
        let dist = to_player.length();
        let player_impulse = if dist < SHOCKWAVE_RADIUS {
            let falloff = (1.0 - dist / SHOCKWAVE_RADIUS).clamp(0.15, 1.0);
            let dir = if dist > 0.05 {
                (to_player / dist + Vec3::Y * 0.45).normalize()
            } else {
                Vec3::Y
            };
            dir * (20.0 * falloff)
        } else {
            Vec3::ZERO
        };

        DetonationEvent {
            center,
            destroyed_blocks,
            spawned_drops,
            player_impulse,
        }
    }

    /// Checks whether a newly placed or existing `TNT` block at `pos` touches `FIRE` or `LAVA`.
    pub fn should_auto_ignite<F: Fn(IVec3) -> BlockId>(pos: IVec3, get_block: F) -> bool {
        NEIGHBORS_6
            .iter()
            .any(|&d| matches!(get_block(pos + d), FIRE | LAVA))
    }

    /// Appends 3D flashing cubes for all active `PrimedTnt` entities to a dynamic mesh.
    pub fn append_mesh(&self, vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>) {
        for tnt in &self.primed {
            // Flash white every 4 ticks (hissing/blinking effect), clamped <= 1.0 (non-emissive)
            let flash_white = (tnt.fuse_ticks / 4) % 2 == 0;
            let side_color = if flash_white {
                [1.0, 1.0, 1.0]
            } else {
                [0.92, 0.20, 0.16]
            };
            let top_color = if flash_white {
                [1.0, 1.0, 1.0]
            } else {
                [0.95, 0.90, 0.85]
            };

            // Slightly pulsing scale as fuse nears zero
            let pulse = 1.0 + 0.06 * ((tnt.fuse_ticks as f32 * 0.6).sin().abs());
            let half = 0.48 * pulse;
            let min = tnt.position - Vec3::splat(half);
            let max = tnt.position + Vec3::splat(half);
            let (x0, y0, z0) = (min.x, min.y, min.z);
            let (x1, y1, z1) = (max.x, max.y, max.z);

            let faces: [([[f32; 3]; 4], [f32; 3], [f32; 3]); 6] = [
                (
                    [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]],
                    [1.0, 0.0, 0.0],
                    side_color,
                ),
                (
                    [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]],
                    [-1.0, 0.0, 0.0],
                    side_color,
                ),
                (
                    [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]],
                    [0.0, 1.0, 0.0],
                    top_color,
                ),
                (
                    [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],
                    [0.0, -1.0, 0.0],
                    top_color,
                ),
                (
                    [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
                    [0.0, 0.0, 1.0],
                    side_color,
                ),
                (
                    [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]],
                    [0.0, 0.0, -1.0],
                    side_color,
                ),
            ];

            const TNT_UVS: [[f32; 2]; 4] = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
            let tex_layer = if flash_white {
                crate::voxel::blocks::NO_TEXTURE
            } else {
                crate::voxel::blocks::TextureLayer::TntSide as u32
            };
            let face_tint = if flash_white { side_color } else { [1.0, 1.0, 1.0] };
            for (corners, normal, _color) in faces {
                let base = vertices.len() as u32;
                for (idx, position) in corners.into_iter().enumerate() {
                    vertices.push(Vertex {
                        position,
                        normal,
                        color: face_tint,
                        light: [15.0, 15.0, 15.0, 15.0],
                        ao_and_flags: [3.0, 0.0],
                        uv: TNT_UVS[idx],
                        tex_layer,
                    });
                }
                indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::chunk::{BEDROCK, OBSIDIAN, STONE};

    #[test]
    fn primed_tnt_detonates_after_60_ticks_destroying_stone_and_sparing_obsidian_bedrock() {
        let mut sys = ExplosionSystem::new();
        let center_pos = IVec3::new(0, 10, 0);
        sys.ignite_tnt(center_pos, TNT_FUSE_TICKS);

        let block_at = |p: IVec3| -> BlockId {
            if p == IVec3::new(1, 10, 0) {
                OBSIDIAN
            } else if p == IVec3::new(-1, 10, 0) {
                BEDROCK
            } else {
                STONE
            }
        };

        let mut det_events = Vec::new();
        for _ in 0..TNT_FUSE_TICKS {
            let (_changes, mut evs) = sys.tick(Vec3::new(2.0, 10.0, 0.0), block_at);
            det_events.append(&mut evs);
        }

        assert_eq!(det_events.len(), 1);
        let ev = &det_events[0];
        // Stone blocks within R=4.5 are destroyed
        assert!(ev
            .destroyed_blocks
            .iter()
            .any(|&(p, _)| p == IVec3::new(2, 10, 0)));
        // Obsidian and Bedrock are spared
        assert!(!ev
            .destroyed_blocks
            .iter()
            .any(|&(p, _)| p == IVec3::new(1, 10, 0)));
        assert!(!ev
            .destroyed_blocks
            .iter()
            .any(|&(p, _)| p == IVec3::new(-1, 10, 0)));
        // Player at distance ~1.5 receives a strong outward + upward shockwave impulse
        assert!(ev.player_impulse.length() > 5.0);
        assert!(ev.player_impulse.x > 0.0);
    }

    #[test]
    fn tnt_shockwave_converts_and_launches_nearby_tnt_within_8m() {
        let mut sys = ExplosionSystem::new();
        let center_pos = IVec3::new(0, 10, 0);
        let neighbor_tnt = IVec3::new(6, 10, 0); // 6m away: outside R=4.5 destruction, inside R=8.0 shockwave
        sys.ignite_tnt(center_pos, 1);

        let block_at = |p: IVec3| -> BlockId {
            if p == neighbor_tnt {
                TNT
            } else {
                AIR
            }
        };

        let (_changes, evs) = sys.tick(Vec3::new(20.0, 10.0, 0.0), block_at);
        assert_eq!(evs.len(), 1);
        assert_eq!(sys.primed.len(), 1);
        let launched = &sys.primed[0];
        assert!((10..=30).contains(&launched.fuse_ticks));
        // Must have positive +X velocity (pushed away from x=0) and upward +Y velocity
        assert!(launched.velocity.x > 2.0);
        assert!(launched.velocity.y >= 4.0);
    }
}
