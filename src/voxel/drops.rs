// Item drops: tiny physical items that fall, rest on voxels, get pulled in by a magnet and
// are picked up into the hotbar.

use glam::{IVec3, Mat3, Vec3};

use crate::inventory::{PlayerInventory, MAX_STACK};
use crate::voxel::chunk::{block_color, BlockId};
use crate::voxel::mesher::Vertex;
use crate::voxel::world::World;

pub const MAX_DROPS: usize = 512;
const MERGE_DISTANCE: f32 = 1.2;
/// Merge pass interval: 10 logic ticks (0.5 s at 20 TPS).
const MERGE_INTERVAL_SECS: f32 = 0.5;

const GRAVITY: f32 = -20.0;
const TERMINAL_VELOCITY: f32 = -30.0;
/// Half the size of the 0.25 cube.
const HALF_SIZE: f32 = 0.125;
const MAGNET_RANGE: f32 = 1.8;
const PICKUP_RANGE: f32 = 0.5;
/// A fresh drop can't be pulled in for a moment, so the toss is visible.
const PICKUP_DELAY: f32 = 0.4;
const MAX_LIFETIME: f32 = 300.0;
const VOID_Y: f32 = -20.0;
const MAX_STEP: f32 = 1.0 / 60.0;

pub struct ItemDrop {
    pub position: Vec3, // centre of the cube
    pub velocity: Vec3,
    pub block_id: BlockId,
    pub count: u16,
    /// Seconds since spawn (also drives the spin/levitation animation).
    pub lifetime: f32,
}

pub struct Drops {
    items: Vec<ItemDrop>,
    rng: u32,
    merge_accum: f32,
}

impl Drops {
    pub fn new() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0x1234_5678);
        Self {
            items: Vec::new(),
            rng: seed | 1,
            merge_accum: 0.0,
        }
    }

    pub fn items(&self) -> &[ItemDrop] {
        &self.items
    }

    /// xorshift32, uniform in 0..1.
    fn random(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 17;
        self.rng ^= self.rng << 5;
        (self.rng >> 8) as f32 / (1u32 << 24) as f32
    }

    /// Spawns a drop at `position` with an upward toss and a random sideways impulse.
    pub fn spawn(&mut self, position: Vec3, block_id: BlockId, count: u16) {
        if self.items.len() >= MAX_DROPS || count == 0 {
            return;
        }
        let velocity = Vec3::new((self.random() - 0.5) * 3.0, 3.0, (self.random() - 0.5) * 3.0);
        self.items.push(ItemDrop { position, velocity, block_id, count, lifetime: 0.0 });
    }

    pub fn update(
        &mut self,
        dt: f32,
        world: &World,
        player_center: Vec3,
        inventory: &mut PlayerInventory,
    ) {
        for drop in &mut self.items {
            drop.lifetime += dt;

            let to_player = player_center - drop.position;
            let dist = to_player.length();
            let magnet = drop.lifetime > PICKUP_DELAY
                && dist < MAGNET_RANGE
                && inventory.can_add(drop.block_id);

            if magnet {
                if dist < PICKUP_RANGE {
                    drop.count = inventory.add(drop.block_id, drop.count);
                } else {
                    // Pulled in faster the closer it gets; ignores gravity and collisions.
                    let speed = 6.0 + (MAGNET_RANGE - dist) * 6.0;
                    drop.position += to_player / dist * speed * dt;
                    drop.velocity = Vec3::ZERO;
                }
            } else {
                let steps = (dt / MAX_STEP).ceil().max(1.0) as u32;
                let h = dt / steps as f32;
                for _ in 0..steps {
                    step_physics(drop, h, world);
                }
            }
        }

        self.merge_accum += dt;
        if self.merge_accum >= MERGE_INTERVAL_SECS {
            self.merge_accum -= MERGE_INTERVAL_SECS;
            self.merge_nearby_stacks();
        }

        self.items
            .retain(|d| d.count > 0 && d.lifetime < MAX_LIFETIME && d.position.y > VOID_Y);
    }

    /// Merges same-type `ItemDrop`s within `1.2` meters of each other (up to stack size 64).
    pub fn merge_nearby_stacks(&mut self) {
        let max_dist_sq = MERGE_DISTANCE * MERGE_DISTANCE;
        let len = self.items.len();
        for i in 0..len {
            if self.items[i].count == 0 || self.items[i].count >= MAX_STACK {
                continue;
            }
            for j in (i + 1)..len {
                if self.items[j].count == 0
                    || self.items[i].block_id != self.items[j].block_id
                {
                    continue;
                }
                if self.items[i]
                    .position
                    .distance_squared(self.items[j].position)
                    < max_dist_sq
                {
                    let space = MAX_STACK - self.items[i].count;
                    let moved = self.items[j].count.min(space);
                    self.items[i].count += moved;
                    self.items[j].count -= moved;
                    if self.items[i].count >= MAX_STACK {
                        break;
                    }
                }
            }
        }
        self.items.retain(|d| d.count > 0);
    }
}

/// Gravity + fluid current push + per-axis collision of the cube's leading face against solid voxels.
fn step_physics(drop: &mut ItemDrop, h: f32, world: &World) {
    let current_cell = drop.position.floor().as_ivec3();
    if world.is_fluid_at(current_cell) {
        let flow = world.fluid_flow_at(current_cell);
        drop.velocity += flow * 5.0 * h;
        // Gentle sink/float in fluid + drag
        drop.velocity.y = (drop.velocity.y + GRAVITY * 0.18 * h).max(-2.5);
        let drag = 0.25_f32.powf(h);
        drop.velocity.x *= drag;
        drop.velocity.z *= drag;
    } else {
        drop.velocity.y = (drop.velocity.y + GRAVITY * h).max(TERMINAL_VELOCITY);
    }
    let mut on_ground = false;

    for axis in 0..3 {
        let delta = drop.velocity[axis] * h;
        if delta == 0.0 {
            continue;
        }
        let mut next = drop.position;
        next[axis] += delta;
        let mut probe = next;
        probe[axis] += HALF_SIZE * delta.signum();
        let cell = probe.floor().as_ivec3();

        if world.is_solid(cell) {
            if axis == 1 && delta < 0.0 {
                on_ground = true;
                // Rest exactly on top of the block.
                drop.position.y = cell.y as f32 + 1.0 + HALF_SIZE;
            }
            drop.velocity[axis] = 0.0;
        } else {
            drop.position = next;
        }
    }

    if on_ground {
        // Ground friction (per-second factor, frame-rate independent).
        let friction = 0.02_f32.powf(h);
        drop.velocity.x *= friction;
        drop.velocity.z *= friction;
    }
}

/// Centre of a block cell.
pub fn block_center(block: IVec3) -> Vec3 {
    block.as_vec3() + Vec3::splat(0.5)
}

/// Builds a combined vertex + index mesh for all active drops (0.25x0.25x0.25 cubes
/// spinning around Y and gently bobbing up and down).
pub fn build_drops_mesh(items: &[ItemDrop]) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::with_capacity(items.len() * 24);
    let mut indices = Vec::with_capacity(items.len() * 36);

    const H: f32 = HALF_SIZE; // 0.125 -> 0.25 cube
    // 6 faces in local space: (4 CCW corners, outward normal).
    const FACES: [([Vec3; 4], Vec3); 6] = [
        // +X
        (
            [
                Vec3::new(H, -H, H),
                Vec3::new(H, -H, -H),
                Vec3::new(H, H, -H),
                Vec3::new(H, H, H),
            ],
            Vec3::X,
        ),
        // -X
        (
            [
                Vec3::new(-H, -H, -H),
                Vec3::new(-H, -H, H),
                Vec3::new(-H, H, H),
                Vec3::new(-H, H, -H),
            ],
            Vec3::NEG_X,
        ),
        // +Y
        (
            [
                Vec3::new(-H, H, H),
                Vec3::new(H, H, H),
                Vec3::new(H, H, -H),
                Vec3::new(-H, H, -H),
            ],
            Vec3::Y,
        ),
        // -Y
        (
            [
                Vec3::new(-H, -H, -H),
                Vec3::new(H, -H, -H),
                Vec3::new(H, -H, H),
                Vec3::new(-H, -H, H),
            ],
            Vec3::NEG_Y,
        ),
        // +Z
        (
            [
                Vec3::new(-H, -H, H),
                Vec3::new(H, -H, H),
                Vec3::new(H, H, H),
                Vec3::new(-H, H, H),
            ],
            Vec3::Z,
        ),
        // -Z
        (
            [
                Vec3::new(H, -H, -H),
                Vec3::new(-H, -H, -H),
                Vec3::new(-H, H, -H),
                Vec3::new(H, H, -H),
            ],
            Vec3::NEG_Z,
        ),
    ];

    for drop in items {
        let angle = drop.lifetime * 2.2;
        let bob = (drop.lifetime * 3.2).sin() * 0.05 + 0.05;
        let center = drop.position + Vec3::Y * bob;
        let rot = Mat3::from_rotation_y(angle);
        let color = block_color(drop.block_id);

        for (corners, normal) in FACES {
            let base = vertices.len() as u32;
            let n = (rot * normal).to_array();
            for c in corners {
                let p = (center + rot * c).to_array();
                vertices.push(Vertex {
                    position: p,
                    normal: n,
                    color,
                    light: [12.0, 12.0, 12.0, 15.0],
                    ao_and_flags: [3.0, 0.0],
                });
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }

    (vertices, indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::chunk::STONE;

    #[test]
    fn nearby_drops_of_same_type_merge_within_1_2_meters() {
        let mut drops = Drops::new();
        drops.spawn(Vec3::new(0.0, 5.0, 0.0), STONE, 20);
        drops.spawn(Vec3::new(0.8, 5.0, 0.2), STONE, 30);
        drops.spawn(Vec3::new(3.0, 5.0, 0.0), STONE, 10);

        drops.merge_nearby_stacks();
        assert_eq!(drops.items().len(), 2);
        assert_eq!(drops.items()[0].count, 50);
        assert_eq!(drops.items()[1].count, 10);
    }
}
