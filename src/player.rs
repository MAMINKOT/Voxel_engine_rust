// Player: AABB physics (walking, swimming in fluids, and creative flight with AABB collisions).
//
// Position is the point between the feet. The hitbox is 0.6 x 1.8 x 0.6, eyes at 1.65.
// Both Walking and Flying use axis-by-axis collision resolution: move along X, resolve;
// then Y; then Z. Each step is small (<= 1/120 s), so the player never tunnels through a wall.

use std::time::{Duration, Instant};

use glam::{IVec3, Vec3};

use crate::camera::{Camera, CameraController};
use crate::voxel::world::World;

const HALF_WIDTH: f32 = 0.3;
const HEIGHT: f32 = 1.8;
const EYE_HEIGHT: f32 = 1.65;

const GRAVITY: f32 = -28.0;
const JUMP_SPEED: f32 = 8.5;
const TERMINAL_VELOCITY: f32 = -60.0;
const WALK_SPEED: f32 = 4.5;
const RUN_MULTIPLIER: f32 = 1.5;
const GROUND_ACCEL: f32 = 50.0;
const AIR_ACCEL: f32 = 12.0;
const FLY_TURBO: f32 = 3.0;

// Fluid buoyancy & drag constants
const FLUID_SPEED_FACTOR: f32 = 0.55;
const FLUID_GRAVITY_FACTOR: f32 = 0.22;
const FLUID_SINK_LIMIT: f32 = -4.5;
const FLUID_SWIM_UP_SPEED: f32 = 4.5;

const DOUBLE_TAP_WINDOW: Duration = Duration::from_millis(250);
const MAX_SUBSTEP: f32 = 1.0 / 120.0;
/// Gap kept between the hitbox and the block it was pushed out of.
const EPS: f32 = 1e-3;
const VOID_Y: f32 = -30.0;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum MoveMode {
    Walking,
    Flying,
}

pub struct Player {
    position: Vec3, // feet
    velocity: Vec3,
    external_impulse: Vec3,
    on_ground: bool,
    mode: MoveMode,
    last_space_tap: Option<Instant>,
    spawn: Vec3,
}

impl Player {
    pub fn new(spawn: Vec3) -> Self {
        Self {
            position: spawn,
            velocity: Vec3::ZERO,
            external_impulse: Vec3::ZERO,
            on_ground: false,
            mode: MoveMode::Walking,
            last_space_tap: None,
            spawn,
        }
    }

    pub fn position(&self) -> Vec3 {
        self.position
    }

    pub fn eye_position(&self) -> Vec3 {
        self.position + Vec3::Y * EYE_HEIGHT
    }

    pub fn center(&self) -> Vec3 {
        self.position + Vec3::Y * (HEIGHT * 0.5)
    }

    pub fn is_flying(&self) -> bool {
        self.mode == MoveMode::Flying
    }

    /// Resets horizontal (and flying vertical) velocity when opening a UI screen or losing window focus.
    pub fn reset_inputs(&mut self) {
        self.velocity.x = 0.0;
        self.velocity.z = 0.0;
        if self.mode == MoveMode::Flying {
            self.velocity.y = 0.0;
        }
        self.last_space_tap = None;
    }

    /// Applies an instantaneous velocity impulse (e.g. from a nearby TNT explosion shockwave).
    pub fn apply_impulse(&mut self, impulse: Vec3) {
        self.velocity += impulse;
        self.external_impulse += impulse;
        if impulse.y > 0.5 {
            self.on_ground = false;
        }
    }

    /// True if the unit block at `block` overlaps the player's hitbox.
    pub fn intersects_block(&self, block: IVec3) -> bool {
        let (min, max) = self.aabb();
        let b_min = block.as_vec3();
        let b_max = b_min + Vec3::ONE;
        min.x < b_max.x
            && max.x > b_min.x
            && min.y < b_max.y
            && max.y > b_min.y
            && min.z < b_max.z
            && max.z > b_min.z
    }

    pub fn update(
        &mut self,
        dt: f32,
        input: &mut CameraController,
        camera: &Camera,
        world: &World,
    ) {
        if input.take_space_press() {
            self.register_space_tap(world);
        }

        match self.mode {
            MoveMode::Flying => self.update_flying(dt, input, camera, world),
            MoveMode::Walking => self.update_walking(dt, input, camera, world),
        }

        if self.position.y < VOID_Y {
            self.position = self.spawn;
            self.velocity = Vec3::ZERO;
            self.external_impulse = Vec3::ZERO;
        }
    }

    // ── Mode switching ────────────────────────────────────────────────────

    fn register_space_tap(&mut self, world: &World) {
        let now = Instant::now();
        match self.last_space_tap {
            Some(t) if now.duration_since(t) < DOUBLE_TAP_WINDOW => {
                self.last_space_tap = None;
                self.toggle_mode(world);
            }
            _ => self.last_space_tap = Some(now),
        }
    }

    fn toggle_mode(&mut self, world: &World) {
        self.velocity = Vec3::ZERO;
        self.external_impulse = Vec3::ZERO;
        match self.mode {
            MoveMode::Flying => {
                self.mode = MoveMode::Walking;
                self.on_ground = false;
                // If inside terrain, push up to free space.
                for _ in 0..64 {
                    if !self.collides(world) {
                        break;
                    }
                    self.position.y += 1.0;
                }
                log::info!("Mode: Walking");
            }
            MoveMode::Walking => {
                self.mode = MoveMode::Flying;
                log::info!("Mode: Flying (AABB collisions active)");
            }
        }
    }

    // ── Flying: zero gravity, smooth gliding, full AABB collision resolution ──

    fn update_flying(
        &mut self,
        dt: f32,
        input: &CameraController,
        camera: &Camera,
        world: &World,
    ) {
        let (f, r) = input.move_axes();
        // Horizontal plane movement along camera yaw + vertical Space/LShift
        let (sin, cos) = camera.yaw.sin_cos();
        let forward = Vec3::new(cos, 0.0, sin);
        let right = Vec3::new(-sin, 0.0, cos);
        let mut dir = (forward * f + right * r).normalize_or_zero();
        if input.up_held() {
            dir.y += 1.0;
        }
        if input.down_held() {
            dir.y -= 1.0;
        }
        if dir.length_squared() > 1.0 {
            dir = dir.normalize();
        }

        let turbo = if input.sprint_held() { FLY_TURBO } else { 1.0 };
        let fluid_mult = if self.in_fluid(world) { 0.7 } else { 1.0 };
        self.velocity = dir * input.fly_speed() * turbo * fluid_mult + self.external_impulse;
        self.external_impulse *= (1.0 - 4.0 * dt).clamp(0.0, 1.0);

        let steps = (dt / MAX_SUBSTEP).ceil().max(1.0) as u32;
        let h = dt / steps as f32;

        for _ in 0..steps {
            // Axis-by-axis AABB collision resolution (X -> Y -> Z)
            self.move_axis(0, self.velocity.x * h, world);
            self.move_axis(1, self.velocity.y * h, world);
            self.move_axis(2, self.velocity.z * h, world);
        }
    }

    // ── Walking: gravity + fluid buoyancy + AABB collisions ───────────────

    fn update_walking(
        &mut self,
        dt: f32,
        input: &CameraController,
        camera: &Camera,
        world: &World,
    ) {
        let (f, r) = input.move_axes();
        let (sin, cos) = camera.yaw.sin_cos();
        let forward = Vec3::new(cos, 0.0, sin);
        let right = Vec3::new(-sin, 0.0, cos);
        let wish = (forward * f + right * r).normalize_or_zero();
        let run = if input.sprint_held() { RUN_MULTIPLIER } else { 1.0 };

        let steps = (dt / MAX_SUBSTEP).ceil().max(1.0) as u32;
        let h = dt / steps as f32;

        for _ in 0..steps {
            let swimming = self.in_fluid(world);
            let speed_mult = if swimming { FLUID_SPEED_FACTOR } else { 1.0 };
            let target = wish * WALK_SPEED * run * speed_mult;

            // Horizontal acceleration / friction: approach the target velocity.
            let accel = if self.on_ground && !swimming {
                GROUND_ACCEL
            } else {
                AIR_ACCEL
            };
            let delta = target - Vec3::new(self.velocity.x, 0.0, self.velocity.z);
            let max_change = accel * h;
            let change = if delta.length() > max_change {
                delta.normalize() * max_change
            } else {
                delta
            };
            self.velocity.x += change.x;
            self.velocity.z += change.z;

            if swimming {
                let flow = self.fluid_flow(world);
                self.velocity += flow * 4.5 * h;
                if input.up_held() {
                    self.velocity.y = FLUID_SWIM_UP_SPEED;
                } else {
                    self.velocity.y = (self.velocity.y + GRAVITY * FLUID_GRAVITY_FACTOR * h)
                        .max(FLUID_SINK_LIMIT);
                }
                self.on_ground = false;
            } else {
                if input.up_held() && self.on_ground {
                    self.velocity.y = JUMP_SPEED;
                    self.on_ground = false;
                }
                self.velocity.y = (self.velocity.y + GRAVITY * h).max(TERMINAL_VELOCITY);
            }

            // Axis-by-axis resolution: X, then Y, then Z.
            self.move_axis(0, self.velocity.x * h, world);
            self.on_ground = false;
            self.move_axis(1, self.velocity.y * h, world);
            self.move_axis(2, self.velocity.z * h, world);
        }
        self.external_impulse = Vec3::ZERO;
    }

    fn aabb(&self) -> (Vec3, Vec3) {
        (
            self.position - Vec3::new(HALF_WIDTH, 0.0, HALF_WIDTH),
            self.position + Vec3::new(HALF_WIDTH, HEIGHT, HALF_WIDTH),
        )
    }

    fn collides(&self, world: &World) -> bool {
        let (min, max) = self.aabb();
        for x in min.x.floor() as i32..=max.x.floor() as i32 {
            for y in min.y.floor() as i32..=max.y.floor() as i32 {
                for z in min.z.floor() as i32..=max.z.floor() as i32 {
                    if world.is_solid(IVec3::new(x, y, z)) {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn in_fluid(&self, world: &World) -> bool {
        let (min, max) = self.aabb();
        for x in min.x.floor() as i32..=max.x.floor() as i32 {
            for y in min.y.floor() as i32..=max.y.floor() as i32 {
                for z in min.z.floor() as i32..=max.z.floor() as i32 {
                    if world.is_fluid_at(IVec3::new(x, y, z)) {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn fluid_flow(&self, world: &World) -> Vec3 {
        let (min, max) = self.aabb();
        let mut sum = Vec3::ZERO;
        let mut count = 0u32;
        for x in min.x.floor() as i32..=max.x.floor() as i32 {
            for y in min.y.floor() as i32..=max.y.floor() as i32 {
                for z in min.z.floor() as i32..=max.z.floor() as i32 {
                    let p = IVec3::new(x, y, z);
                    if world.is_fluid_at(p) {
                        sum += world.fluid_flow_at(p);
                        count += 1;
                    }
                }
            }
        }
        if count > 0 {
            (sum / count as f32).normalize_or_zero()
        } else {
            Vec3::ZERO
        }
    }

    /// Moves along one axis; on overlap snaps the hitbox flush against the blocking voxel
    /// face, zeroes the velocity on that axis and sets `on_ground` when landing.
    fn move_axis(&mut self, axis: usize, delta: f32, world: &World) {
        if delta == 0.0 {
            return;
        }
        self.position[axis] += delta;
        if !self.collides(world) {
            return;
        }

        // Extent of the hitbox below/above the position along this axis.
        let (neg, pos) = if axis == 1 {
            (0.0, HEIGHT)
        } else {
            (HALF_WIDTH, HALF_WIDTH)
        };
        if delta > 0.0 {
            self.position[axis] = (self.position[axis] + pos).floor() - pos - EPS;
        } else {
            self.position[axis] = (self.position[axis] - neg).floor() + 1.0 + neg + EPS;
            if axis == 1 {
                self.on_ground = true;
            }
        }
        self.velocity[axis] = 0.0;
        self.external_impulse[axis] = 0.0;
    }
}
