// Camera module — FPS camera (view/projection) + input collector (CameraController).
//
// Coordinate system: right-handed, Y-up.
// Yaw = rotation around Y, pitch = rotation around X (clamped to ±89°).
// The camera does not move by itself: the Player (src/player.rs) owns the position and
// feeds it back into `Camera::position` every frame.

use glam::{Mat4, Vec3};
use winit::event::{ElementState, KeyEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

// ─── Camera ──────────────────────────────────────────────────────────────────

pub struct Camera {
    pub position: Vec3,
    pub yaw: f32,   // radians
    pub pitch: f32, // radians

    pub fov_y_rad: f32,
    pub aspect: f32,
    pub z_near: f32,
    pub z_far: f32,
}

impl Camera {
    pub fn new(aspect: f32) -> Self {
        Self {
            position: Vec3::new(0.0, 40.0, 0.0),
            yaw: -std::f32::consts::FRAC_PI_2, // face -Z
            pitch: -0.35,
            fov_y_rad: 70_f32.to_radians(),
            aspect,
            z_near: 0.1,
            z_far: 1000.0,
        }
    }

    /// Unit vector the camera is looking at.
    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize()
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_to_rh(self.position, self.forward(), Vec3::Y)
    }

    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov_y_rad, self.aspect, self.z_near, self.z_far)
    }

    pub fn view_proj(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }

    pub fn set_aspect(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height.max(1) as f32;
    }
}

// ─── Camera Uniform (GPU layout) ─────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone)]
pub struct CameraUniform {
    /// Column-major 4×4 view-projection matrix.
    pub view_proj: [[f32; 4]; 4],
    /// Inverse of the rotational view-projection matrix (for sky ray direction reconstruction).
    pub inv_view_proj: [[f32; 4]; 4],
    /// `xyz` = normalized direction towards the sun, `w` = ambient light strength.
    pub sun_dir: [f32; 4],
    /// `rgb` = directional sun light color, `w` = `1.0` if camera is underwater else `0.0`.
    pub sun_color: [f32; 4],
    /// `rgb` = linear sky color, `w` = sunrise/sunset horizon factor (`0.0..=1.0`).
    pub sky_color: [f32; 4],
    /// `xyz` = camera world-space position (used to orbit 3D Sun/Moon billboards at `camera.pos ± sun_dir * 400.0`).
    pub cam_pos: [f32; 4],
}

impl CameraUniform {
    pub fn from_camera_and_sky(
        camera: &Camera,
        sky: &crate::render::sky::SkyState,
        is_underwater: bool,
    ) -> Self {
        let dir = sky.sun_direction();
        let col = sky.sun_color();
        let sky_rgb = sky.sky_linear_rgb();
        let horizon = sky.sun_horizon_factor();
        // Rotational view matrix (origin at 0) so sky rays are translation-invariant.
        let rot_view = Mat4::look_to_rh(Vec3::ZERO, camera.forward(), Vec3::Y);
        let rot_vp = camera.projection_matrix() * rot_view;
        Self {
            view_proj: camera.view_proj().to_cols_array_2d(),
            inv_view_proj: rot_vp.inverse().to_cols_array_2d(),
            sun_dir: [dir.x, dir.y, dir.z, sky.ambient_strength()],
            sun_color: [
                col[0],
                col[1],
                col[2],
                if is_underwater { 1.0 } else { 0.0 },
            ],
            sky_color: [sky_rgb[0], sky_rgb[1], sky_rgb[2], horizon],
            cam_pos: [camera.position.x, camera.position.y, camera.position.z, 1.0],
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: repr(C), only f32 fields, no padding.
        unsafe {
            std::slice::from_raw_parts(
                (self as *const Self) as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }
}

// ─── Camera Controller (input collector + mouse look) ────────────────────────

pub struct CameraController {
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    sprint: bool,

    mouse_delta_x: f32,
    mouse_delta_y: f32,
    mouse_sens: f32, // radians per pixel
    fly_speed: f32,  // units per second

    /// Set on a (non-repeat) Space key press; consumed by the player (double-tap detection).
    space_pressed_edge: bool,
}

impl CameraController {
    pub fn new() -> Self {
        Self {
            forward: false,
            backward: false,
            left: false,
            right: false,
            up: false,
            down: false,
            sprint: false,
            mouse_delta_x: 0.0,
            mouse_delta_y: 0.0,
            mouse_sens: 0.0015,
            fly_speed: 10.0,
            space_pressed_edge: false,
        }
    }

    pub fn process_keyboard(&mut self, event: &KeyEvent) {
        let pressed = event.state == ElementState::Pressed;
        if let PhysicalKey::Code(code) = event.physical_key {
            match code {
                KeyCode::KeyW | KeyCode::ArrowUp => self.forward = pressed,
                KeyCode::KeyS | KeyCode::ArrowDown => self.backward = pressed,
                KeyCode::KeyA | KeyCode::ArrowLeft => self.left = pressed,
                KeyCode::KeyD | KeyCode::ArrowRight => self.right = pressed,
                KeyCode::Space => {
                    if pressed && !event.repeat {
                        self.space_pressed_edge = true;
                    }
                    self.up = pressed;
                }
                KeyCode::ShiftLeft => self.down = pressed,
                KeyCode::ControlLeft => self.sprint = pressed,
                _ => {}
            }
        }
    }

    pub fn process_mouse_motion(&mut self, dx: f64, dy: f64) {
        self.mouse_delta_x += dx as f32;
        self.mouse_delta_y += dy as f32;
    }

    /// Applies the accumulated mouse delta to yaw/pitch.
    pub fn update_look(&mut self, camera: &mut Camera) {
        camera.yaw += self.mouse_delta_x * self.mouse_sens;
        camera.pitch -= self.mouse_delta_y * self.mouse_sens;
        camera.pitch = camera.pitch.clamp(-89_f32.to_radians(), 89_f32.to_radians());
        self.mouse_delta_x = 0.0;
        self.mouse_delta_y = 0.0;
    }

    /// (forward axis, right axis), each in -1..=1.
    pub fn move_axes(&self) -> (f32, f32) {
        let axis = |pos: bool, neg: bool| pos as i32 as f32 - neg as i32 as f32;
        (axis(self.forward, self.backward), axis(self.right, self.left))
    }

    pub fn up_held(&self) -> bool {
        self.up
    }

    pub fn down_held(&self) -> bool {
        self.down
    }

    pub fn sprint_held(&self) -> bool {
        self.sprint
    }

    pub fn fly_speed(&self) -> f32 {
        self.fly_speed
    }

    /// Returns true once per Space key press.
    pub fn take_space_press(&mut self) -> bool {
        std::mem::take(&mut self.space_pressed_edge)
    }

    /// Resets all movement key flags and accumulated mouse deltas (called when opening UI
    /// modals or when the window loses focus so movement never sticks).
    pub fn clear_input(&mut self) {
        self.forward = false;
        self.backward = false;
        self.left = false;
        self.right = false;
        self.up = false;
        self.down = false;
        self.sprint = false;
        self.mouse_delta_x = 0.0;
        self.mouse_delta_y = 0.0;
        self.space_pressed_edge = false;
    }
}
