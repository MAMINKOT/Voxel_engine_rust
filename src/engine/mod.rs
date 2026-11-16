// engine/mod.rs — Central EngineState.
//
// winit 0.29 API:
//   • EventLoop::run(|event, elwt| { … })  — 2 args, no control_flow param
//   • elwt.exit() instead of ControlFlow::Exit
//   • Event::AboutToWait instead of MainEventsCleared
//   • ControlFlow set via elwt.set_control_flow()
//
// WGPU 0.19: Surface<'static> via Arc<Window>.

use std::sync::Arc;
use std::time::{Duration, Instant};

use glam::Vec3;
use sysinfo::{Pid, System};
use winit::dpi::PhysicalSize;
use winit::event::{
    DeviceEvent, ElementState, Event, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent,
};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowBuilder};

use crate::camera::{Camera, CameraController, CameraUniform};
use crate::inventory::PlayerInventory;
use crate::player::Player;
use crate::render::clouds::CloudRenderer;
use crate::render::overlay::OverlayRenderer;
use crate::render::pipeline::{
    create_camera_bind_group, create_camera_bind_group_layout, create_camera_buffer,
    create_render_pipeline, create_scene_env_bind_group, create_scene_env_bind_group_layout,
    create_sky_pipeline, create_translucent_pipeline, DepthTexture,
};
use crate::render::postprocess::PostProcessor;
use crate::render::shadows::ShadowRenderer;
use crate::render::sky::SkyState;
use crate::render::texture::BlockTextureArray;
use crate::render::ui::{UiScreen, UiSystem};
use crate::save::{
    default_world_meta, delete_world, hash_seed_str, load_world_meta, load_world_overrides,
    pack_inventory, sanitize_folder_name, save_world, unpack_inventory, WorldMeta,
};
use crate::sim::energy_net::TICK_DT;
use crate::state::AppState;
use crate::ui::debug::DebugInfo;
use crate::ui::main_menu::{MainMenuAction, MainMenuState};
use crate::ui::menu::{GraphicsSettings, PauseMenuAction, ShadowQuality};
use crate::voxel::chunk::{is_fluid, is_placeable, AIR, FIRE, LIGHTER, TNT, WATER};
use crate::voxel::raycast::{raycast_blocks, RaycastHit};
use crate::voxel::world::World;

// ─── FPS Counter ─────────────────────────────────────────────────────────────

struct FpsCounter {
    frame_count: u32,
    elapsed: Duration,
    last_fps: f64,
}

impl FpsCounter {
    fn new() -> Self {
        Self { frame_count: 0, elapsed: Duration::ZERO, last_fps: 0.0 }
    }

    /// Call once per frame; returns `Some(fps)` once per second.
    fn tick(&mut self, dt: Duration) -> Option<f64> {
        self.frame_count += 1;
        self.elapsed += dt;
        if self.elapsed >= Duration::from_secs(1) {
            let fps = self.frame_count as f64 / self.elapsed.as_secs_f64();
            self.frame_count = 0;
            self.elapsed = Duration::ZERO;
            self.last_fps = fps;
            Some(fps)
        } else {
            None
        }
    }
}

// ─── GPU State ───────────────────────────────────────────────────────────────

struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    sky_pipeline: wgpu::RenderPipeline,
    render_pipeline: wgpu::RenderPipeline,
    translucent_pipeline: wgpu::RenderPipeline,
    depth_texture: DepthTexture,
    scene_depth_copy: DepthTexture,
    shadows: ShadowRenderer,
    clouds: CloudRenderer,
    block_textures: BlockTextureArray,
    scene_env_bgl: wgpu::BindGroupLayout,
    scene_env_bg: wgpu::BindGroup,
    postprocess: PostProcessor,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    world: World,
    overlay: OverlayRenderer,
    ui: UiSystem,
}

// ─── Engine ───────────────────────────────────────────────────────────────────

/// Maximum block interaction distance.
const REACH: f32 = 6.0;

struct Engine {
    window: Arc<Window>,
    gpu: GpuState,
    app_state: AppState,
    main_menu: MainMenuState,
    active_world_folder: Option<String>,
    active_world_name: String,
    active_world_seed: u64,
    active_world_creative: bool,
    autosave_accum: f32,
    camera: Camera,
    camera_ctrl: CameraController,
    sky: SkyState,
    player: Player,
    inventory: PlayerInventory,
    settings: GraphicsSettings,
    quit_requested: bool,
    target: Option<RaycastHit>,
    tick_accum: f32,
    elapsed_secs: f32,
    last_frame: Instant,
    last_dt_secs: f32,
    fps_counter: FpsCounter,
    sys: System,
    pid: Option<Pid>,
    used_process_mb: u64,
    total_system_mb: u64,
    mem_percent: f32,
    cursor_grabbed: bool,
}

impl Engine {
    // ── Cursor capture & UI modal helpers ─────────────────────────────────

    fn grab_cursor(&mut self, grab: bool) {
        if grab {
            if self.window.set_cursor_grab(CursorGrabMode::Locked).is_err() {
                let _ = self.window.set_cursor_grab(CursorGrabMode::Confined);
            }
            self.window.set_cursor_visible(false);
        } else {
            let _ = self.window.set_cursor_grab(CursorGrabMode::None);
            self.window.set_cursor_visible(true);
        }
        self.cursor_grabbed = grab;
    }

    fn open_ui(&mut self, screen: UiScreen) {
        self.gpu.ui.screen = screen;
        self.camera_ctrl.clear_input();
        self.player.reset_inputs();
        self.grab_cursor(false);
    }

    fn close_ui(&mut self, recapture_mouse: bool) {
        self.gpu.ui.screen = UiScreen::Closed;
        if self.app_state == AppState::Paused {
            self.app_state = AppState::InGame;
        }
        for leftover in self.inventory.return_cursor_and_crafting() {
            self.gpu
                .world
                .drops
                .spawn(self.player.center(), leftover.id, leftover.count);
        }
        if recapture_mouse {
            self.grab_cursor(true);
        }
    }

    fn save_active_world(&self) {
        let Some(folder) = &self.active_world_folder else {
            return;
        };
        let meta = WorldMeta {
            world_name: self.active_world_name.clone(),
            seed: self.active_world_seed,
            time_of_day: self.sky.day_time,
            game_mode_creative: self.active_world_creative,
            player_pos: self.player.position().to_array(),
            player_yaw: self.camera.yaw,
            player_pitch: self.camera.pitch,
            player_inventory: pack_inventory(&self.inventory),
        };
        if let Err(err) = save_world(folder, &meta, &self.gpu.world) {
            log::error!("[Save] Failed to save world '{}': {}", folder, err);
        }
    }

    fn start_new_world(&mut self, name: String, seed_text: String, creative: bool) {
        let clean_name = if name.trim().is_empty() {
            "New Industrial World".to_string()
        } else {
            name.trim().to_string()
        };
        let folder = sanitize_folder_name(&clean_name);
        let seed = hash_seed_str(&seed_text);

        self.app_state = AppState::Loading {
            progress: 0.4,
            message: format!("Генерація світу '{}'...", clean_name),
        };
        self.gpu.world = World::generate_with_seed(&self.gpu.device, seed);
        self.inventory = PlayerInventory::new();
        self.sky.day_time = 0.25;

        let spawn_y = self.gpu.world.surface_height_at(0, 0) as f32 + 1.0;
        let spawn = Vec3::new(0.5, spawn_y.max(18.0), 0.5);
        self.player.set_pose_and_mode(spawn, creative);
        self.camera.position = self.player.eye_position();
        self.camera.yaw = 0.785;
        self.camera.pitch = -0.25;

        self.active_world_folder = Some(folder.clone());
        self.active_world_name = clean_name.clone();
        self.active_world_seed = seed;
        self.active_world_creative = creative;
        self.autosave_accum = 0.0;

        let meta = default_world_meta(clean_name, seed, creative, spawn, &self.inventory);
        let _ = save_world(&folder, &meta, &self.gpu.world);
        self.main_menu.refresh_worlds();

        self.gpu.ui.screen = UiScreen::Closed;
        self.app_state = AppState::InGame;
        self.grab_cursor(true);
        self.refresh_title();
    }

    fn load_saved_world(&mut self, folder: String) {
        let Ok(meta) = load_world_meta(&folder) else {
            log::error!("[Load] Failed to read level.dat for '{}'", folder);
            return;
        };

        self.app_state = AppState::Loading {
            progress: 0.5,
            message: format!("Завантаження світу '{}'...", meta.world_name),
        };

        self.gpu.world = World::generate_with_seed(&self.gpu.device, meta.seed);
        if let Err(err) = load_world_overrides(&folder, &self.gpu.device, &mut self.gpu.world) {
            log::warn!("[Load] Partial load warning for '{}': {}", folder, err);
        }

        self.inventory = PlayerInventory::new();
        unpack_inventory(&meta.player_inventory, &mut self.inventory);
        self.sky.day_time = meta.time_of_day;
        self.player
            .set_pose_and_mode(Vec3::from_array(meta.player_pos), meta.game_mode_creative);
        self.camera.position = self.player.eye_position();
        self.camera.yaw = meta.player_yaw;
        self.camera.pitch = meta.player_pitch;

        self.active_world_folder = Some(folder);
        self.active_world_name = meta.world_name;
        self.active_world_seed = meta.seed;
        self.active_world_creative = meta.game_mode_creative;
        self.autosave_accum = 0.0;

        self.gpu.ui.screen = UiScreen::Closed;
        self.app_state = AppState::InGame;
        self.grab_cursor(true);
        self.refresh_title();
    }

    // ── Resize ────────────────────────────────────────────────────────────

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.gpu.config.width = new_size.width;
        self.gpu.config.height = new_size.height;
        self.gpu.surface.configure(&self.gpu.device, &self.gpu.config);
        self.gpu.depth_texture = DepthTexture::create(
            &self.gpu.device,
            new_size.width,
            new_size.height,
            "depth_texture",
        );
        self.gpu.scene_depth_copy = DepthTexture::create(
            &self.gpu.device,
            new_size.width,
            new_size.height,
            "scene_depth_copy",
        );
        self.gpu.scene_env_bg = create_scene_env_bind_group(
            &self.gpu.device,
            &self.gpu.scene_env_bgl,
            &self.gpu.shadows.view,
            &self.gpu.shadows.sampler,
            &self.gpu.scene_depth_copy.view,
            &self.gpu.block_textures.view,
            &self.gpu.block_textures.sampler,
        );
        self.gpu.postprocess.resize(
            &self.gpu.device,
            new_size.width,
            new_size.height,
            &self.gpu.scene_depth_copy,
            &self.gpu.camera_buffer,
        );
        self.camera.set_aspect(new_size.width, new_size.height);
        self.gpu.overlay.resize(&self.gpu.queue, new_size.width, new_size.height);
    }

    // ── Window title / HUD & RAM refresh ──────────────────────────────────

    fn refresh_memory_stats(&mut self) {
        if let Some(pid) = self.pid {
            self.sys.refresh_process(pid);
            if let Some(proc_info) = self.sys.process(pid) {
                self.used_process_mb = proc_info.memory() / (1024 * 1024);
            }
        }
        self.total_system_mb = (self.sys.total_memory() / (1024 * 1024)).max(1);
        self.mem_percent =
            (self.used_process_mb as f32 / self.total_system_mb as f32) * 100.0;
    }

    fn refresh_title(&self) {
        let machine_info = self
            .target
            .and_then(|hit| self.gpu.world.energy.status(hit.block_pos));
        let title = match machine_info {
            Some(status) => format!(
                "Voxel Industrial Engine | FPS: {:.0} | {} | {}",
                self.fps_counter.last_fps,
                self.inventory.status_line(),
                status
            ),
            None => format!(
                "Voxel Industrial Engine | FPS: {:.0} | {}",
                self.fps_counter.last_fps,
                self.inventory.status_line()
            ),
        };
        self.window.set_title(&title);
    }

    // ── Input ─────────────────────────────────────────────────────────────

    fn handle_key(&mut self, event: &KeyEvent) {
        let in_menu_screen = matches!(
            self.app_state,
            AppState::MainMenu | AppState::WorldSelect | AppState::WorldCreate | AppState::Loading { .. }
        );
        if in_menu_screen {
            let pressed = event.state == ElementState::Pressed;
            if let PhysicalKey::Code(code) = event.physical_key {
                let mapped = match code {
                    KeyCode::Backspace => Some(egui::Key::Backspace),
                    KeyCode::Delete => Some(egui::Key::Delete),
                    KeyCode::Enter => Some(egui::Key::Enter),
                    KeyCode::Escape => Some(egui::Key::Escape),
                    KeyCode::ArrowLeft => Some(egui::Key::ArrowLeft),
                    KeyCode::ArrowRight => Some(egui::Key::ArrowRight),
                    KeyCode::ArrowUp => Some(egui::Key::ArrowUp),
                    KeyCode::ArrowDown => Some(egui::Key::ArrowDown),
                    KeyCode::Home => Some(egui::Key::Home),
                    KeyCode::End => Some(egui::Key::End),
                    KeyCode::Tab => Some(egui::Key::Tab),
                    _ => None,
                };
                if let Some(k) = mapped {
                    self.gpu.ui.on_egui_key(k, pressed);
                }
            }
            if pressed {
                if let Some(txt) = &event.text {
                    let filtered: String = txt.chars().filter(|c| !c.is_control()).collect();
                    if !filtered.is_empty() {
                        self.gpu.ui.on_text_input(&filtered);
                    }
                }
            }
            return;
        }

        if event.state == ElementState::Pressed && !event.repeat {
            if let PhysicalKey::Code(code) = event.physical_key {
                // F3: toggle diagnostic debug overlay.
                if code == KeyCode::F3 {
                    self.gpu.ui.debug_open = !self.gpu.ui.debug_open;
                    return;
                }
                // T: step day/night time forward by +150s for quick testing of sunrise/noon/sunset/night.
                if code == KeyCode::KeyT && !self.gpu.ui.is_open() {
                    self.sky.skip(150.0);
                    return;
                }
                // Escape: close Inventory/Machine UI if open; otherwise toggle Pause Menu (`AppState::Paused`).
                if code == KeyCode::Escape {
                    match self.gpu.ui.screen {
                        UiScreen::Inventory | UiScreen::Machine(_) | UiScreen::PauseMenu => {
                            self.close_ui(true);
                        }
                        UiScreen::OptionsMenu => {
                            self.gpu.ui.screen = UiScreen::PauseMenu;
                        }
                        UiScreen::Closed => {
                            self.app_state = AppState::Paused;
                            self.open_ui(UiScreen::PauseMenu);
                        }
                    }
                    return;
                }
                // E / I / Tab: toggle inventory / close machine UI (ignored while in Pause/Options menu).
                if matches!(code, KeyCode::KeyE | KeyCode::KeyI | KeyCode::Tab) {
                    if self.gpu.ui.is_paused() {
                        return;
                    }
                    if self.gpu.ui.is_open() {
                        self.close_ui(true);
                    } else {
                        self.open_ui(UiScreen::Inventory);
                    }
                    return;
                }
                // 1-9: select hotbar slot.
                let slot = match code {
                    KeyCode::Digit1 => Some(0),
                    KeyCode::Digit2 => Some(1),
                    KeyCode::Digit3 => Some(2),
                    KeyCode::Digit4 => Some(3),
                    KeyCode::Digit5 => Some(4),
                    KeyCode::Digit6 => Some(5),
                    KeyCode::Digit7 => Some(6),
                    KeyCode::Digit8 => Some(7),
                    KeyCode::Digit9 => Some(8),
                    _ => None,
                };
                if let Some(idx) = slot {
                    self.inventory.select(idx);
                    self.refresh_title();
                }
            }
        }
        // Only feed movement keys when no UI window is open.
        if !self.gpu.ui.is_open() {
            self.camera_ctrl.process_keyboard(event);
        }
    }

    fn handle_scroll(&mut self, delta: &MouseScrollDelta) {
        if self.app_state != AppState::InGame || self.gpu.ui.is_open() {
            return;
        }
        let scroll = match delta {
            MouseScrollDelta::LineDelta(_, y) => *y,
            MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.05,
        };
        if scroll > 0.0 {
            self.inventory.cycle(-1);
            self.refresh_title();
        } else if scroll < 0.0 {
            self.inventory.cycle(1);
            self.refresh_title();
        }
    }

    /// First click captures the cursor; afterwards LMB breaks and RMB opens machine UI or places.
    fn handle_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        if self.app_state != AppState::InGame || self.gpu.ui.is_open() {
            self.gpu.ui.on_mouse_button(button, state);
            return;
        }
        if state != ElementState::Pressed {
            return;
        }
        if !self.cursor_grabbed {
            self.grab_cursor(true);
            return;
        }
        let Some(hit) = self.target else { return };

        match button {
            MouseButton::Left => {
                self.gpu.world.set_block(&self.gpu.device, hit.block_pos, AIR);
                self.refresh_title();
            }
            MouseButton::Right => {
                // Clicking an interactive block/machine (without holding LShift) opens its UI window.
                // Holding LShift allows placing cables/blocks directly onto an interactive block face.
                if !self.camera_ctrl.down_held()
                    && self.gpu.world.energy.is_interactive(hit.block_pos)
                {
                    if let Some(status) = self.gpu.world.energy.status(hit.block_pos) {
                        log::info!("[Machine] {}", status);
                    }
                    self.open_ui(UiScreen::Machine(hit.block_pos));
                    return;
                }

                // Place the block or use tool (`LIGHTER`) from the selected hotbar slot.
                if let Some(stack) = self.inventory.selected_stack() {
                    let hit_id = self.gpu.world.block(hit.block_pos);

                    // Lighter (Flint & Steel): ignites TNT into PrimedTNT or places Fire on the face.
                    if stack.id == LIGHTER {
                        if hit_id == TNT {
                            self.gpu
                                .world
                                .ignite_tnt_at(&self.gpu.device, hit.block_pos);
                        } else {
                            let fire_pos = hit.block_pos + hit.face_normal;
                            let dest_id = self.gpu.world.block(fire_pos);
                            if dest_id == AIR {
                                self.gpu.world.set_block(&self.gpu.device, fire_pos, FIRE);
                            }
                        }
                        self.refresh_title();
                        return;
                    }

                    if is_placeable(stack.id) {
                        let pos = hit.block_pos + hit.face_normal;
                        let can_occupy = is_fluid(stack.id)
                            || stack.id == FIRE
                            || !self.player.intersects_block(pos);
                        // Compute horizontal Facing (0=North(-Z), 1=East(+X), 2=South(+Z), 3=West(-X))
                        // so the placed block's front face points back toward the player.
                        let fwd_x = self.camera.yaw.cos();
                        let fwd_z = self.camera.yaw.sin();
                        let facing = if fwd_x.abs() >= fwd_z.abs() {
                            if fwd_x >= 0.0 { 3 } else { 1 }
                        } else if fwd_z >= 0.0 {
                            0
                        } else {
                            2
                        };
                        if can_occupy
                            && self.gpu.world.set_block_with_facing(
                                &self.gpu.device,
                                pos,
                                stack.id,
                                facing,
                            )
                        {
                            self.inventory.take_one_selected();
                            self.refresh_title();
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // ── Per-frame update ──────────────────────────────────────────────────

    fn update(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame);
        self.last_frame = now;
        // Cap dt to prevent spiral-of-death after alt-tab / breakpoints.
        let dt_secs = dt.as_secs_f32().min(0.1);
        self.last_dt_secs = dt_secs;

        // Apply live FOV from Graphics Settings
        self.camera.fov_y_rad = self.settings.fov_deg.clamp(60.0, 110.0).to_radians();

        let one_sec_elapsed = self.fps_counter.tick(dt).is_some();
        if one_sec_elapsed {
            self.refresh_memory_stats();
        }

        // When not in active gameplay (`MainMenu`, `WorldSelect`, `WorldCreate`, `Loading`, `Paused`),
        // freeze world simulation & player movement.
        if self.app_state != AppState::InGame || self.gpu.ui.is_paused() {
            if one_sec_elapsed {
                self.refresh_title();
            }
            return;
        }

        // Periodic background autosave every 60 seconds
        self.autosave_accum += dt_secs;
        if self.autosave_accum >= 60.0 {
            self.autosave_accum = 0.0;
            self.save_active_world();
        }

        self.elapsed_secs = (self.elapsed_secs + dt_secs) % 100_000.0;

        // Advance Day/Night cycle.
        self.sky.update(dt_secs);

        // Fixed 20 TPS simulation tick (energy network, fluids, fire, and TNT detonations).
        self.tick_accum += dt_secs;
        let mut ticked = false;
        while self.tick_accum >= TICK_DT {
            let impulse = self
                .gpu
                .world
                .tick(&self.gpu.device, self.player.center());
            if impulse != Vec3::ZERO {
                self.player.apply_impulse(impulse);
            }
            self.tick_accum -= TICK_DT;
            ticked = true;
        }

        if !self.gpu.ui.is_open() {
            self.camera_ctrl.update_look(&mut self.camera);
        }
        self.player.update(dt_secs, &mut self.camera_ctrl, &self.camera, &self.gpu.world);
        self.camera.position = self.player.eye_position();

        // Update item drops & PrimedTNT physics and rebuild dynamic mesh.
        let world = &mut self.gpu.world;
        let mut drops = std::mem::replace(&mut world.drops, crate::voxel::drops::Drops::new());
        drops.update(dt_secs, world, self.player.center(), &mut self.inventory);
        world.drops = drops;

        let mut explosions = std::mem::replace(
            &mut world.explosions,
            crate::sim::explosions::ExplosionSystem::new(),
        );
        explosions.update_physics(dt_secs, |p| world.block(p));
        world.explosions = explosions;

        world.update_drops_mesh(&self.gpu.device);

        // Block under the crosshair (tests thin cable sub-blocks accurately).
        let prev_target = self.target.map(|h| h.block_pos);
        let world_ref = &self.gpu.world;
        self.target = raycast_blocks(self.camera.position, self.camera.forward(), REACH, |p| {
            world_ref.block(p)
        });
        let cur_target = self.target.map(|h| h.block_pos);
        self.gpu.overlay.update_wire(
            &self.gpu.queue,
            self.target.map(|h| (h.bounds_min, h.bounds_max)),
        );

        if cur_target != prev_target {
            if let Some(pos) = cur_target {
                if let Some(status) = self.gpu.world.energy.status(pos) {
                    log::info!("[Inspect] {}", status);
                }
            }
            self.refresh_title();
        } else if one_sec_elapsed || ticked {
            self.refresh_title();
        }
    }

    // ── Render ────────────────────────────────────────────────────────────

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // Check whether the camera eye is submerged in a Water block.
        let eye_cell = self.camera.position.floor().as_ivec3();
        let is_underwater = self.gpu.world.block(eye_cell) == WATER;

        // Upload camera + directional shadow matrix + dynamic sun + water wave time + graphics settings uniform.
        let uniform = CameraUniform::from_camera_and_sky(
            &self.camera,
            &self.sky,
            is_underwater,
            self.elapsed_secs,
            &self.settings,
        );
        self.gpu.queue.write_buffer(&self.gpu.camera_buffer, 0, uniform.as_bytes());
        let clear_color = self.sky.clear_color();
        let player_pos = self.player.position();
        let render_dist = self.settings.render_distance_chunks;

        let output = self.gpu.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("render_encoder") },
        );

        // 1. Directional Shadow Pass (`2048x2048` Depth32Float) when Shadows != Off
        if self.settings.shadow_quality != ShadowQuality::Off {
            let mut shadow_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sun_shadow_pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.gpu.shadows.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            shadow_pass.set_pipeline(&self.gpu.shadows.pipeline);
            shadow_pass.set_bind_group(0, &self.gpu.camera_bind_group, &[]);
            self.gpu
                .world
                .draw_opaque(&mut shadow_pass, player_pos, render_dist);
        }

        // 2. HDR Sky + Opaque Voxel Pass (`Rgba16Float` + `depth_texture`)
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hdr_opaque_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.gpu.postprocess.hdr_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.gpu.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            // 2a. Sky dome gradient + 3D Sun & Moon Billboard Quads (15 vertices)
            pass.set_pipeline(&self.gpu.sky_pipeline);
            pass.set_bind_group(0, &self.gpu.camera_bind_group, &[]);
            pass.draw(0..15, 0..1);

            // 2b. Opaque geometry pass (terrain, machines, cables, torches, drops, primed TNT)
            pass.set_pipeline(&self.gpu.render_pipeline);
            pass.set_bind_group(0, &self.gpu.camera_bind_group, &[]);
            pass.set_bind_group(1, &self.gpu.scene_env_bg, &[]);
            self.gpu
                .world
                .draw_opaque(&mut pass, player_pos, render_dist);
        }

        // 3. Copy opaque depth buffer to `scene_depth_copy` for Water Depth Fade & God Rays
        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: &self.gpu.depth_texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: &self.gpu.scene_depth_copy.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.gpu.config.width.max(1),
                height: self.gpu.config.height.max(1),
                depth_or_array_layers: 1,
            },
        );

        // 4. HDR Translucent Pass (Clouds at Y=128, Water with Waves/Fresnel/Depth Fade, Fire, Stained Glass) + Wireframe
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hdr_translucent_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.gpu.postprocess.hdr_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.gpu.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            if self.settings.clouds_enabled {
                self.gpu.clouds.draw(&mut pass, &self.gpu.camera_bind_group);
            }

            pass.set_pipeline(&self.gpu.translucent_pipeline);
            pass.set_bind_group(0, &self.gpu.camera_bind_group, &[]);
            pass.set_bind_group(1, &self.gpu.scene_env_bg, &[]);
            self.gpu
                .world
                .draw_translucent(&mut pass, player_pos, render_dist);

            self.gpu.overlay.draw_wire(&mut pass);
        }

        // 5. Post-Processing: 1/4-Res Kawase Bloom + 32-step Volumetric God Rays + ACES Tonemap + Vignette
        self.gpu.postprocess.execute(&mut encoder, &view);

        // 6. 2D Crosshair on top of the tonemapped swapchain surface (only in active gameplay)
        if self.app_state == AppState::InGame && !self.gpu.ui.is_open() {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("crosshair_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            self.gpu.overlay.draw_crosshair(&mut pass);
        }

        // Render immediate-mode UI (F3 debug overlay + HUD hotbar + open inventory/machine windows).
        let scale_factor = self.window.scale_factor() as f32;
        let width = self.gpu.config.width;
        let height = self.gpu.config.height;

        let debug_info = if self.gpu.ui.debug_open {
            let (energy_nets, machines_count, cables_count) =
                self.gpu.world.energy.network_stats();
            let targeted_block = self.target.map(|h| (h.block_pos, self.gpu.world.block(h.block_pos)));
            Some(DebugInfo {
                fps: self.fps_counter.last_fps,
                frame_time_ms: self.last_dt_secs * 1000.0,
                used_process_mb: self.used_process_mb,
                total_system_mb: self.total_system_mb,
                mem_percent: self.mem_percent,
                player_pos: self.player.position(),
                yaw_deg: self.camera.yaw.to_degrees(),
                pitch_deg: self.camera.pitch.to_degrees(),
                flying: self.player.is_flying(),
                day_time: self.sky.day_time,
                day_phase: self.sky.phase_name(),
                targeted_block,
                active_chunks: self.gpu.world.active_chunk_count(),
                loaded_fluids: self.gpu.world.fluids.loaded_fluid_count(),
                energy_nets,
                machines_count,
                cables_count,
                primed_tnt: self.gpu.world.explosions.primed_count(),
            })
        } else {
            None
        };

        let GpuState { device, queue, world, ui, .. } = &mut self.gpu;
        let mut energy =
            std::mem::replace(&mut world.energy, crate::sim::energy_net::EnergyNetwork::new());
        let (menu_action, main_action) = ui.render(
            device,
            queue,
            &mut encoder,
            &view,
            width,
            height,
            scale_factor,
            self.last_dt_secs,
            &self.app_state,
            &mut self.main_menu,
            &mut self.inventory,
            &mut energy,
            &mut self.settings,
            debug_info.as_ref(),
            |p| world.block(p),
        );
        world.energy = energy;

        match menu_action {
            PauseMenuAction::None => {}
            PauseMenuAction::Resume => {
                self.close_ui(true);
            }
            PauseMenuAction::OpenOptions => {
                self.gpu.ui.screen = UiScreen::OptionsMenu;
            }
            PauseMenuAction::BackToPause => {
                self.gpu.ui.screen = UiScreen::PauseMenu;
            }
            PauseMenuAction::SaveAndQuitToMenu => {
                self.save_active_world();
                self.gpu.ui.screen = UiScreen::Closed;
                self.grab_cursor(false);
                self.main_menu.refresh_worlds();
                self.app_state = AppState::MainMenu;
            }
            PauseMenuAction::Quit => {
                self.save_active_world();
                self.quit_requested = true;
            }
        }

        match main_action {
            MainMenuAction::None => {}
            MainMenuAction::OpenWorldSelect => {
                self.main_menu.refresh_worlds();
                self.app_state = AppState::WorldSelect;
            }
            MainMenuAction::OpenOptions => {
                self.main_menu.options_open = true;
            }
            MainMenuAction::Quit => {
                self.quit_requested = true;
            }
            MainMenuAction::BackToMain => {
                self.app_state = AppState::MainMenu;
            }
            MainMenuAction::OpenWorldCreate => {
                self.app_state = AppState::WorldCreate;
            }
            MainMenuAction::PlayWorld(folder) => {
                self.load_saved_world(folder);
            }
            MainMenuAction::DeleteWorld(folder) => {
                if let Err(err) = delete_world(&folder) {
                    log::error!("[Save] Failed to delete world '{}': {}", folder, err);
                }
                self.main_menu.refresh_worlds();
            }
            MainMenuAction::CreateWorld {
                name,
                seed_text,
                creative,
            } => {
                self.start_new_world(name, seed_text, creative);
            }
        }

        self.gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
}

// ─── WGPU Init ───────────────────────────────────────────────────────────────

async fn init_gpu(window: Arc<Window>) -> GpuState {
    let size = window.inner_size();

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    // SAFETY: `window` is owned by Engine which lives for the entire
    // program duration, so the surface lifetime is valid.
    let surface = instance
        .create_surface(Arc::clone(&window))
        .expect("Failed to create wgpu surface");

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .expect("No suitable GPU adapter found");

    log::info!("GPU adapter: {}", adapter.get_info().name);

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("voxel_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .expect("Failed to acquire wgpu device");

    let caps = surface.get_capabilities(&adapter);
    let format = caps
        .formats
        .iter()
        .find(|f| f.is_srgb())
        .copied()
        .unwrap_or(caps.formats[0]);

    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);

    // Camera uniform buffer.
    let cam_placeholder = Camera::new(size.width as f32 / size.height.max(1) as f32);
    let sky_placeholder = SkyState::new();
    let default_settings = GraphicsSettings::default();
    let cam_uniform = CameraUniform::from_camera_and_sky(
        &cam_placeholder,
        &sky_placeholder,
        false,
        0.0,
        &default_settings,
    );
    let camera_buffer = create_camera_buffer(&device, cam_uniform.as_bytes());

    let cam_bgl = create_camera_bind_group_layout(&device);
    let camera_bind_group = create_camera_bind_group(&device, &cam_bgl, &camera_buffer);

    let shadows = ShadowRenderer::new(&device, &cam_bgl);
    let clouds = CloudRenderer::new(&device, &cam_bgl);
    let depth_texture =
        DepthTexture::create(&device, size.width.max(1), size.height.max(1), "depth_texture");
    let scene_depth_copy = DepthTexture::create(
        &device,
        size.width.max(1),
        size.height.max(1),
        "scene_depth_copy",
    );

    let block_textures = BlockTextureArray::load(&device, &queue);
    let scene_env_bgl = create_scene_env_bind_group_layout(&device);
    let scene_env_bg = create_scene_env_bind_group(
        &device,
        &scene_env_bgl,
        &shadows.view,
        &shadows.sampler,
        &scene_depth_copy.view,
        &block_textures.view,
        &block_textures.sampler,
    );

    let sky_pipeline = create_sky_pipeline(&device, &cam_bgl);
    let render_pipeline = create_render_pipeline(&device, &cam_bgl, &scene_env_bgl);
    let translucent_pipeline = create_translucent_pipeline(&device, &cam_bgl, &scene_env_bgl);
    let postprocess = PostProcessor::new(
        &device,
        format,
        size.width.max(1),
        size.height.max(1),
        &scene_depth_copy,
        &camera_buffer,
    );

    let world = World::generate(&device);
    let overlay = OverlayRenderer::new(&device, format, &cam_bgl, size.width.max(1), size.height.max(1));
    let ui = UiSystem::new(&device, format);

    GpuState {
        surface,
        device,
        queue,
        config,
        sky_pipeline,
        render_pipeline,
        translucent_pipeline,
        depth_texture,
        scene_depth_copy,
        shadows,
        clouds,
        block_textures,
        scene_env_bgl,
        scene_env_bg,
        postprocess,
        camera_buffer,
        camera_bind_group,
        world,
        overlay,
        ui,
    }
}

// ─── Public entry point ───────────────────────────────────────────────────────

pub async fn run() {
    let event_loop = EventLoop::new().expect("Failed to create event loop");

    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Voxel Industrial Engine | FPS: 0")
            .with_inner_size(PhysicalSize::new(1280u32, 720u32))
            .with_resizable(true)
            .build(&event_loop)
            .expect("Failed to create window"),
    );

    let gpu = init_gpu(Arc::clone(&window)).await;

    let mut sys = System::new();
    sys.refresh_memory();
    let pid = sysinfo::get_current_pid().ok();

    let initial_size = window.inner_size();
    let mut engine = Engine {
        window,
        gpu,
        app_state: AppState::MainMenu,
        main_menu: MainMenuState::new(),
        active_world_folder: None,
        active_world_name: "New Industrial World".to_string(),
        active_world_seed: 0,
        active_world_creative: true,
        autosave_accum: 0.0,
        camera: Camera::new(initial_size.width as f32 / initial_size.height.max(1) as f32),
        camera_ctrl: CameraController::new(),
        sky: SkyState::new(),
        player: Player::new(Vec3::new(0.5, 36.0, 0.5)),
        inventory: PlayerInventory::new(),
        settings: GraphicsSettings::default(),
        quit_requested: false,
        target: None,
        tick_accum: 0.0,
        elapsed_secs: 0.0,
        last_frame: Instant::now(),
        last_dt_secs: 1.0 / 60.0,
        fps_counter: FpsCounter::new(),
        sys,
        pid,
        used_process_mb: 0,
        total_system_mb: 1,
        mem_percent: 0.0,
        cursor_grabbed: false,
    };

    engine.refresh_memory_stats();
    engine.refresh_title();
    log::info!(
        "Engine ready. Click to capture mouse, E to open inventory, RMB on machine to open UI."
    );

    // winit 0.29 closure signature: (Event<()>, &EventLoopWindowTarget<()>)
    // Control flow via elwt.exit() / elwt.set_control_flow().
    event_loop
        .run(move |event, elwt| {
            // Busy-loop: always Poll.
            elwt.set_control_flow(ControlFlow::Poll);

            match event {
                // ── Window events ─────────────────────────────────────────
                Event::WindowEvent { event: wev, window_id }
                    if window_id == engine.window.id() =>
                {
                    match wev {
                        WindowEvent::CloseRequested => {
                            if matches!(engine.app_state, AppState::InGame | AppState::Paused) {
                                engine.save_active_world();
                            }
                            log::info!("Close requested.");
                            elwt.exit();
                        }

                        WindowEvent::Resized(size) => {
                            engine.resize(size);
                        }

                        WindowEvent::ScaleFactorChanged { .. } => {}

                        WindowEvent::Focused(false) => {
                            engine.camera_ctrl.clear_input();
                            engine.player.reset_inputs();
                            engine.grab_cursor(false);
                        }

                        WindowEvent::CursorMoved { position, .. } => {
                            let sf = engine.window.scale_factor() as f32;
                            engine.gpu.ui.on_cursor_moved(position.x, position.y, sf);
                        }

                        WindowEvent::MouseInput { state, button, .. } => {
                            engine.handle_mouse_button(button, state);
                        }

                        WindowEvent::MouseWheel { delta, .. } => {
                            engine.handle_scroll(&delta);
                        }

                        WindowEvent::KeyboardInput { event: key_event, .. } => {
                            engine.handle_key(&key_event);
                        }

                        _ => {}
                    }
                }

                // ── Raw device input (mouse look) ─────────────────────────
                Event::DeviceEvent { event: dev, .. } => {
                    if engine.cursor_grabbed && !engine.gpu.ui.is_open() {
                        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = dev {
                            engine.camera_ctrl.process_mouse_motion(dx, dy);
                        }
                    }
                }

                // ── Main loop tick (winit 0.29: AboutToWait) ──────────────
                Event::AboutToWait => {
                    engine.update();

                    match engine.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            let size = engine.window.inner_size();
                            engine.resize(size);
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            log::error!("GPU out of memory — exiting.");
                            elwt.exit();
                        }
                        Err(e) => log::warn!("Frame skip: {:?}", e),
                    }

                    if engine.quit_requested {
                        log::info!("Quit requested from Pause Menu.");
                        elwt.exit();
                    }
                }

                _ => {}
            }
        })
        .expect("Event loop error");
}
