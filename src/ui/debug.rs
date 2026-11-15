// F3 Debug Overlay (`src/ui/debug.rs`):
//   • Top-left: engine version, FPS + frame time (ms), XYZ coordinates, block & local chunk
//     coordinates, chunk [cx, cz], and cardinal facing direction (North/South/East/West + Yaw/Pitch).
//   • Top-right: DDA raycast targeted block (`[ID: Name] at [X, Y, Z]`), active chunks (`256 (16x16)`),
//     loaded fluids count, energy networks count, and primed TNT count.

use egui::{Align2, Color32, Context, FontId, Id, Vec2};
use glam::{IVec3, Vec3};

use crate::voxel::chunk::{block_name, BlockId, CHUNK_SIZE};

pub struct DebugInfo {
    pub fps: f64,
    pub frame_time_ms: f32,
    pub used_process_mb: u64,
    pub total_system_mb: u64,
    pub mem_percent: f32,
    pub player_pos: Vec3,
    pub yaw_deg: f32,
    pub pitch_deg: f32,
    pub flying: bool,
    pub day_time: f32,
    pub day_phase: &'static str,
    pub targeted_block: Option<(IVec3, BlockId)>,
    pub active_chunks: usize,
    pub loaded_fluids: usize,
    pub energy_nets: usize,
    pub machines_count: usize,
    pub cables_count: usize,
    pub primed_tnt: usize,
}

fn cardinal_direction(yaw_rad: f32) -> &'static str {
    // Camera forward is `(cos(yaw), *, sin(yaw))`.
    let (sin, cos) = yaw_rad.sin_cos();
    if cos.abs() >= sin.abs() {
        if cos >= 0.0 {
            "East (+X)"
        } else {
            "West (-X)"
        }
    } else if sin >= 0.0 {
        "South (+Z)"
    } else {
        "North (-Z)"
    }
}

pub fn draw_debug_overlay(ctx: &Context, info: &DebugInfo) {
    let n = CHUNK_SIZE as i32;
    let bx = info.player_pos.x.floor() as i32;
    let by = info.player_pos.y.floor() as i32;
    let bz = info.player_pos.z.floor() as i32;

    let chunk_x = bx.div_euclid(n);
    let chunk_z = bz.div_euclid(n);
    let lx = bx.rem_euclid(n);
    let ly = by.rem_euclid(n);
    let lz = bz.rem_euclid(n);

    let yaw_rad = info.yaw_deg.to_radians();
    let facing = cardinal_direction(yaw_rad);
    let mode_str = if info.flying { "Flying" } else { "Walking" };

    // ── Top-Left Diagnostic Panel (no background window, pure monospace text) ──
    egui::Area::new(Id::new("f3_debug_top_left"))
        .anchor(Align2::LEFT_TOP, Vec2::new(10.0, 10.0))
        .interactable(false)
        .show(ctx, |ui| {
            egui::Frame::none().show(ui, |ui| {
                let lines = [
                    "Voxel Engine v0.1.0".to_string(),
                    format!(
                        "FPS: {:.0} (frame time: {:.2} ms)",
                        info.fps, info.frame_time_ms
                    ),
                    format!(
                        "Mem: {} MB / {} MB ({:.1}%)",
                        info.used_process_mb, info.total_system_mb, info.mem_percent
                    ),
                    format!(
                        "XYZ: {:.3} / {:.3} / {:.3} ({})",
                        info.player_pos.x, info.player_pos.y, info.player_pos.z, mode_str
                    ),
                    format!("Block: [{bx}, {by}, {bz}] (in chunk: {lx}, {ly}, {lz})"),
                    format!("Chunk: [{chunk_x}, {chunk_z}]"),
                    format!(
                        "Facing: {facing} (Yaw: {:.1}, Pitch: {:.1})",
                        info.yaw_deg, info.pitch_deg
                    ),
                    format!(
                        "Time: {:.1}s / 1200.0s ({}) [Press T to step time]",
                        info.day_time, info.day_phase
                    ),
                ];

                for line in lines {
                    draw_crisp_mono_line(ui, &line);
                }
            });
        });

    // ── Top-Right World & Target Panel (no background window, pure monospace text) ──
    egui::Area::new(Id::new("f3_debug_top_right"))
        .anchor(Align2::RIGHT_TOP, Vec2::new(-10.0, 10.0))
        .interactable(false)
        .show(ctx, |ui| {
            egui::Frame::none().show(ui, |ui| {
                ui.with_layout(egui::Layout::top_down(egui::Align::Max), |ui| {
                    let target_str = match info.targeted_block {
                        Some((p, id)) => format!(
                            "Targeted Block: [{}: {}] at [{}, {}, {}]",
                            id,
                            block_name(id),
                            p.x,
                            p.y,
                            p.z
                        ),
                        None => "Targeted Block: None".to_string(),
                    };

                    let right_lines = [
                        target_str,
                        format!("Active Chunks: {} (16x16)", info.active_chunks),
                        format!(
                            "Loaded Fluids: {} | Energy Nets: {} (Machines: {}, Cables: {})",
                            info.loaded_fluids,
                            info.energy_nets,
                            info.machines_count,
                            info.cables_count
                        ),
                        format!("Primed TNT: {}", info.primed_tnt),
                    ];

                    for line in right_lines {
                        draw_crisp_mono_line(ui, &line);
                    }
                });
            });
        });
}

fn draw_crisp_mono_line(ui: &mut egui::Ui, text: &str) {
    ui.label(
        egui::RichText::new(text)
            .font(FontId::monospace(13.5))
            .color(Color32::from_rgb(240, 245, 250)),
    );
}
