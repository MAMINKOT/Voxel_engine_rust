// Pause Menu (`Esc`) and Live Graphics Settings (`GraphicsSettings`) rendered via `egui`.

use egui::{Align2, Color32, Context, RichText, Vec2};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ShadowQuality {
    Off,
    Low,
    High,
}

impl ShadowQuality {
    pub fn as_shader_code(self) -> f32 {
        match self {
            Self::Off => 0.0,
            Self::Low => 1.0,
            Self::High => 2.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GraphicsSettings {
    /// Field of view in degrees (`60.0..=110.0`).
    pub fov_deg: f32,
    /// Render distance in chunks (`4..=16`).
    pub render_distance_chunks: i32,
    /// Directional sun shadow quality (`Off`, `Low`, `High (16-tap Poisson PCF)`).
    pub shadow_quality: ShadowQuality,
    /// Enable/disable HDR Bloom on emissive light sources (`Lava`, `Torch`, `Fire`, active machines).
    pub bloom_enabled: bool,
    /// Bloom mix strength (`0.0..=1.0`).
    pub bloom_intensity: f32,
    /// Enable/disable the procedural cloud layer at `Y = 128.0` and cloud ground shadows.
    pub clouds_enabled: bool,
    /// Enable/disable vertex wave displacement on water surfaces.
    pub water_waves: bool,
}

impl Default for GraphicsSettings {
    fn default() -> Self {
        Self {
            fov_deg: 70.0,
            render_distance_chunks: 12,
            shadow_quality: ShadowQuality::High,
            bloom_enabled: true,
            bloom_intensity: 0.35,
            clouds_enabled: true,
            water_waves: true,
        }
    }
}

impl GraphicsSettings {
    pub fn effective_bloom_strength(&self) -> f32 {
        if self.bloom_enabled {
            self.bloom_intensity.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PauseMenuAction {
    None,
    Resume,
    OpenOptions,
    BackToPause,
    Quit,
}

pub fn draw_pause_menu(ctx: &Context) -> PauseMenuAction {
    let mut action = PauseMenuAction::None;

    egui::Window::new("⏸ Меню Паузи (Pause)")
        .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
        .collapsible(false)
        .resizable(false)
        .default_width(300.0)
        .show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(8.0);
                let btn_size = Vec2::new(260.0, 38.0);

                if ui
                    .add_sized(
                        btn_size,
                        egui::Button::new(
                            RichText::new("▶ Продовжити гру (Resume)")
                                .size(15.0)
                                .color(Color32::WHITE),
                        ),
                    )
                    .clicked()
                {
                    action = PauseMenuAction::Resume;
                }

                ui.add_space(8.0);
                if ui
                    .add_sized(
                        btn_size,
                        egui::Button::new(
                            RichText::new("⚙ Налаштування (Options)")
                                .size(15.0)
                                .color(Color32::from_rgb(220, 230, 250)),
                        ),
                    )
                    .clicked()
                {
                    action = PauseMenuAction::OpenOptions;
                }

                ui.add_space(8.0);
                if ui
                    .add_sized(
                        btn_size,
                        egui::Button::new(
                            RichText::new("✖ Вийти на робочий стіл (Quit)")
                                .size(15.0)
                                .color(Color32::from_rgb(255, 140, 140)),
                        ),
                    )
                    .clicked()
                {
                    action = PauseMenuAction::Quit;
                }
                ui.add_space(6.0);
            });
        });

    action
}

pub fn draw_options_menu(ctx: &Context, settings: &mut GraphicsSettings) -> PauseMenuAction {
    let mut action = PauseMenuAction::None;

    egui::Window::new("⚙ Налаштування Графіки (Graphics Settings)")
        .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
        .collapsible(false)
        .resizable(false)
        .default_width(420.0)
        .show(ctx, |ui| {
            ui.add_space(4.0);

            // 1. FOV Slider (60° .. 110°)
            ui.horizontal(|ui| {
                ui.label(RichText::new("Кут огляду (FOV):").size(14.0));
                ui.add(
                    egui::Slider::new(&mut settings.fov_deg, 60.0..=110.0)
                        .suffix("°")
                        .step_by(1.0),
                );
            });
            ui.add_space(6.0);

            // 2. Render Distance (4 .. 16 chunks)
            ui.horizontal(|ui| {
                ui.label(RichText::new("Дальність промальовки:").size(14.0));
                ui.add(
                    egui::Slider::new(&mut settings.render_distance_chunks, 4..=16)
                        .suffix(" чанків"),
                );
            });
            ui.add_space(6.0);

            // 3. Shadows [Вимк / Низькі / Високі (PCF)]
            ui.horizontal(|ui| {
                ui.label(RichText::new("Тіні (Shadows):").size(14.0));
                ui.selectable_value(
                    &mut settings.shadow_quality,
                    ShadowQuality::Off,
                    "Вимк",
                );
                ui.selectable_value(
                    &mut settings.shadow_quality,
                    ShadowQuality::Low,
                    "Низькі",
                );
                ui.selectable_value(
                    &mut settings.shadow_quality,
                    ShadowQuality::High,
                    "Високі (PCF)",
                );
            });
            ui.add_space(6.0);

            // 4. Bloom checkbox + intensity slider (0.0 .. 1.0)
            ui.checkbox(
                &mut settings.bloom_enabled,
                RichText::new("Bloom (Світіння лави, факелів, вогню та печей)").size(14.0),
            );
            if settings.bloom_enabled {
                ui.horizontal(|ui| {
                    ui.add_space(20.0);
                    ui.label("Інтенсивність Bloom:");
                    ui.add(
                        egui::Slider::new(&mut settings.bloom_intensity, 0.0..=1.0).step_by(0.02),
                    );
                });
            }
            ui.add_space(6.0);

            // 5. Clouds checkbox
            ui.checkbox(
                &mut settings.clouds_enabled,
                RichText::new("Хмари (Clouds Y=128 + рухомі тіні)").size(14.0),
            );
            ui.add_space(6.0);

            // 6. Water Waves checkbox
            ui.checkbox(
                &mut settings.water_waves,
                RichText::new("Вода з хвилями (Vertex Waves)").size(14.0),
            );

            ui.separator();
            ui.vertical_centered(|ui| {
                if ui
                    .add_sized(
                        Vec2::new(220.0, 34.0),
                        egui::Button::new(RichText::new("⬅ Назад (Back)").size(14.0)),
                    )
                    .clicked()
                {
                    action = PauseMenuAction::BackToPause;
                }
            });
        });

    action
}
