// Main Menu, World Select, World Create, and Loading screens rendered with `egui`.

use egui::{Align2, Color32, Context, RichText, Rounding, ScrollArea, Stroke, Vec2};

use crate::save::{list_saved_worlds, SavedWorldSummary};

#[derive(Clone, Debug, PartialEq)]
pub enum MainMenuAction {
    None,
    OpenWorldSelect,
    OpenOptions,
    Quit,
    BackToMain,
    OpenWorldCreate,
    PlayWorld(String),
    DeleteWorld(String),
    CreateWorld {
        name: String,
        seed_text: String,
        creative: bool,
    },
}

pub struct MainMenuState {
    pub worlds: Vec<SavedWorldSummary>,
    pub selected_folder: Option<String>,
    pub confirm_delete: Option<String>,
    pub new_world_name: String,
    pub new_world_seed: String,
    pub new_world_creative: bool,
    pub options_open: bool,
}

impl MainMenuState {
    pub fn new() -> Self {
        let mut s = Self {
            worlds: Vec::new(),
            selected_folder: None,
            confirm_delete: None,
            new_world_name: "New Industrial World".to_string(),
            new_world_seed: "2026".to_string(),
            new_world_creative: true,
            options_open: false,
        };
        s.refresh_worlds();
        s
    }

    pub fn refresh_worlds(&mut self) {
        self.worlds = list_saved_worlds();
        if let Some(sel) = &self.selected_folder {
            if !self.worlds.iter().any(|w| &w.folder_name == sel) {
                self.selected_folder = self.worlds.first().map(|w| w.folder_name.clone());
            }
        } else {
            self.selected_folder = self.worlds.first().map(|w| w.folder_name.clone());
        }
        self.confirm_delete = None;
    }
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.1} KB", (bytes as f64 / 1024.0).max(0.1))
    }
}

fn format_age(unix_secs: u64) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(unix_secs);
    let diff = now.saturating_sub(unix_secs);
    if diff < 60 {
        "Щойно".to_string()
    } else if diff < 3600 {
        format!("{} хв тому", diff / 60)
    } else if diff < 86400 {
        format!("{} год тому", diff / 3600)
    } else {
        format!("{} дн тому", diff / 86400)
    }
}

fn draw_backdrop(ctx: &Context) {
    let screen = ctx.screen_rect();
    let painter = ctx.layer_painter(egui::LayerId::background());
    painter.rect_filled(screen, Rounding::ZERO, Color32::from_rgba_unmultiplied(8, 12, 20, 175));
}

pub fn draw_main_menu(ctx: &Context) -> MainMenuAction {
    draw_backdrop(ctx);
    let mut action = MainMenuAction::None;

    egui::Area::new(egui::Id::new("main_menu_area"))
        .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
        .show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.label(
                    RichText::new("VOXEL INDUSTRIAL ENGINE")
                        .size(38.0)
                        .strong()
                        .color(Color32::from_rgb(245, 205, 85)),
                );
                ui.add_space(4.0);
                ui.label(
                    RichText::new("3D Voxel Sandbox & Industrial Automation")
                        .size(14.0)
                        .color(Color32::from_gray(185)),
                );
                ui.add_space(28.0);

                egui::Frame::none()
                    .fill(Color32::from_rgba_unmultiplied(18, 22, 32, 235))
                    .stroke(Stroke::new(1.5_f32, Color32::from_rgb(80, 92, 118)))
                    .rounding(Rounding::same(10.0))
                    .inner_margin(egui::Margin::same(22.0))
                    .show(ui, |ui| {
                        let btn_size = Vec2::new(280.0, 42.0);
                        if ui
                            .add_sized(
                                btn_size,
                                egui::Button::new(RichText::new("Одиночна гра (Singleplayer)").size(16.0).strong()),
                            )
                            .clicked()
                        {
                            action = MainMenuAction::OpenWorldSelect;
                        }
                        ui.add_space(10.0);
                        if ui
                            .add_sized(
                                btn_size,
                                egui::Button::new(RichText::new("Налаштування (Options)").size(15.0)),
                            )
                            .clicked()
                        {
                            action = MainMenuAction::OpenOptions;
                        }
                        ui.add_space(10.0);
                        if ui
                            .add_sized(
                                btn_size,
                                egui::Button::new(
                                    RichText::new("Вийти (Quit)")
                                        .size(15.0)
                                        .color(Color32::from_rgb(245, 130, 130)),
                                ),
                            )
                            .clicked()
                        {
                            action = MainMenuAction::Quit;
                        }
                    });
            });
        });

    action
}

pub fn draw_world_select(ctx: &Context, state: &mut MainMenuState) -> MainMenuAction {
    draw_backdrop(ctx);
    let mut action = MainMenuAction::None;

    egui::Window::new("Вибір світу (World Select)")
        .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
        .collapsible(false)
        .resizable(false)
        .fixed_size(Vec2::new(560.0, 420.0))
        .show(ctx, |ui| {
            ui.label(
                RichText::new("Збережені світи у директорії saves/:")
                    .size(13.0)
                    .color(Color32::from_gray(190)),
            );
            ui.add_space(6.0);

            ScrollArea::vertical()
                .max_height(265.0)
                .show(ui, |ui| {
                    if state.worlds.is_empty() {
                        ui.add_space(40.0);
                        ui.vertical_centered(|ui| {
                            ui.label(
                                RichText::new("Збережених світів поки немає.")
                                    .size(15.0)
                                    .color(Color32::from_gray(150)),
                            );
                            ui.label(
                                RichText::new("Натисніть «Створити новий світ», щоб почати.")
                                    .size(13.0)
                                    .color(Color32::from_gray(130)),
                            );
                        });
                    } else {
                        for w in &state.worlds {
                            let selected = state.selected_folder.as_deref() == Some(&w.folder_name);
                            let fill = if selected {
                                Color32::from_rgb(44, 58, 84)
                            } else {
                                Color32::from_rgb(28, 32, 42)
                            };
                            let stroke = if selected {
                                Stroke::new(1.5_f32, Color32::from_rgb(245, 205, 85))
                            } else {
                                Stroke::new(1.0_f32, Color32::from_rgb(65, 70, 84))
                            };
                            let resp = egui::Frame::none()
                                .fill(fill)
                                .stroke(stroke)
                                .rounding(Rounding::same(6.0))
                                .inner_margin(egui::Margin::symmetric(12.0, 8.0))
                                .show(ui, |ui| {
                                    ui.set_width(510.0);
                                    ui.horizontal(|ui| {
                                        ui.vertical(|ui| {
                                            ui.label(
                                                RichText::new(&w.world_name)
                                                    .size(16.0)
                                                    .strong()
                                                    .color(Color32::WHITE),
                                            );
                                            let mode = if w.game_mode_creative {
                                                "Creative"
                                            } else {
                                                "Survival"
                                            };
                                            ui.label(
                                                RichText::new(format!(
                                                    "Seed: {} • {} • {} • {}",
                                                    w.seed,
                                                    mode,
                                                    format_size(w.size_bytes),
                                                    format_age(w.last_saved_secs)
                                                ))
                                                .size(12.0)
                                                .color(Color32::from_gray(170)),
                                            );
                                        });
                                    });
                                })
                                .response
                                .interact(egui::Sense::click());
                            if resp.clicked() {
                                state.selected_folder = Some(w.folder_name.clone());
                                state.confirm_delete = None;
                            }
                            if resp.double_clicked() {
                                action = MainMenuAction::PlayWorld(w.folder_name.clone());
                            }
                            ui.add_space(4.0);
                        }
                    }
                });

            ui.separator();
            ui.add_space(4.0);

            if let Some(target_del) = state.confirm_delete.clone() {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(format!("Видалити світ '{}' назавжди?", target_del))
                            .color(Color32::from_rgb(255, 140, 140)),
                    );
                    if ui.button("Підтвердити видалення").clicked() {
                        action = MainMenuAction::DeleteWorld(target_del);
                        state.confirm_delete = None;
                    }
                    if ui.button("Скасувати").clicked() {
                        state.confirm_delete = None;
                    }
                });
            } else {
                ui.horizontal(|ui| {
                    let has_sel = state.selected_folder.is_some();
                    if ui
                        .add_enabled(
                            has_sel,
                            egui::Button::new(RichText::new("Грати у вибраному світі").strong())
                                .min_size(Vec2::new(175.0, 32.0)),
                        )
                        .clicked()
                    {
                        if let Some(f) = &state.selected_folder {
                            action = MainMenuAction::PlayWorld(f.clone());
                        }
                    }
                    if ui
                        .add(
                            egui::Button::new("Створити новий світ")
                                .min_size(Vec2::new(150.0, 32.0)),
                        )
                        .clicked()
                    {
                        action = MainMenuAction::OpenWorldCreate;
                    }
                    if ui
                        .add_enabled(
                            has_sel,
                            egui::Button::new("Видалити").min_size(Vec2::new(90.0, 32.0)),
                        )
                        .clicked()
                    {
                        state.confirm_delete = state.selected_folder.clone();
                    }
                    if ui
                        .add(egui::Button::new("Назад").min_size(Vec2::new(80.0, 32.0)))
                        .clicked()
                    {
                        action = MainMenuAction::BackToMain;
                    }
                });
            }
        });

    action
}

pub fn draw_world_create(ctx: &Context, state: &mut MainMenuState) -> MainMenuAction {
    draw_backdrop(ctx);
    let mut action = MainMenuAction::None;

    egui::Window::new("Створення нового світу (Create World)")
        .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
        .collapsible(false)
        .resizable(false)
        .default_width(400.0)
        .show(ctx, |ui| {
            ui.label("Назва світу:");
            ui.text_edit_singleline(&mut state.new_world_name);
            ui.add_space(8.0);

            ui.label("Seed (число або текст):");
            ui.text_edit_singleline(&mut state.new_world_seed);
            ui.add_space(8.0);

            ui.horizontal(|ui| {
                ui.label("Режим гри:");
                ui.selectable_value(&mut state.new_world_creative, false, "Survival");
                ui.selectable_value(
                    &mut state.new_world_creative,
                    true,
                    "Creative (Політ)",
                );
            });

            ui.add_space(14.0);
            ui.separator();
            ui.add_space(6.0);

            ui.horizontal(|ui| {
                if ui
                    .add(
                        egui::Button::new(RichText::new("Створити світ").strong())
                            .min_size(Vec2::new(170.0, 34.0)),
                    )
                    .clicked()
                {
                    action = MainMenuAction::CreateWorld {
                        name: state.new_world_name.clone(),
                        seed_text: state.new_world_seed.clone(),
                        creative: state.new_world_creative,
                    };
                }
                if ui
                    .add(egui::Button::new("Назад").min_size(Vec2::new(110.0, 34.0)))
                    .clicked()
                {
                    action = MainMenuAction::OpenWorldSelect;
                }
            });
        });

    action
}

pub fn draw_loading_screen(ctx: &Context, progress: f32, message: &str) {
    draw_backdrop(ctx);
    egui::Window::new("Завантаження світу")
        .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
        .collapsible(false)
        .resizable(false)
        .default_width(360.0)
        .show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.label(RichText::new(message).size(15.0));
                ui.add_space(10.0);
                ui.add(egui::ProgressBar::new(progress.clamp(0.0, 1.0)).show_percentage());
            });
        });
}
