// Immediate-mode UI powered by `egui` + `egui-wgpu`:
//   • Always-visible bottom hotbar (9 textureless colored slots).
//   • Player Inventory modal (`E` / `I` / `Tab`): 2x2 hand-crafting grid + 3x9 backpack + 1x9 hotbar.
//   • Machine & Block UI modals (Right-click on `CoalGenerator`, `ElectricFurnace`, `SimpleFurnace`,
//     `AlloySmelter`, `BatBox`, `Chest`, or `Workbench`).
//   • NEI / JEI-style right sidebar: all items browser + interactive Recipe & Usage Viewer.
//   • Hover tooltips with item names and floating cursor stack icon.

use std::collections::HashMap;
use std::sync::Arc;

use egui::{
    Align2, Color32, Context, Event as EguiEvent, FontId, Id, Key, Modifiers, PointerButton, Pos2,
    RawInput, Rect, Rounding, Sense, Stroke, TextureHandle, TextureId, TextureOptions, Vec2,
};
use egui_wgpu::{Renderer as EguiRenderer, ScreenDescriptor};
use glam::IVec3;
use winit::event::{ElementState, MouseButton};

use crate::crafting::{consume_crafting_ingredients, Recipe, RecipeRegistry};
use crate::inventory::{
    interact_slot, ItemStack, PlayerInventory, HOTBAR_SIZE, MAIN_INV_COLS, MAIN_INV_ROWS, MAX_STACK,
};
use crate::render::texture::build_ui_item_icons;
use crate::sim::energy_net::EnergyNetwork;
use crate::ui::debug::{draw_debug_overlay, DebugInfo};
use crate::ui::menu::{draw_options_menu, draw_pause_menu, GraphicsSettings, PauseMenuAction};
use crate::voxel::chunk::{
    block_color, block_name, BlockId, ALL_ITEMS, ALLOY_SMELTER, BATBOX, CHEST, COAL_GENERATOR,
    COAL_ORE, ELECTRIC_FURNACE, SIMPLE_FURNACE, WORKBENCH,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum UiScreen {
    Closed,
    Inventory,
    Machine(IVec3),
    PauseMenu,
    OptionsMenu,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum NeiMode {
    Recipes,
    Usages,
}

pub struct UiSystem {
    ctx: Context,
    renderer: EguiRenderer,
    events: Vec<EguiEvent>,
    cursor_pos: Pos2,
    pub screen: UiScreen,
    pub debug_open: bool,
    nei_query: Option<(BlockId, NeiMode)>,
    #[allow(dead_code)]
    item_icons: HashMap<BlockId, TextureHandle>,
}

impl UiSystem {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let ctx = Context::default();
        let renderer = EguiRenderer::new(device, surface_format, None, 1);
        let raw_icons = build_ui_item_icons();
        let mut item_icons = HashMap::with_capacity(raw_icons.len());
        let mut icon_ids = HashMap::with_capacity(raw_icons.len());
        for (id, color_img) in raw_icons {
            let handle = ctx.load_texture(
                format!("item_icon_{}", id),
                color_img,
                TextureOptions::NEAREST,
            );
            icon_ids.insert(id, handle.id());
            item_icons.insert(id, handle);
        }
        ctx.data_mut(|d| {
            d.insert_temp(Id::new("item_icon_ids"), Arc::new(icon_ids));
        });

        Self {
            ctx,
            renderer,
            events: Vec::new(),
            cursor_pos: Pos2::ZERO,
            screen: UiScreen::Closed,
            debug_open: false,
            nei_query: None,
            item_icons,
        }
    }

    pub fn is_open(&self) -> bool {
        self.screen != UiScreen::Closed
    }

    pub fn is_paused(&self) -> bool {
        matches!(self.screen, UiScreen::PauseMenu | UiScreen::OptionsMenu)
    }

    pub fn on_cursor_moved(&mut self, x: f64, y: f64, scale_factor: f32) {
        let pos = Pos2::new(x as f32 / scale_factor, y as f32 / scale_factor);
        self.cursor_pos = pos;
        self.events.push(EguiEvent::PointerMoved(pos));
    }

    pub fn on_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        let Some(egui_btn) = (match button {
            MouseButton::Left => Some(PointerButton::Primary),
            MouseButton::Right => Some(PointerButton::Secondary),
            MouseButton::Middle => Some(PointerButton::Middle),
            _ => None,
        }) else {
            return;
        };
        self.events.push(EguiEvent::PointerButton {
            pos: self.cursor_pos,
            button: egui_btn,
            pressed: state == ElementState::Pressed,
            modifiers: Modifiers::NONE,
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        scale_factor: f32,
        dt: f32,
        inventory: &mut PlayerInventory,
        energy: &mut EnergyNetwork,
        settings: &mut GraphicsSettings,
        debug_info: Option<&DebugInfo>,
        block_at: impl Fn(IVec3) -> BlockId,
    ) -> PauseMenuAction {
        if let UiScreen::Machine(pos) = self.screen {
            if !energy.is_interactive(pos) {
                self.screen = UiScreen::Closed;
            }
        }

        let screen_desc = ScreenDescriptor {
            size_in_pixels: [width.max(1), height.max(1)],
            pixels_per_point: scale_factor.max(0.5),
        };
        let screen_points = Vec2::new(
            width as f32 / screen_desc.pixels_per_point,
            height as f32 / screen_desc.pixels_per_point,
        );

        let raw_input = RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, screen_points)),
            predicted_dt: dt,
            events: std::mem::take(&mut self.events),
            ..Default::default()
        };

        let screen = self.screen;
        let cursor_pos = self.cursor_pos;
        let mut nei_query = self.nei_query;
        let mut menu_action = PauseMenuAction::None;

        let full_output = self.ctx.run(raw_input, |ctx| {
            if let Some(info) = debug_info {
                draw_debug_overlay(ctx, info);
            }

            draw_hud_hotbar(ctx, inventory);

            match screen {
                UiScreen::Closed => {}
                UiScreen::Inventory => {
                    draw_inventory_window(ctx, inventory, &energy.recipes);
                }
                UiScreen::Machine(pos) => {
                    let kind = block_at(pos);
                    draw_machine_window(ctx, pos, kind, inventory, energy);
                }
                UiScreen::PauseMenu => {
                    menu_action = draw_pause_menu(ctx);
                }
                UiScreen::OptionsMenu => {
                    menu_action = draw_options_menu(ctx, settings);
                }
            }

            if matches!(screen, UiScreen::Inventory | UiScreen::Machine(_)) {
                draw_nei_sidebar(ctx, &energy.recipes, inventory, &mut nei_query);

                if let Some(held) = inventory.cursor_item {
                    draw_cursor_stack(ctx, cursor_pos, held);
                }
            }
        });

        self.nei_query = nei_query;

        let tris = self
            .ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);

        for (id, image_delta) in &full_output.textures_delta.set {
            self.renderer.update_texture(device, queue, *id, image_delta);
        }

        self.renderer
            .update_buffers(device, queue, encoder, &tris, &screen_desc);

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
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
            self.renderer.render(&mut pass, &tris, &screen_desc);
        }

        for id in &full_output.textures_delta.free {
            self.renderer.free_texture(id);
        }

        menu_action
    }
}

// ─── Slot, Crafting & Gauge Widgets ──────────────────────────────────────────

const SLOT_SIZE: f32 = 44.0;
const SLOT_GAP: f32 = 5.0;

fn item_color32(id: BlockId) -> Color32 {
    let [r, g, b] = block_color(id);
    Color32::from_rgb(
        (r.clamp(0.0, 1.0) * 255.0) as u8,
        (g.clamp(0.0, 1.0) * 255.0) as u8,
        (b.clamp(0.0, 1.0) * 255.0) as u8,
    )
}

fn draw_item_icon(ctx: &Context, painter: &egui::Painter, icon_rect: Rect, id: BlockId) {
    let tex_id = ctx.data(|d| {
        d.get_temp::<Arc<HashMap<BlockId, TextureId>>>(Id::new("item_icon_ids"))
            .and_then(|map| map.get(&id).copied())
    });
    if let Some(tid) = tex_id {
        painter.image(
            tid,
            icon_rect,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
    } else {
        painter.rect(
            icon_rect,
            Rounding::same(3.0),
            item_color32(id),
            Stroke::new(1.0_f32, Color32::from_black_alpha(140)),
        );
    }
}

/// Draws a single textureless colored slot and handles left/right click + tooltip on hover.
fn draw_slot_widget<F: Fn(BlockId) -> bool>(
    ui: &mut egui::Ui,
    slot: &mut Option<ItemStack>,
    cursor: &mut Option<ItemStack>,
    selected: bool,
    hint_label: Option<&str>,
    filter: F,
) {
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(SLOT_SIZE), Sense::click());

    let painter = ui.painter();
    let bg = if response.hovered() {
        Color32::from_rgb(65, 68, 78)
    } else {
        Color32::from_rgb(36, 38, 46)
    };
    let border = if selected {
        Stroke::new(2.5_f32, Color32::from_rgb(245, 210, 80))
    } else if response.hovered() {
        Stroke::new(1.5_f32, Color32::from_rgb(180, 185, 200))
    } else {
        Stroke::new(1.0_f32, Color32::from_rgb(80, 84, 96))
    };

    painter.rect(rect, Rounding::same(4.0), bg, border);

    if let Some(stack) = *slot {
        let icon_rect = rect.shrink(6.0);
        draw_item_icon(ui.ctx(), painter, icon_rect, stack.id);
        if stack.count > 1 {
            painter.text(
                rect.right_bottom() - Vec2::new(4.0, 2.0),
                Align2::RIGHT_BOTTOM,
                format!("{}", stack.count),
                FontId::monospace(13.0),
                Color32::WHITE,
            );
        }
    } else if let Some(hint) = hint_label {
        painter.text(
            rect.center(),
            Align2::CENTER_CENTER,
            hint,
            FontId::proportional(10.0),
            Color32::from_gray(120),
        );
    }

    if response.clicked_by(PointerButton::Primary) {
        interact_slot(slot, cursor, false, &filter);
    } else if response.clicked_by(PointerButton::Secondary) {
        interact_slot(slot, cursor, true, &filter);
    }

    if let Some(stack) = *slot {
        response.on_hover_text(format!("{} (x{})", block_name(stack.id), stack.count));
    } else if let Some(hint) = hint_label {
        response.on_hover_text(hint);
    }
}

/// Draws a read-only recipe slot icon (clickable to jump to that item's recipe/usage in NEI).
fn draw_readonly_item_badge(
    ui: &mut egui::Ui,
    stack: Option<ItemStack>,
    size: f32,
    nei_query: &mut Option<(BlockId, NeiMode)>,
) {
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(size), Sense::click());
    let painter = ui.painter();

    let bg = if response.hovered() && stack.is_some() {
        Color32::from_rgb(68, 72, 84)
    } else {
        Color32::from_rgb(32, 34, 42)
    };
    painter.rect(
        rect,
        Rounding::same(4.0),
        bg,
        Stroke::new(1.0_f32, Color32::from_rgb(76, 80, 92)),
    );

    if let Some(st) = stack {
        let icon_rect = rect.shrink(4.0);
        draw_item_icon(ui.ctx(), painter, icon_rect, st.id);
        if st.count > 1 {
            painter.text(
                rect.right_bottom() - Vec2::new(3.0, 1.0),
                Align2::RIGHT_BOTTOM,
                format!("{}", st.count),
                FontId::monospace(11.0),
                Color32::WHITE,
            );
        }
        if response.clicked_by(PointerButton::Primary) {
            *nei_query = Some((st.id, NeiMode::Recipes));
        } else if response.clicked_by(PointerButton::Secondary) {
            *nei_query = Some((st.id, NeiMode::Usages));
        }
        response.on_hover_text(format!(
            "{} (x{})\nLMB: Recipes • RMB: Usages",
            block_name(st.id),
            st.count
        ));
    }
}

/// Draws an interactive `grid_w x grid_h` crafting grid + result output slot.
fn draw_crafting_widget(
    ui: &mut egui::Ui,
    grid: &mut [Option<ItemStack>],
    grid_w: usize,
    grid_h: usize,
    cursor: &mut Option<ItemStack>,
    recipes: &RecipeRegistry,
) {
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            for r in 0..grid_h {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing = Vec2::splat(SLOT_GAP);
                    for c in 0..grid_w {
                        let idx = r * grid_w + c;
                        draw_slot_widget(ui, &mut grid[idx], cursor, false, None, |_| true);
                    }
                });
                if r + 1 < grid_h {
                    ui.add_space(SLOT_GAP);
                }
            }
        });

        ui.add_space(16.0);
        ui.vertical(|ui| {
            ui.add_space(((grid_h as f32 - 1.0) * (SLOT_SIZE + SLOT_GAP)) * 0.5 + 10.0);
            ui.label(
                egui::RichText::new("➡")
                    .size(22.0)
                    .color(Color32::from_gray(200)),
            );
        });
        ui.add_space(16.0);

        let matched = recipes.match_crafting(grid, grid_w, grid_h);
        ui.vertical(|ui| {
            ui.add_space(((grid_h as f32 - 1.0) * (SLOT_SIZE + SLOT_GAP)) * 0.5);
            let (rect, response) =
                ui.allocate_exact_size(Vec2::splat(SLOT_SIZE + 6.0), Sense::click());
            let painter = ui.painter();
            let bg = if response.hovered() && matched.is_some() {
                Color32::from_rgb(75, 82, 68)
            } else {
                Color32::from_rgb(40, 44, 52)
            };
            painter.rect(
                rect,
                Rounding::same(5.0),
                bg,
                Stroke::new(1.5_f32, Color32::from_rgb(210, 180, 75)),
            );

            if let Some(out) = matched {
                let icon_rect = rect.shrink(6.0);
                draw_item_icon(ui.ctx(), painter, icon_rect, out.id);
                if out.count > 1 {
                    painter.text(
                        rect.right_bottom() - Vec2::new(4.0, 2.0),
                        Align2::RIGHT_BOTTOM,
                        format!("{}", out.count),
                        FontId::monospace(13.0),
                        Color32::WHITE,
                    );
                }
                if response.clicked_by(PointerButton::Primary)
                    || response.clicked_by(PointerButton::Secondary)
                {
                    let can_take = match *cursor {
                        None => true,
                        Some(cur) => cur.id == out.id && cur.count + out.count <= MAX_STACK,
                    };
                    if can_take {
                        match cursor {
                            None => *cursor = Some(out),
                            Some(cur) => cur.count += out.count,
                        }
                        consume_crafting_ingredients(grid);
                    }
                }
                response.on_hover_text(format!(
                    "Craft: {} (x{})\nClick to craft",
                    block_name(out.id),
                    out.count
                ));
            } else {
                painter.text(
                    rect.center(),
                    Align2::CENTER_CENTER,
                    "Out",
                    FontId::proportional(10.0),
                    Color32::from_gray(115),
                );
                response.on_hover_text("Crafting Output");
            }
        });
    });
}

/// Draws a vertical EU charge gauge (`stored_eu / max_eu`) with hover tooltip.
fn draw_vertical_eu_bar(ui: &mut egui::Ui, stored_eu: u32, max_eu: u32, height: f32) {
    let size = Vec2::new(22.0, height);
    let (rect, response) = ui.allocate_exact_size(size, Sense::hover());
    let painter = ui.painter();

    painter.rect(
        rect,
        Rounding::same(4.0),
        Color32::from_rgb(28, 30, 38),
        Stroke::new(1.0_f32, Color32::from_rgb(90, 96, 110)),
    );

    let frac = (stored_eu as f32 / max_eu.max(1) as f32).clamp(0.0, 1.0);
    if frac > 0.0 {
        let inner = rect.shrink(2.0);
        let fill_h = inner.height() * frac;
        let fill_rect = Rect::from_min_max(
            Pos2::new(inner.min.x, inner.max.y - fill_h),
            inner.max,
        );
        painter.rect_filled(
            fill_rect,
            Rounding::same(2.0),
            Color32::from_rgb(55, 200, 145),
        );
    }

    response.on_hover_text(format!("Energy Buffer: {} / {} EU", stored_eu, max_eu));
}

/// Bottom-center HUD hotbar (always visible).
fn draw_hud_hotbar(ctx: &Context, inventory: &mut PlayerInventory) {
    egui::Area::new(Id::new("hud_hotbar"))
        .anchor(Align2::CENTER_BOTTOM, Vec2::new(0.0, -14.0))
        .interactable(false)
        .show(ctx, |ui| {
            egui::Frame::none()
                .fill(Color32::from_black_alpha(170))
                .rounding(Rounding::same(6.0))
                .inner_margin(egui::Margin::same(6.0))
                .stroke(Stroke::new(1.0_f32, Color32::from_gray(90)))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing = Vec2::splat(SLOT_GAP);
                        let mut dummy_cursor = None;
                        for i in 0..HOTBAR_SIZE {
                            let is_sel = i == inventory.selected_slot;
                            let mut copy = inventory.hotbar[i];
                            draw_slot_widget(
                                ui,
                                &mut copy,
                                &mut dummy_cursor,
                                is_sel,
                                None,
                                |_| true,
                            );
                        }
                    });
                });
        });
}

/// Draws the 3x9 backpack + 1x9 hotbar inside an open modal window.
fn draw_player_inventory_grid(ui: &mut egui::Ui, inventory: &mut PlayerInventory) {
    ui.label(
        egui::RichText::new("Inventory (3x9)")
            .size(13.0)
            .color(Color32::from_gray(190)),
    );
    ui.add_space(2.0);

    for row in 0..MAIN_INV_ROWS {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing = Vec2::splat(SLOT_GAP);
            for col in 0..MAIN_INV_COLS {
                let idx = row * MAIN_INV_COLS + col;
                draw_slot_widget(
                    ui,
                    &mut inventory.main[idx],
                    &mut inventory.cursor_item,
                    false,
                    None,
                    |_| true,
                );
            }
        });
        ui.add_space(SLOT_GAP);
    }

    ui.add_space(6.0);
    ui.label(
        egui::RichText::new("Hotbar (1-9)")
            .size(13.0)
            .color(Color32::from_gray(190)),
    );
    ui.add_space(2.0);

    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing = Vec2::splat(SLOT_GAP);
        for i in 0..HOTBAR_SIZE {
            let is_sel = i == inventory.selected_slot;
            draw_slot_widget(
                ui,
                &mut inventory.hotbar[i],
                &mut inventory.cursor_item,
                is_sel,
                None,
                |_| true,
            );
        }
    });
}

fn draw_inventory_window(
    ctx: &Context,
    inventory: &mut PlayerInventory,
    recipes: &RecipeRegistry,
) {
    egui::Window::new("Player Inventory")
        .anchor(Align2::CENTER_CENTER, Vec2::new(-110.0, 0.0))
        .collapsible(false)
        .resizable(false)
        .show(ctx, |ui| {
            ui.label(
                egui::RichText::new("Crafting (2x2)")
                    .size(13.0)
                    .color(Color32::from_gray(190)),
            );
            ui.add_space(2.0);
            draw_crafting_widget(
                ui,
                &mut inventory.craft_2x2,
                2,
                2,
                &mut inventory.cursor_item,
                recipes,
            );
            ui.separator();
            draw_player_inventory_grid(ui, inventory);
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new("LMB: take/place stack • RMB: split/place 1 • E/Esc: close")
                    .size(11.0)
                    .color(Color32::from_gray(140)),
            );
        });
}

fn draw_machine_window(
    ctx: &Context,
    pos: IVec3,
    kind: BlockId,
    inventory: &mut PlayerInventory,
    energy: &mut EnergyNetwork,
) {
    let title = match kind {
        COAL_GENERATOR => "Coal Generator (10 EU/t -> 32 EU/t out)",
        ELECTRIC_FURNACE => "Electric Furnace (3 EU/t • 100t)",
        SIMPLE_FURNACE => "Stone Furnace (Coal-Fired • 160t)",
        ALLOY_SMELTER => "Alloy Smelter (Bronze: 3 Cu + 1 Sn • Steel: 1 Fe + 2 Coal)",
        BATBOX => "BatBox Energy Storage (40,000 EU)",
        CHEST => "Storage Chest (27 Slots)",
        WORKBENCH => "Workbench (3x3 Crafting Table)",
        _ => "Machine",
    };

    egui::Window::new(title)
        .anchor(Align2::CENTER_CENTER, Vec2::new(-110.0, 0.0))
        .collapsible(false)
        .resizable(false)
        .show(ctx, |ui| {
            match kind {
                COAL_GENERATOR => {
                    if let Some((mut fuel, burn_ticks, max_burn, stored_eu, max_eu)) =
                        energy.generator_ui_data(pos)
                    {
                        ui.horizontal(|ui| {
                            draw_vertical_eu_bar(ui, stored_eu, max_eu, 68.0);
                            ui.add_space(10.0);

                            ui.vertical(|ui| {
                                ui.label("Fuel (Coal)");
                                draw_slot_widget(
                                    ui,
                                    &mut fuel,
                                    &mut inventory.cursor_item,
                                    false,
                                    Some("Coal"),
                                    |id| id == COAL_ORE,
                                );
                            });

                            ui.add_space(12.0);
                            ui.vertical(|ui| {
                                let burn_frac =
                                    (burn_ticks as f32 / max_burn.max(1) as f32).clamp(0.0, 1.0);
                                ui.add(
                                    egui::ProgressBar::new(burn_frac)
                                        .desired_width(220.0)
                                        .fill(Color32::from_rgb(235, 110, 35))
                                        .text(format!(
                                            "🔥 Burn Time: {} / {} ticks",
                                            burn_ticks, max_burn
                                        )),
                                );
                                ui.add_space(6.0);
                                let eu_frac =
                                    (stored_eu as f32 / max_eu.max(1) as f32).clamp(0.0, 1.0);
                                ui.add(
                                    egui::ProgressBar::new(eu_frac)
                                        .desired_width(220.0)
                                        .fill(Color32::from_rgb(55, 190, 135))
                                        .text(format!("⚡ Buffer: {} / {} EU", stored_eu, max_eu)),
                                );
                            });
                        });

                        energy.set_generator_fuel(pos, fuel);
                    }
                }

                ELECTRIC_FURNACE => {
                    if let Some((mut input, mut out, progress, max_prog, stored_eu, max_eu)) =
                        energy.furnace_ui_data(pos)
                    {
                        ui.horizontal(|ui| {
                            draw_vertical_eu_bar(ui, stored_eu, max_eu, 68.0);
                            ui.add_space(10.0);

                            ui.vertical(|ui| {
                                ui.label("Input (Ore)");
                                draw_slot_widget(
                                    ui,
                                    &mut input,
                                    &mut inventory.cursor_item,
                                    false,
                                    Some("Ore"),
                                    |id| energy.recipes.match_smelting(id).is_some(),
                                );
                            });

                            ui.add_space(12.0);
                            ui.vertical(|ui| {
                                ui.label(format!(
                                    "⚡ Internal Buffer: {} / {} EU",
                                    stored_eu, max_eu
                                ));
                                let frac =
                                    (progress as f32 / max_prog.max(1) as f32).clamp(0.0, 1.0);
                                ui.add(
                                    egui::ProgressBar::new(frac)
                                        .desired_width(175.0)
                                        .fill(Color32::from_rgb(235, 145, 45))
                                        .text(format!("➡ Smelting: {} / {} t", progress, max_prog)),
                                );
                            });

                            ui.add_space(12.0);
                            ui.vertical(|ui| {
                                ui.label("Output (Ingot)");
                                draw_slot_widget(
                                    ui,
                                    &mut out,
                                    &mut inventory.cursor_item,
                                    false,
                                    Some("Output"),
                                    |_| false,
                                );
                            });
                        });

                        energy.set_furnace_slots(pos, input, out);
                    }
                }

                SIMPLE_FURNACE => {
                    if let Some((
                        mut input,
                        mut fuel,
                        mut out,
                        burn_ticks,
                        max_burn,
                        progress,
                        max_prog,
                    )) = energy.simple_furnace_ui_data(pos)
                    {
                        ui.horizontal(|ui| {
                            ui.vertical(|ui| {
                                ui.label("Input (Ore)");
                                draw_slot_widget(
                                    ui,
                                    &mut input,
                                    &mut inventory.cursor_item,
                                    false,
                                    Some("Ore"),
                                    |id| energy.recipes.match_smelting(id).is_some(),
                                );
                                ui.add_space(4.0);
                                ui.label("Fuel (Coal)");
                                draw_slot_widget(
                                    ui,
                                    &mut fuel,
                                    &mut inventory.cursor_item,
                                    false,
                                    Some("Coal"),
                                    |id| id == COAL_ORE,
                                );
                            });

                            ui.add_space(14.0);
                            ui.vertical(|ui| {
                                let burn_frac =
                                    (burn_ticks as f32 / max_burn.max(1) as f32).clamp(0.0, 1.0);
                                ui.add(
                                    egui::ProgressBar::new(burn_frac)
                                        .desired_width(185.0)
                                        .fill(Color32::from_rgb(235, 105, 35))
                                        .text(format!("🔥 Fuel: {} / {} t", burn_ticks, max_burn)),
                                );
                                ui.add_space(10.0);
                                let smelt_frac =
                                    (progress as f32 / max_prog.max(1) as f32).clamp(0.0, 1.0);
                                ui.add(
                                    egui::ProgressBar::new(smelt_frac)
                                        .desired_width(185.0)
                                        .fill(Color32::from_rgb(235, 155, 45))
                                        .text(format!(
                                            "➡ Smelting: {} / {} t",
                                            progress, max_prog
                                        )),
                                );
                            });

                            ui.add_space(14.0);
                            ui.vertical(|ui| {
                                ui.add_space(18.0);
                                ui.label("Output");
                                draw_slot_widget(
                                    ui,
                                    &mut out,
                                    &mut inventory.cursor_item,
                                    false,
                                    Some("Ingot"),
                                    |_| false,
                                );
                            });
                        });

                        energy.set_simple_furnace_slots(pos, input, fuel, out);
                    }
                }

                ALLOY_SMELTER => {
                    if let Some((
                        mut in_a,
                        mut in_b,
                        mut out,
                        progress,
                        max_prog,
                        stored_eu,
                        max_eu,
                    )) = energy.alloy_ui_data(pos)
                    {
                        ui.horizontal(|ui| {
                            draw_vertical_eu_bar(ui, stored_eu, max_eu, 68.0);
                            ui.add_space(10.0);

                            ui.vertical(|ui| {
                                ui.label("Inputs (Cu+Sn or Fe+Coal)");
                                ui.horizontal(|ui| {
                                    draw_slot_widget(
                                        ui,
                                        &mut in_a,
                                        &mut inventory.cursor_item,
                                        false,
                                        Some("Input A"),
                                        |id| energy.recipes.is_alloy_ingredient(id),
                                    );
                                    ui.add_space(4.0);
                                    draw_slot_widget(
                                        ui,
                                        &mut in_b,
                                        &mut inventory.cursor_item,
                                        false,
                                        Some("Input B"),
                                        |id| energy.recipes.is_alloy_ingredient(id),
                                    );
                                });
                            });

                            ui.add_space(12.0);
                            ui.vertical(|ui| {
                                ui.label(format!("⚡ Charge: {} / {} EU", stored_eu, max_eu));
                                let frac =
                                    (progress as f32 / max_prog.max(1) as f32).clamp(0.0, 1.0);
                                ui.add(
                                    egui::ProgressBar::new(frac)
                                        .desired_width(155.0)
                                        .fill(Color32::from_rgb(225, 165, 50))
                                        .text(format!("➡ Alloying: {} / {} t", progress, max_prog)),
                                );
                            });

                            ui.add_space(12.0);
                            ui.vertical(|ui| {
                                ui.label("Output (Alloy)");
                                draw_slot_widget(
                                    ui,
                                    &mut out,
                                    &mut inventory.cursor_item,
                                    false,
                                    Some("Bronze/Steel"),
                                    |_| false,
                                );
                            });
                        });

                        energy.set_alloy_slots(pos, in_a, in_b, out);
                    }
                }

                BATBOX => {
                    if let Some((mut slot, stored_eu, max_eu)) = energy.batbox_ui_data(pos) {
                        ui.horizontal(|ui| {
                            draw_vertical_eu_bar(ui, stored_eu, max_eu, 68.0);
                            ui.add_space(12.0);

                            ui.vertical(|ui| {
                                ui.label("Battery Slot");
                                draw_slot_widget(
                                    ui,
                                    &mut slot,
                                    &mut inventory.cursor_item,
                                    false,
                                    Some("Battery"),
                                    |_| true,
                                );
                            });

                            ui.add_space(14.0);
                            ui.vertical(|ui| {
                                ui.label("Stored Electrical Energy (Max Out: 32 EU/t)");
                                let frac =
                                    (stored_eu as f32 / max_eu.max(1) as f32).clamp(0.0, 1.0);
                                ui.add(
                                    egui::ProgressBar::new(frac)
                                        .desired_width(240.0)
                                        .fill(Color32::from_rgb(45, 205, 150))
                                        .text(format!("⚡ {} / {} EU", stored_eu, max_eu)),
                                );
                            });
                        });

                        energy.set_batbox_slot(pos, slot);
                    }
                }

                CHEST => {
                    if let Some(slots) = energy.chest_slots_mut(pos) {
                        ui.label(
                            egui::RichText::new("Chest Contents (3x9)")
                                .size(13.0)
                                .color(Color32::from_gray(190)),
                        );
                        ui.add_space(2.0);
                        for row in 0..MAIN_INV_ROWS {
                            ui.horizontal(|ui| {
                                ui.spacing_mut().item_spacing = Vec2::splat(SLOT_GAP);
                                for col in 0..MAIN_INV_COLS {
                                    let idx = row * MAIN_INV_COLS + col;
                                    draw_slot_widget(
                                        ui,
                                        &mut slots[idx],
                                        &mut inventory.cursor_item,
                                        false,
                                        None,
                                        |_| true,
                                    );
                                }
                            });
                            ui.add_space(SLOT_GAP);
                        }
                    }
                }

                WORKBENCH => {
                    ui.label(
                        egui::RichText::new("Workbench Crafting (3x3)")
                            .size(13.0)
                            .color(Color32::from_gray(190)),
                    );
                    ui.add_space(2.0);
                    draw_crafting_widget(
                        ui,
                        &mut inventory.craft_3x3,
                        3,
                        3,
                        &mut inventory.cursor_item,
                        &energy.recipes,
                    );
                }

                _ => {}
            }

            ui.separator();
            draw_player_inventory_grid(ui, inventory);
            ui.add_space(4.0);
            ui.label(
                egui::RichText::new("LMB: take/place stack • RMB: split/place 1 • E/Esc: close")
                    .size(11.0)
                    .color(Color32::from_gray(140)),
            );
        });
}

// ─── NEI / JEI Item & Recipe Browser Sidebar ─────────────────────────────────

fn draw_nei_sidebar(
    ctx: &Context,
    recipes: &RecipeRegistry,
    inventory: &mut PlayerInventory,
    nei_query: &mut Option<(BlockId, NeiMode)>,
) {
    egui::Window::new("📦 NEI Item & Recipe Browser")
        .anchor(Align2::RIGHT_CENTER, Vec2::new(-14.0, 0.0))
        .collapsible(false)
        .resizable(false)
        .default_width(250.0)
        .show(ctx, |ui| {
            ui.label(
                egui::RichText::new("LMB: Recipes • RMB: Usages • MMB: +16")
                    .size(11.0)
                    .color(Color32::from_gray(155)),
            );
            ui.add_space(4.0);

            // Grid of all items (6 columns x 5 rows = 30 items)
            const COLS: usize = 6;
            for chunk in ALL_ITEMS.chunks(COLS) {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing = Vec2::splat(4.0);
                    for &item_id in chunk {
                        let (rect, response) =
                            ui.allocate_exact_size(Vec2::splat(38.0), Sense::click());
                        let is_active = nei_query.map(|(q, _)| q == item_id).unwrap_or(false);
                        let bg = if response.hovered() {
                            Color32::from_rgb(70, 74, 88)
                        } else {
                            Color32::from_rgb(34, 36, 44)
                        };
                        let border = if is_active {
                            Stroke::new(2.0_f32, Color32::from_rgb(245, 205, 75))
                        } else {
                            Stroke::new(1.0_f32, Color32::from_rgb(78, 82, 94))
                        };
                        ui.painter().rect(rect, Rounding::same(4.0), bg, border);
                        draw_item_icon(ui.ctx(), ui.painter(), rect.shrink(5.0), item_id);

                        if response.clicked_by(PointerButton::Primary) {
                            *nei_query = Some((item_id, NeiMode::Recipes));
                        } else if response.clicked_by(PointerButton::Secondary) {
                            *nei_query = Some((item_id, NeiMode::Usages));
                        } else if response.clicked_by(PointerButton::Middle) {
                            let _ = inventory.add(item_id, 16);
                        }

                        response.on_hover_text(format!(
                            "{}\nLMB: View Recipes\nRMB: View Usages\nMMB: Give x16",
                            block_name(item_id)
                        ));
                    }
                });
                ui.add_space(4.0);
            }

            // Recipe / Usage viewer section below the item palette
            if let Some((target_id, mode)) = *nei_query {
                ui.separator();
                ui.horizontal(|ui| {
                    let mode_label = match mode {
                        NeiMode::Recipes => "Recipes for:",
                        NeiMode::Usages => "Usages of:",
                    };
                    ui.label(
                        egui::RichText::new(format!("{} {}", mode_label, block_name(target_id)))
                            .strong()
                            .color(Color32::from_rgb(245, 215, 110)),
                    );
                });

                ui.horizontal(|ui| {
                    if ui
                        .selectable_label(mode == NeiMode::Recipes, "📖 Recipes")
                        .clicked()
                    {
                        *nei_query = Some((target_id, NeiMode::Recipes));
                    }
                    if ui
                        .selectable_label(mode == NeiMode::Usages, "🔧 Usages")
                        .clicked()
                    {
                        *nei_query = Some((target_id, NeiMode::Usages));
                    }
                    if ui.small_button("✖").clicked() {
                        *nei_query = None;
                    }
                });
                ui.add_space(4.0);

                let matched_recipes = match mode {
                    NeiMode::Recipes => recipes.recipes_for_output(target_id),
                    NeiMode::Usages => recipes.recipes_using_input(target_id),
                };

                if matched_recipes.is_empty() {
                    ui.label(
                        egui::RichText::new("No registered recipes found.")
                            .italics()
                            .color(Color32::from_gray(140)),
                    );
                } else {
                    egui::ScrollArea::vertical()
                        .max_height(260.0)
                        .show(ui, |ui| {
                            for (idx, recipe) in matched_recipes.iter().enumerate() {
                                if idx > 0 {
                                    ui.separator();
                                }
                                draw_recipe_card(ui, recipe, nei_query);
                            }
                        });
                }
            }
        });
}

fn draw_recipe_card(
    ui: &mut egui::Ui,
    recipe: &Recipe,
    nei_query: &mut Option<(BlockId, NeiMode)>,
) {
    match recipe {
        Recipe::CraftingShaped {
            width,
            height,
            inputs,
            output,
        } => {
            let station = if *width <= 2 && *height <= 2 {
                "2x2 Inventory / Workbench"
            } else {
                "3x3 Workbench"
            };
            ui.label(
                egui::RichText::new(format!("🛠 Shaped ({station})"))
                    .size(11.5)
                    .color(Color32::from_rgb(170, 205, 245)),
            );
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    for r in 0..*height {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing = Vec2::splat(3.0);
                            for c in 0..*width {
                                let cell = inputs[r * *width + c]
                                    .map(|id| ItemStack { id, count: 1 });
                                draw_readonly_item_badge(ui, cell, 32.0, nei_query);
                            }
                        });
                        if r + 1 < *height {
                            ui.add_space(3.0);
                        }
                    }
                });
                ui.add_space(8.0);
                ui.label(egui::RichText::new("➡").size(18.0));
                ui.add_space(8.0);
                draw_readonly_item_badge(ui, Some(*output), 36.0, nei_query);
            });
        }

        Recipe::CraftingShapeless { inputs, output } => {
            ui.label(
                egui::RichText::new("🛠 Shapeless Crafting")
                    .size(11.5)
                    .color(Color32::from_rgb(170, 205, 245)),
            );
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing = Vec2::splat(3.0);
                for &id in inputs {
                    draw_readonly_item_badge(
                        ui,
                        Some(ItemStack { id, count: 1 }),
                        32.0,
                        nei_query,
                    );
                }
                ui.add_space(6.0);
                ui.label(egui::RichText::new("➡").size(18.0));
                ui.add_space(6.0);
                draw_readonly_item_badge(ui, Some(*output), 36.0, nei_query);
            });
        }

        Recipe::Smelting {
            input_id,
            output,
            cook_time_ticks,
        } => {
            ui.label(
                egui::RichText::new(format!(
                    "🔥 Smelting — Stone / Electric Furnace ({cook_time_ticks}t)"
                ))
                .size(11.5)
                .color(Color32::from_rgb(245, 175, 110)),
            );
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                draw_readonly_item_badge(
                    ui,
                    Some(ItemStack {
                        id: *input_id,
                        count: 1,
                    }),
                    34.0,
                    nei_query,
                );
                ui.add_space(8.0);
                ui.label(egui::RichText::new("🔥 ➡").size(16.0));
                ui.add_space(8.0);
                draw_readonly_item_badge(ui, Some(*output), 36.0, nei_query);
            });
        }

        Recipe::Alloy {
            input_a,
            input_b,
            count_a,
            count_b,
            output,
            cook_time_ticks,
        } => {
            ui.label(
                egui::RichText::new(format!("⚡ Alloy Smelter ({cook_time_ticks}t)"))
                    .size(11.5)
                    .color(Color32::from_rgb(235, 195, 95)),
            );
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                draw_readonly_item_badge(
                    ui,
                    Some(ItemStack {
                        id: *input_a,
                        count: *count_a,
                    }),
                    34.0,
                    nei_query,
                );
                ui.label("+");
                draw_readonly_item_badge(
                    ui,
                    Some(ItemStack {
                        id: *input_b,
                        count: *count_b,
                    }),
                    34.0,
                    nei_query,
                );
                ui.add_space(6.0);
                ui.label(egui::RichText::new("➡").size(18.0));
                ui.add_space(6.0);
                draw_readonly_item_badge(ui, Some(*output), 36.0, nei_query);
            });
        }
    }
}

fn draw_cursor_stack(ctx: &Context, pos: Pos2, stack: ItemStack) {
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Tooltip,
        Id::new("cursor_item_layer"),
    ));
    let rect = Rect::from_center_size(pos + Vec2::new(14.0, 14.0), Vec2::splat(30.0));
    draw_item_icon(ctx, &painter, rect, stack.id);
    painter.text(
        rect.right_bottom() - Vec2::new(2.0, 1.0),
        Align2::RIGHT_BOTTOM,
        format!("{}", stack.count),
        FontId::monospace(12.0),
        Color32::WHITE,
    );
}

#[allow(dead_code)]
fn _unused_key_check(_k: Key) {}
