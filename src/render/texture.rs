// 2D Texture Array (`wgpu::TextureViewDimension::D2Array`) loader with:
//   • Full 5-level Mipmap chain (`16x16 -> 8x8 -> 4x4 -> 2x2 -> 1x1`) + Nearest/Linear filtering
//   • Multi-frame animated spritesheet loading (`lava_still.png` 20 frames, `fire_layer_0.png` 32 frames)
//   • CPU grayscale tinting (`grass_top`, `leaves_oak`) & composite overlay (`grass_side`)
//   • 16x16 pixel-art `egui::ColorImage` icon generation for UI slots & NEI browser

use std::collections::HashMap;
use std::path::Path;

use image::{imageops::FilterType, DynamicImage, GenericImageView, Rgba, RgbaImage};

use crate::voxel::blocks::{
    get_face_texture, texture_tint_color, BlockFace, TextureLayer, FIRE_ANIM_START, FIRE_FRAMES,
    LAVA_ANIM_START, LAVA_FRAMES, NO_TEXTURE, TOTAL_LAYERS,
};
use crate::voxel::chunk::{
    block_color, BlockId, ALL_ITEMS, BRONZE_INGOT, CABLE, COPPER_INGOT, IRON_INGOT, LIGHTER, OIL,
    STEEL_INGOT, TIN_INGOT, WATER,
};

pub const TEX_SIZE: u32 = 16;
pub const MIP_LEVELS: u32 = 5; // 16 -> 8 -> 4 -> 2 -> 1
const FOLIAGE_TINT: [f32; 3] = [0.38, 0.73, 0.22];

pub struct BlockTextureArray {
    #[allow(dead_code)]
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

impl BlockTextureArray {
    /// Loads all block PNGs from `assets/textures/blocks/`, generates 5 mipmap levels per layer,
    /// loads all animation frames for Lava and Fire, and uploads into `16x16 x TOTAL_LAYERS` `D2Array`.
    pub fn load(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let base_dir = Path::new("assets/textures/blocks");

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("block_texture_array"),
            size: wgpu::Extent3d {
                width: TEX_SIZE,
                height: TEX_SIZE,
                depth_or_array_layers: TOTAL_LAYERS,
            },
            mip_level_count: MIP_LEVELS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let layer_specs: [(TextureLayer, &str); 32] = [
            (TextureLayer::Stone, "stone.png"),
            (TextureLayer::Dirt, "dirt.png"),
            (TextureLayer::GrassTop, "grass_top.png"),
            (TextureLayer::GrassSide, "grass_side.png"),
            (TextureLayer::Cobblestone, "cobblestone.png"),
            (TextureLayer::Brick, "brick.png"),
            (TextureLayer::Sand, "sand.png"),
            (TextureLayer::Gravel, "gravel.png"),
            (TextureLayer::IronOre, "iron_ore.png"),
            (TextureLayer::CoalOre, "coal_ore.png"),
            (TextureLayer::IronBlock, "iron_block.png"),
            (TextureLayer::CoalBlock, "coal_block.png"),
            (TextureLayer::Obsidian, "obsidian.png"),
            (TextureLayer::Clay, "clay.png"),
            (TextureLayer::LogOakSide, "log_oak.png"),
            (TextureLayer::LogOakTop, "log_oak_top.png"),
            (TextureLayer::LeavesOak, "leaves_oak.png"),
            (TextureLayer::Glass, "glass.png"),
            (TextureLayer::CraftingTableTop, "crafting_table_top.png"),
            (TextureLayer::CraftingTableFront, "crafting_table_front.png"),
            (TextureLayer::CraftingTableSide, "crafting_table_side.png"),
            (TextureLayer::FurnaceFrontOff, "furnace_front_off.png"),
            (TextureLayer::FurnaceFrontOn, "furnace_front_on.png"),
            (TextureLayer::FurnaceSide, "furnace_side.png"),
            (TextureLayer::FurnaceTop, "furnace_top.png"),
            (TextureLayer::TntTop, "tnt_top.png"),
            (TextureLayer::TntBottom, "tnt_bottom.png"),
            (TextureLayer::TntSide, "tnt_side.png"),
            (TextureLayer::LavaStill, "lava_still.png"),
            (TextureLayer::TorchOn, "torch_on.png"),
            (TextureLayer::FireLayer0, "fire_layer_0.png"),
            (TextureLayer::GlassMagenta, "glass_magenta.png"),
        ];

        for (layer, filename) in layer_specs {
            let mut img = load_normalized_16x16(&base_dir.join(filename));
            match layer {
                TextureLayer::GrassTop | TextureLayer::LeavesOak => {
                    tint_rgba_in_place(&mut img, FOLIAGE_TINT);
                }
                TextureLayer::GrassSide => {
                    let mut overlay = load_normalized_16x16(&base_dir.join("grass_side_overlay.png"));
                    tint_rgba_in_place(&mut overlay, FOLIAGE_TINT);
                    alpha_blend_in_place(&mut img, &overlay);
                }
                _ => {}
            }
            upload_layer_with_mipmaps(queue, &texture, layer as u32, &img);
        }

        // Load all 20 animation frames of `lava_still.png` into `32..52`
        let lava_frames = load_spritesheet_frames(&base_dir.join("lava_still.png"), LAVA_FRAMES);
        for (i, frame) in lava_frames.iter().enumerate() {
            upload_layer_with_mipmaps(queue, &texture, LAVA_ANIM_START + i as u32, frame);
        }

        // Load all 32 animation frames of `fire_layer_0.png` into `52..84`
        let fire_frames = load_spritesheet_frames(&base_dir.join("fire_layer_0.png"), FIRE_FRAMES);
        for (i, frame) in fire_frames.iter().enumerate() {
            upload_layer_with_mipmaps(queue, &texture, FIRE_ANIM_START + i as u32, frame);
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("block_texture_array_view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            base_array_layer: 0,
            array_layer_count: Some(TOTAL_LAYERS),
            ..Default::default()
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("block_texture_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        log::info!(
            "Loaded {} block texture layers ({}x{} D2Array, {} mip levels)",
            TOTAL_LAYERS,
            TEX_SIZE,
            TEX_SIZE,
            MIP_LEVELS
        );

        Self {
            texture,
            view,
            sampler,
        }
    }
}

fn upload_layer_with_mipmaps(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    layer_idx: u32,
    base_img: &RgbaImage,
) {
    let mut current = base_img.clone();
    for mip in 0..MIP_LEVELS {
        let size = (TEX_SIZE >> mip).max(1);
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture,
                mip_level: mip,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer_idx,
                },
                aspect: wgpu::TextureAspect::All,
            },
            current.as_raw(),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * size),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );
        if mip + 1 < MIP_LEVELS {
            current = downsample_2x2_rgba(&current);
        }
    }
}

/// Bilinear 2x2 box-filter downsampling for mipmap generation (`16 -> 8 -> 4 -> 2 -> 1`).
/// Weights RGB by alpha when transparent pixels are present so cutouts don't bleed black fringes.
pub fn downsample_2x2_rgba(src: &RgbaImage) -> RgbaImage {
    let dst_w = (src.width() / 2).max(1);
    let dst_h = (src.height() / 2).max(1);
    RgbaImage::from_fn(dst_w, dst_h, |x, y| {
        let sx = x * 2;
        let sy = y * 2;
        let samples = [
            src.get_pixel(sx, sy).0,
            src.get_pixel((sx + 1).min(src.width() - 1), sy).0,
            src.get_pixel(sx, (sy + 1).min(src.height() - 1)).0,
            src.get_pixel(
                (sx + 1).min(src.width() - 1),
                (sy + 1).min(src.height() - 1),
            )
            .0,
        ];
        let mut sum_a = 0.0f32;
        let mut sum_rgb = [0.0f32; 3];
        let mut raw_rgb = [0.0f32; 3];
        for p in samples {
            let a = p[3] as f32;
            sum_a += a;
            for c in 0..3 {
                sum_rgb[c] += (p[c] as f32) * a;
                raw_rgb[c] += p[c] as f32;
            }
        }
        let avg_a = (sum_a * 0.25).clamp(0.0, 255.0).round() as u8;
        if sum_a > 1.0 {
            Rgba([
                (sum_rgb[0] / sum_a).clamp(0.0, 255.0).round() as u8,
                (sum_rgb[1] / sum_a).clamp(0.0, 255.0).round() as u8,
                (sum_rgb[2] / sum_a).clamp(0.0, 255.0).round() as u8,
                avg_a,
            ])
        } else {
            Rgba([
                (raw_rgb[0] * 0.25).clamp(0.0, 255.0).round() as u8,
                (raw_rgb[1] * 0.25).clamp(0.0, 255.0).round() as u8,
                (raw_rgb[2] * 0.25).clamp(0.0, 255.0).round() as u8,
                avg_a,
            ])
        }
    })
}

/// Loads all square (`width x width`) frames from a vertical animation spritesheet
/// (e.g. `lava_still.png` 64x1280 = 20 frames, `fire_layer_0.png` 32x1024 = 32 frames)
/// and normalizes each frame to `16x16` RGBA.
fn load_spritesheet_frames(path: &Path, expected_frames: u32) -> Vec<RgbaImage> {
    let Ok(img) = image::open(path) else {
        log::warn!(
            "Failed to load animated spritesheet {} (using fallback)",
            path.display()
        );
        return vec![fallback_checkerboard(); expected_frames as usize];
    };
    let (w, h) = img.dimensions();
    let frame_size = w.max(1);
    let total_frames = (h / frame_size).max(1);
    let mut frames = Vec::with_capacity(expected_frames as usize);
    for i in 0..expected_frames {
        let src_idx = i % total_frames;
        let y_off = src_idx * frame_size;
        let cropped = if y_off + frame_size <= h {
            let sub = img.crop_imm(0, y_off, frame_size, frame_size);
            if frame_size == TEX_SIZE {
                sub.to_rgba8()
            } else {
                sub.resize_exact(TEX_SIZE, TEX_SIZE, FilterType::Nearest)
                    .to_rgba8()
            }
        } else {
            normalize_to_16x16(img.clone())
        };
        frames.push(cropped);
    }
    frames
}

/// Loads a PNG, crops vertical animation spritesheets (`height > width`) to frame 0 (`16x16`),
/// and guarantees a `16x16` `RgbaImage` output.
pub fn load_normalized_16x16(path: &Path) -> RgbaImage {
    let img = match image::open(path) {
        Ok(i) => i,
        Err(err) => {
            log::warn!(
                "Failed to load texture {}: {} (using fallback)",
                path.display(),
                err
            );
            return fallback_checkerboard();
        }
    };
    normalize_to_16x16(img)
}

pub fn normalize_to_16x16(img: DynamicImage) -> RgbaImage {
    let (w, h) = img.dimensions();
    let cropped = if h > w && w >= TEX_SIZE && h >= TEX_SIZE {
        img.crop_imm(0, 0, TEX_SIZE, TEX_SIZE)
    } else if h > w && w > 0 {
        img.crop_imm(0, 0, w, w)
    } else {
        img
    };

    if cropped.width() == TEX_SIZE && cropped.height() == TEX_SIZE {
        cropped.to_rgba8()
    } else {
        cropped
            .resize_exact(TEX_SIZE, TEX_SIZE, FilterType::Nearest)
            .to_rgba8()
    }
}

pub fn tint_rgba_in_place(img: &mut RgbaImage, tint: [f32; 3]) {
    for px in img.pixels_mut() {
        px.0[0] = ((px.0[0] as f32) * tint[0]).clamp(0.0, 255.0).round() as u8;
        px.0[1] = ((px.0[1] as f32) * tint[1]).clamp(0.0, 255.0).round() as u8;
        px.0[2] = ((px.0[2] as f32) * tint[2]).clamp(0.0, 255.0).round() as u8;
    }
}

pub fn alpha_blend_in_place(base: &mut RgbaImage, overlay: &RgbaImage) {
    for (b, o) in base.pixels_mut().zip(overlay.pixels()) {
        let a = (o.0[3] as f32) / 255.0;
        if a <= 0.0 {
            continue;
        }
        let inv_a = 1.0 - a;
        b.0[0] = ((o.0[0] as f32) * a + (b.0[0] as f32) * inv_a)
            .clamp(0.0, 255.0)
            .round() as u8;
        b.0[1] = ((o.0[1] as f32) * a + (b.0[1] as f32) * inv_a)
            .clamp(0.0, 255.0)
            .round() as u8;
        b.0[2] = ((o.0[2] as f32) * a + (b.0[2] as f32) * inv_a)
            .clamp(0.0, 255.0)
            .round() as u8;
        b.0[3] = 255;
    }
}

fn fallback_checkerboard() -> RgbaImage {
    RgbaImage::from_fn(TEX_SIZE, TEX_SIZE, |x, y| {
        if ((x / 4) + (y / 4)) % 2 == 0 {
            Rgba([255, 0, 255, 255])
        } else {
            Rgba([20, 20, 20, 255])
        }
    })
}

/// Builds 16x16 `egui::ColorImage` icons for every item in `ALL_ITEMS` so hotbar,
/// inventory, chests, furnaces, and NEI display crisp pixel-art textures.
pub fn build_ui_item_icons() -> HashMap<BlockId, egui::ColorImage> {
    let base_dir = Path::new("assets/textures/blocks");
    let mut icons = HashMap::new();

    let layer_file = |layer: u32| -> Option<&'static str> {
        match layer {
            0 => Some("stone.png"),
            1 => Some("dirt.png"),
            2 => Some("grass_top.png"),
            3 => Some("grass_side.png"),
            4 => Some("cobblestone.png"),
            8 => Some("iron_ore.png"),
            9 => Some("coal_ore.png"),
            10 => Some("iron_block.png"),
            12 => Some("obsidian.png"),
            14 => Some("log_oak.png"),
            15 => Some("log_oak_top.png"),
            16 => Some("leaves_oak.png"),
            19 => Some("crafting_table_front.png"),
            21 => Some("furnace_front_off.png"),
            27 => Some("tnt_side.png"),
            28 => Some("lava_still.png"),
            29 => Some("torch_on.png"),
            30 => Some("fire_layer_0.png"),
            31 => Some("glass_magenta.png"),
            _ => None,
        }
    };

    for &id in &ALL_ITEMS {
        let layer = get_face_texture(id, BlockFace::PosZ, false);
        let rgba: RgbaImage = if layer != NO_TEXTURE {
            if let Some(filename) = layer_file(layer) {
                let mut img = load_normalized_16x16(&base_dir.join(filename));
                if layer == TextureLayer::LeavesOak as u32 || layer == TextureLayer::GrassTop as u32 {
                    tint_rgba_in_place(&mut img, FOLIAGE_TINT);
                } else if layer == TextureLayer::GrassSide as u32 {
                    let mut overlay =
                        load_normalized_16x16(&base_dir.join("grass_side_overlay.png"));
                    tint_rgba_in_place(&mut overlay, FOLIAGE_TINT);
                    alpha_blend_in_place(&mut img, &overlay);
                }
                let tint = texture_tint_color(id, 1);
                if tint != [1.0, 1.0, 1.0] {
                    tint_rgba_in_place(&mut img, tint);
                }
                img
            } else {
                procedural_item_icon(id)
            }
        } else {
            procedural_item_icon(id)
        };

        let color_img = egui::ColorImage::from_rgba_unmultiplied(
            [TEX_SIZE as usize, TEX_SIZE as usize],
            rgba.as_raw(),
        );
        icons.insert(id, color_img);
    }

    icons
}

fn procedural_item_icon(id: BlockId) -> RgbaImage {
    let [r, g, b] = block_color(id);
    let to_u8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    let base = [to_u8(r), to_u8(g), to_u8(b)];
    let hi = [
        to_u8(r * 1.25),
        to_u8(g * 1.25),
        to_u8(b * 1.25),
    ];
    let lo = [
        to_u8(r * 0.65),
        to_u8(g * 0.65),
        to_u8(b * 0.65),
    ];

    RgbaImage::from_fn(TEX_SIZE, TEX_SIZE, |x, y| {
        match id {
            COPPER_INGOT | TIN_INGOT | IRON_INGOT | BRONZE_INGOT | STEEL_INGOT => {
                // Classic angled metal ingot bar silhouette
                let dx = x as i32 - 8;
                let dy = y as i32 - 8;
                let u = dx + dy;
                let v = dx - dy;
                if u.abs() <= 4 && v.abs() <= 7 {
                    if u < -1 {
                        Rgba([hi[0], hi[1], hi[2], 255])
                    } else if u > 2 {
                        Rgba([lo[0], lo[1], lo[2], 255])
                    } else {
                        Rgba([base[0], base[1], base[2], 255])
                    }
                } else {
                    Rgba([0, 0, 0, 0])
                }
            }
            CABLE => {
                // Cross-shaped insulated copper cable
                let in_h = (6..=9).contains(&y) && (1..=14).contains(&x);
                let in_v = (6..=9).contains(&x) && (1..=14).contains(&y);
                if in_h || in_v {
                    if x == 7 || y == 7 {
                        Rgba([hi[0], hi[1], hi[2], 255])
                    } else {
                        Rgba([base[0], base[1], base[2], 255])
                    }
                } else {
                    Rgba([0, 0, 0, 0])
                }
            }
            LIGHTER => {
                // Flint & Steel sparker icon
                if (4..=11).contains(&x) && (5..=12).contains(&y) {
                    if x <= 6 && y <= 8 {
                        Rgba([255, 190, 60, 255])
                    } else {
                        Rgba([base[0], base[1], base[2], 255])
                    }
                } else {
                    Rgba([0, 0, 0, 0])
                }
            }
            WATER | OIL => {
                // Fluid tile with top wave highlight
                if y < 3 {
                    Rgba([hi[0], hi[1], hi[2], 220])
                } else {
                    Rgba([base[0], base[1], base[2], 235])
                }
            }
            _ => {
                if x == 0 || y == 0 || x == 15 || y == 15 {
                    Rgba([lo[0], lo[1], lo[2], 255])
                } else {
                    Rgba([base[0], base[1], base[2], 255])
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spritesheet_crops_to_16x16_and_tinting_scales_channels() {
        let tall =
            DynamicImage::ImageRgba8(RgbaImage::from_pixel(16, 320, Rgba([200, 200, 200, 255])));
        let mut norm = normalize_to_16x16(tall);
        assert_eq!(norm.dimensions(), (16, 16));

        tint_rgba_in_place(&mut norm, FOLIAGE_TINT);
        let p = norm.get_pixel(0, 0).0;
        assert_eq!(p, [76, 146, 44, 255]);
    }

    #[test]
    fn mipmap_chain_downsamples_16x16_to_1x1() {
        let base = RgbaImage::from_pixel(16, 16, Rgba([100, 150, 200, 255]));
        let m1 = downsample_2x2_rgba(&base);
        let m2 = downsample_2x2_rgba(&m1);
        let m3 = downsample_2x2_rgba(&m2);
        let m4 = downsample_2x2_rgba(&m3);
        assert_eq!(m1.dimensions(), (8, 8));
        assert_eq!(m2.dimensions(), (4, 4));
        assert_eq!(m3.dimensions(), (2, 2));
        assert_eq!(m4.dimensions(), (1, 1));
        assert_eq!(m4.get_pixel(0, 0).0, [100, 150, 200, 255]);
    }
}
