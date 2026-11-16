// 2D Texture Array (`wgpu::TextureViewDimension::D2Array`) loader with 16x16 normalization,
// animated spritesheet cropping, CPU grayscale tinting (`grass_top`, `leaves_oak`),
// and composite overlay blending (`grass_side` + `grass_side_overlay`).

use std::path::Path;

use image::{imageops::FilterType, DynamicImage, GenericImageView, Rgba, RgbaImage};

use crate::voxel::blocks::{TextureLayer, TOTAL_LAYERS};

pub const TEX_SIZE: u32 = 16;
const FOLIAGE_TINT: [f32; 3] = [0.38, 0.73, 0.22];

pub struct BlockTextureArray {
    #[allow(dead_code)]
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}

impl BlockTextureArray {
    /// Loads all block PNGs from `assets/textures/blocks/`, normalizes them to `16x16` RGBA,
    /// applies CPU grayscale tinting and overlay compositing, and uploads them into a
    /// `16x16 x TOTAL_LAYERS` `wgpu::TextureViewDimension::D2Array`.
    pub fn load(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let base_dir = Path::new("assets/textures/blocks");

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("block_texture_array"),
            size: wgpu::Extent3d {
                width: TEX_SIZE,
                height: TEX_SIZE,
                depth_or_array_layers: TOTAL_LAYERS,
            },
            mip_level_count: 1,
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

            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                img.as_raw(),
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * TEX_SIZE),
                    rows_per_image: Some(TEX_SIZE),
                },
                wgpu::Extent3d {
                    width: TEX_SIZE,
                    height: TEX_SIZE,
                    depth_or_array_layers: 1,
                },
            );
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
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        log::info!(
            "Loaded {} block textures ({}x{} D2Array) from assets/textures/blocks",
            TOTAL_LAYERS,
            TEX_SIZE,
            TEX_SIZE
        );

        Self {
            texture,
            view,
            sampler,
        }
    }
}

/// Loads a PNG, crops vertical animation spritesheets (`height > width`) to frame 0 (`16x16`),
/// and guarantees a `16x16` `RgbaImage` output.
pub fn load_normalized_16x16(path: &Path) -> RgbaImage {
    let img = match image::open(path) {
        Ok(i) => i,
        Err(err) => {
            log::warn!("Failed to load texture {}: {} (using fallback)", path.display(), err);
            return fallback_checkerboard();
        }
    };
    normalize_to_16x16(img)
}

pub fn normalize_to_16x16(img: DynamicImage) -> RgbaImage {
    let (w, h) = img.dimensions();
    let cropped = if h > w && w >= TEX_SIZE && h >= TEX_SIZE {
        // Vertical animation spritesheet (e.g., lava_still.png, fire_layer_0.png): take frame 0.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spritesheet_crops_to_16x16_and_tinting_scales_channels() {
        let tall = DynamicImage::ImageRgba8(RgbaImage::from_pixel(16, 320, Rgba([200, 200, 200, 255])));
        let mut norm = normalize_to_16x16(tall);
        assert_eq!(norm.dimensions(), (16, 16));

        tint_rgba_in_place(&mut norm, FOLIAGE_TINT);
        let p = norm.get_pixel(0, 0).0;
        assert_eq!(p, [76, 146, 44, 255]);
    }
}
