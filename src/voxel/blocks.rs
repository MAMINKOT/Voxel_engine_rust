// Texture layer registry for the 2D Texture Array (`wgpu::TextureViewDimension::D2Array`)
// and per-face block-to-texture mapping (`get_face_texture`).

use crate::voxel::chunk::{
    base_block_id, block_color, is_active_block, BlockId, ALLOY_SMELTER, BATBOX, BEDROCK, CABLE,
    CHEST, COAL_GENERATOR, COAL_ORE, COBBLESTONE, COPPER_ORE, DIRT, ELECTRIC_FURNACE, FIRE,
    GRASS, IRON_ORE, LAVA, OAK_LEAVES, OBSIDIAN, OIL, PURPLE_STAINED_GLASS, SIMPLE_FURNACE,
    STONE, TIN_ORE, TNT, TORCH, WATER, WOOD, WORKBENCH,
};

/// Sentinel value in `Vertex::tex_layer` indicating an untextured vertex-colored surface
/// (e.g. `Water`, `Oil`, `Cable`).
pub const NO_TEXTURE: u32 = u32::MAX;

/// Total number of 16x16 layers in the `texture_2d_array<f32>`.
pub const TOTAL_LAYERS: u32 = 32;

/// Sequence of 16x16 layers in the GPU 2D Texture Array (`0..32`).
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureLayer {
    Stone = 0,
    Dirt = 1,
    GrassTop = 2,
    GrassSide = 3,
    Cobblestone = 4,
    Brick = 5,
    Sand = 6,
    Gravel = 7,
    IronOre = 8,
    CoalOre = 9,
    IronBlock = 10,
    CoalBlock = 11,
    Obsidian = 12,
    Clay = 13,
    LogOakSide = 14,
    LogOakTop = 15,
    LeavesOak = 16,
    Glass = 17,
    CraftingTableTop = 18,
    CraftingTableFront = 19,
    CraftingTableSide = 20,
    FurnaceFrontOff = 21,
    FurnaceFrontOn = 22,
    FurnaceSide = 23,
    FurnaceTop = 24,
    TntTop = 25,
    TntBottom = 26,
    TntSide = 27,
    LavaStill = 28,
    TorchOn = 29,
    FireLayer0 = 30,
    GlassMagenta = 31,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockFace {
    PosX,
    NegX,
    PosY,
    NegY,
    PosZ,
    NegZ,
}

impl BlockFace {
    #[inline]
    pub fn from_axis(d: usize, positive: bool) -> Self {
        match (d, positive) {
            (0, true) => Self::PosX,
            (0, false) => Self::NegX,
            (1, true) => Self::PosY,
            (1, false) => Self::NegY,
            (2, true) => Self::PosZ,
            _ => Self::NegZ,
        }
    }
}

/// Maps `(block_id, face, is_active)` to a `TextureLayer` index in the GPU `texture_2d_array`,
/// or `NO_TEXTURE` (`u32::MAX`) for procedural/vertex-colored materials (`Water`, `Oil`, `Cable`).
pub fn get_face_texture(block_id: BlockId, face: BlockFace, is_active: bool) -> u32 {
    let base = base_block_id(block_id);
    let active = is_active || is_active_block(block_id);

    match base {
        GRASS => match face {
            BlockFace::PosY => TextureLayer::GrassTop as u32,
            BlockFace::NegY => TextureLayer::Dirt as u32,
            _ => TextureLayer::GrassSide as u32,
        },
        WORKBENCH => match face {
            BlockFace::PosY => TextureLayer::CraftingTableTop as u32,
            BlockFace::PosZ => TextureLayer::CraftingTableFront as u32,
            BlockFace::NegY => TextureLayer::LogOakSide as u32,
            _ => TextureLayer::CraftingTableSide as u32,
        },
        SIMPLE_FURNACE | COAL_GENERATOR | ELECTRIC_FURNACE | ALLOY_SMELTER => match face {
            BlockFace::PosZ => {
                if active {
                    TextureLayer::FurnaceFrontOn as u32
                } else {
                    TextureLayer::FurnaceFrontOff as u32
                }
            }
            BlockFace::PosY => TextureLayer::FurnaceTop as u32,
            _ => TextureLayer::FurnaceSide as u32,
        },
        TNT => match face {
            BlockFace::PosY => TextureLayer::TntTop as u32,
            BlockFace::NegY => TextureLayer::TntBottom as u32,
            _ => TextureLayer::TntSide as u32,
        },
        WOOD => match face {
            BlockFace::PosY | BlockFace::NegY => TextureLayer::LogOakTop as u32,
            _ => TextureLayer::LogOakSide as u32,
        },
        STONE => TextureLayer::Stone as u32,
        DIRT => TextureLayer::Dirt as u32,
        COBBLESTONE => TextureLayer::Cobblestone as u32,
        IRON_ORE | COPPER_ORE | TIN_ORE => TextureLayer::IronOre as u32,
        COAL_ORE => TextureLayer::CoalOre as u32,
        BATBOX => TextureLayer::IronBlock as u32,
        OBSIDIAN | BEDROCK => TextureLayer::Obsidian as u32,
        OAK_LEAVES => TextureLayer::LeavesOak as u32,
        PURPLE_STAINED_GLASS => TextureLayer::GlassMagenta as u32,
        LAVA => TextureLayer::LavaStill as u32,
        TORCH => TextureLayer::TorchOn as u32,
        FIRE => TextureLayer::FireLayer0 as u32,
        CHEST => TextureLayer::LogOakSide as u32,
        CABLE | WATER | OIL => NO_TEXTURE,
        _ => NO_TEXTURE,
    }
}

/// Returns the vertex tint multiplier `[R, G, B]` for a block face.
/// Blocks with dedicated full-color PNG textures return `[1.0, 1.0, 1.0]` so the texture
/// displays with true colors; shared/untextured blocks use subtle tinting to remain distinct.
pub fn texture_tint_color(raw_id: BlockId, axis: usize) -> [f32; 3] {
    let base = base_block_id(raw_id);
    match base {
        STONE | DIRT | GRASS | COBBLESTONE | COAL_ORE | IRON_ORE | OBSIDIAN | OAK_LEAVES
        | WOOD | WORKBENCH | SIMPLE_FURNACE | TNT | LAVA | TORCH | FIRE
        | PURPLE_STAINED_GLASS => [1.0, 1.0, 1.0],
        COPPER_ORE => [1.0, 0.72, 0.48],
        TIN_ORE => [0.85, 0.92, 1.0],
        BEDROCK => [0.42, 0.42, 0.46],
        CHEST => [0.92, 0.72, 0.46],
        COAL_GENERATOR => [0.82, 0.78, 0.74],
        ELECTRIC_FURNACE => [0.82, 0.90, 1.05],
        ALLOY_SMELTER => [1.05, 0.86, 0.68],
        BATBOX => [0.68, 0.92, 0.98],
        _ => {
            let _ = axis;
            block_color(base)
        }
    }
}
