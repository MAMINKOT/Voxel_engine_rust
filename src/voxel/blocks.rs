// Declarative Block Definition Registry (`BlockDef`), 2D Texture Array layer registry,
// and per-face block-to-texture mapping (`get_face_texture`).

use crate::voxel::chunk::{
    base_block_id, block_color, facing_from_id, is_active_block, BlockId, ALLOY_SMELTER, BATBOX,
    BEDROCK, CABLE, CHEST, COAL_GENERATOR, COAL_ORE, COBBLESTONE, COPPER_ORE, DIRT,
    ELECTRIC_FURNACE, FIRE, GRASS, IRON_ORE, LAVA, OAK_LEAVES, OBSIDIAN, OIL,
    PURPLE_STAINED_GLASS, SIMPLE_FURNACE, STONE, TIN_ORE, TNT, TORCH, WATER, WOOD, WORKBENCH,
};

/// Sentinel value in `Vertex::tex_layer` indicating an untextured vertex-colored surface
/// (e.g. `Water`, `Oil`, `Cable`).
pub const NO_TEXTURE: u32 = u32::MAX;

pub const LAVA_ANIM_START: u32 = 32;
pub const LAVA_FRAMES: u32 = 20;
pub const FIRE_ANIM_START: u32 = 52;
pub const FIRE_FRAMES: u32 = 32;

/// Total number of 16x16 layers in the `texture_2d_array<f32>`:
///   - `0..32`: base static textures + frame 0 of lava/fire
///   - `32..52`: 20 animation frames of `lava_still.png`
///   - `52..84`: 32 animation frames of `fire_layer_0.png`
pub const TOTAL_LAYERS: u32 = FIRE_ANIM_START + FIRE_FRAMES;

/// Sequence of 16x16 base layers in the GPU 2D Texture Array (`0..32`).
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

/// Unified declarative block definition: per-face textures, opacity, and light emission.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub struct BlockDef {
    pub id: u16,
    pub name: &'static str,
    pub top_tex: &'static str,
    pub bottom_tex: &'static str,
    pub north_tex: &'static str,
    pub south_tex: &'static str,
    pub east_tex: &'static str,
    pub west_tex: &'static str,
    pub is_opaque: bool,
    /// > 0.0 ONLY for light sources (`lava`, `torch`, `fire`, active furnace front).
    pub emission: f32,
}

impl BlockDef {
    const fn uniform(
        id: u16,
        name: &'static str,
        tex: &'static str,
        is_opaque: bool,
        emission: f32,
    ) -> Self {
        Self {
            id,
            name,
            top_tex: tex,
            bottom_tex: tex,
            north_tex: tex,
            south_tex: tex,
            east_tex: tex,
            west_tex: tex,
            is_opaque,
            emission,
        }
    }
}

#[allow(dead_code)]
pub const BLOCK_DEFS: &[BlockDef] = &[
    BlockDef::uniform(STONE, "Stone", "stone.png", true, 0.0),
    BlockDef::uniform(DIRT, "Dirt", "dirt.png", true, 0.0),
    BlockDef {
        id: GRASS,
        name: "Grass",
        top_tex: "grass_top.png",
        bottom_tex: "dirt.png",
        north_tex: "grass_side.png",
        south_tex: "grass_side.png",
        east_tex: "grass_side.png",
        west_tex: "grass_side.png",
        is_opaque: true,
        emission: 0.0,
    },
    BlockDef::uniform(COBBLESTONE, "Cobblestone", "cobblestone.png", true, 0.0),
    BlockDef::uniform(COAL_ORE, "Coal Ore", "coal_ore.png", true, 0.0),
    BlockDef::uniform(COPPER_ORE, "Copper Ore", "iron_ore.png", true, 0.0),
    BlockDef::uniform(TIN_ORE, "Tin Ore", "iron_ore.png", true, 0.0),
    BlockDef::uniform(IRON_ORE, "Iron Ore", "iron_ore.png", true, 0.0),
    BlockDef::uniform(OBSIDIAN, "Obsidian", "obsidian.png", true, 0.0),
    BlockDef::uniform(BEDROCK, "Bedrock", "obsidian.png", true, 0.0),
    BlockDef {
        id: WOOD,
        name: "Oak Log",
        top_tex: "log_oak_top.png",
        bottom_tex: "log_oak_top.png",
        north_tex: "log_oak.png",
        south_tex: "log_oak.png",
        east_tex: "log_oak.png",
        west_tex: "log_oak.png",
        is_opaque: true,
        emission: 0.0,
    },
    BlockDef::uniform(OAK_LEAVES, "Oak Leaves", "leaves_oak.png", false, 0.0),
    BlockDef::uniform(
        PURPLE_STAINED_GLASS,
        "Purple Stained Glass",
        "glass_magenta.png",
        false,
        0.0,
    ),
    BlockDef {
        id: WORKBENCH,
        name: "Crafting Workbench",
        top_tex: "crafting_table_top.png",
        bottom_tex: "log_oak.png",
        north_tex: "crafting_table_side.png",
        south_tex: "crafting_table_front.png",
        east_tex: "crafting_table_side.png",
        west_tex: "crafting_table_side.png",
        is_opaque: true,
        emission: 0.0,
    },
    BlockDef {
        id: SIMPLE_FURNACE,
        name: "Stone Furnace",
        top_tex: "furnace_top.png",
        bottom_tex: "furnace_side.png",
        north_tex: "furnace_side.png",
        south_tex: "furnace_front_off.png",
        east_tex: "furnace_side.png",
        west_tex: "furnace_side.png",
        is_opaque: true,
        emission: 0.0,
    },
    BlockDef {
        id: TNT,
        name: "TNT",
        top_tex: "tnt_top.png",
        bottom_tex: "tnt_bottom.png",
        north_tex: "tnt_side.png",
        south_tex: "tnt_side.png",
        east_tex: "tnt_side.png",
        west_tex: "tnt_side.png",
        is_opaque: true,
        emission: 0.0,
    },
    BlockDef::uniform(LAVA, "Lava", "lava_still.png", false, 2.2),
    BlockDef::uniform(TORCH, "Torch", "torch_on.png", false, 2.0),
    BlockDef::uniform(FIRE, "Fire", "fire_layer_0.png", false, 2.5),
];

#[allow(dead_code)]
pub fn block_def(id: BlockId) -> Option<&'static BlockDef> {
    let base = base_block_id(id);
    BLOCK_DEFS.iter().find(|d| d.id == base)
}

#[inline]
fn front_face_for(facing: u8) -> BlockFace {
    match facing & 0x03 {
        0 => BlockFace::NegZ, // North (-Z)
        1 => BlockFace::PosX, // East (+X)
        2 => BlockFace::PosZ, // South (+Z)
        _ => BlockFace::NegX, // West (-X)
    }
}

/// Returns the emissive HDR radiance (`0.0` for all normal blocks, `> 0.0` only for
/// `Lava`, `Torch`, `Fire`, and the glowing front face of active furnaces/generators).
pub fn block_face_emission(block_id: BlockId, face: BlockFace, is_active: bool) -> f32 {
    let base = base_block_id(block_id);
    let active = is_active || is_active_block(block_id);
    let front = front_face_for(facing_from_id(block_id));
    match base {
        LAVA => 2.2,
        TORCH => 2.0,
        FIRE => 2.5,
        SIMPLE_FURNACE | COAL_GENERATOR | ELECTRIC_FURNACE | ALLOY_SMELTER => {
            if active && face == front {
                2.4
            } else {
                0.0
            }
        }
        _ => 0.0,
    }
}

/// Maps `(block_id, face, is_active)` to a `TextureLayer` index in the GPU `texture_2d_array`,
/// or `NO_TEXTURE` (`u32::MAX`) for procedural/vertex-colored materials (`Water`, `Oil`, `Cable`).
pub fn get_face_texture(block_id: BlockId, face: BlockFace, is_active: bool) -> u32 {
    let base = base_block_id(block_id);
    let active = is_active || is_active_block(block_id);
    let front = front_face_for(facing_from_id(block_id));

    match base {
        GRASS => match face {
            BlockFace::PosY => TextureLayer::GrassTop as u32,
            BlockFace::NegY => TextureLayer::Dirt as u32,
            _ => TextureLayer::GrassSide as u32,
        },
        WORKBENCH => {
            if face == BlockFace::PosY {
                TextureLayer::CraftingTableTop as u32
            } else if face == BlockFace::NegY {
                TextureLayer::LogOakSide as u32
            } else if face == front {
                TextureLayer::CraftingTableFront as u32
            } else {
                TextureLayer::CraftingTableSide as u32
            }
        }
        SIMPLE_FURNACE | COAL_GENERATOR | ELECTRIC_FURNACE | ALLOY_SMELTER => {
            if face == front {
                if active {
                    TextureLayer::FurnaceFrontOn as u32
                } else {
                    TextureLayer::FurnaceFrontOff as u32
                }
            } else if face == BlockFace::PosY {
                TextureLayer::FurnaceTop as u32
            } else {
                TextureLayer::FurnaceSide as u32
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_def_registry_has_expected_textures_and_emissions() {
        let tnt = block_def(TNT).expect("TNT registered");
        assert_eq!(tnt.top_tex, "tnt_top.png");
        assert_eq!(tnt.bottom_tex, "tnt_bottom.png");
        assert_eq!(tnt.north_tex, "tnt_side.png");
        assert_eq!(tnt.emission, 0.0);

        let log = block_def(WOOD).expect("Oak Log registered");
        assert_eq!(log.top_tex, "log_oak_top.png");
        assert_eq!(log.north_tex, "log_oak.png");
        assert_eq!(log.emission, 0.0);

        let torch = block_def(TORCH).expect("Torch registered");
        assert!(torch.emission > 0.0);
        let lava = block_def(LAVA).expect("Lava registered");
        assert!(lava.emission > 0.0);
    }

    #[test]
    fn facing_rotates_furnace_and_workbench_front_face() {
        use crate::voxel::chunk::with_facing;

        // Facing 0 = North (-Z), 1 = East (+X), 2 = South (+Z), 3 = West (-X)
        let f_east = with_facing(SIMPLE_FURNACE, 1);
        assert_eq!(
            get_face_texture(f_east, BlockFace::PosX, true),
            TextureLayer::FurnaceFrontOn as u32
        );
        assert_eq!(
            get_face_texture(f_east, BlockFace::PosZ, true),
            TextureLayer::FurnaceSide as u32
        );

        let w_west = with_facing(WORKBENCH, 3);
        assert_eq!(
            get_face_texture(w_west, BlockFace::NegX, false),
            TextureLayer::CraftingTableFront as u32
        );
    }
}
