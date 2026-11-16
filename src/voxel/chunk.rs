// Voxel chunk: 32x32x32 blocks stored flat on the heap (DOD, cache-friendly).

pub const CHUNK_SIZE: usize = 32;
const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

pub type BlockId = u16;

pub const AIR: BlockId = 0;
pub const STONE: BlockId = 1;
pub const DIRT: BlockId = 2;
pub const GRASS: BlockId = 3;
pub const COAL_ORE: BlockId = 4;
pub const COPPER_ORE: BlockId = 5;
pub const CABLE: BlockId = 6;
pub const COAL_GENERATOR: BlockId = 7;
pub const ELECTRIC_FURNACE: BlockId = 8;
/// Item only (furnace output); not a placeable block.
pub const COPPER_INGOT: BlockId = 9;
pub const ALLOY_SMELTER: BlockId = 10;
/// Item only (alloy smelter output); not a placeable block.
pub const BRONZE_INGOT: BlockId = 11;
pub const BATBOX: BlockId = 12;
pub const CHEST: BlockId = 13;
pub const WORKBENCH: BlockId = 14;
pub const SIMPLE_FURNACE: BlockId = 15;

// Step 7, 7.5 & 8: Metals, fluids, foliage, explosives, Torch, and Purple Stained Glass
pub const TIN_ORE: BlockId = 16;
pub const TIN_INGOT: BlockId = 17;
pub const IRON_ORE: BlockId = 18;
pub const IRON_INGOT: BlockId = 19;
pub const STEEL_INGOT: BlockId = 20;
pub const WATER: BlockId = 21;
pub const LAVA: BlockId = 22;
pub const OIL: BlockId = 23;
pub const OBSIDIAN: BlockId = 24;
pub const COBBLESTONE: BlockId = 25;
pub const TNT: BlockId = 26;
pub const FIRE: BlockId = 27;
pub const OAK_LEAVES: BlockId = 28;
pub const TORCH: BlockId = 29;
pub const LIGHTER: BlockId = 30;
pub const PURPLE_STAINED_GLASS: BlockId = 31;
pub const WOOD: BlockId = 32;
pub const BEDROCK: BlockId = 33;

// Step 17: Stone variants, Wood Planks, Sandstone, Bookshelf, Mossy Cobble, Stained Glass & Cross Foliage
pub const GRANITE: BlockId = 34;
pub const DIORITE: BlockId = 35;
pub const ANDESITE: BlockId = 36;
pub const OAK_PLANKS: BlockId = 37;
pub const BIRCH_PLANKS: BlockId = 38;
pub const SPRUCE_PLANKS: BlockId = 39;
pub const JUNGLE_PLANKS: BlockId = 40;
pub const ACACIA_PLANKS: BlockId = 41;
pub const DARK_OAK_PLANKS: BlockId = 42;
pub const SANDSTONE: BlockId = 43;
pub const BOOKSHELF: BlockId = 44;
pub const MOSSY_COBBLESTONE: BlockId = 45;
pub const BLACK_STAINED_GLASS: BlockId = 46;
pub const RED_STAINED_GLASS: BlockId = 47;
pub const BLUE_STAINED_GLASS: BlockId = 48;
pub const GREEN_STAINED_GLASS: BlockId = 49;
pub const YELLOW_STAINED_GLASS: BlockId = 50;
pub const TALL_GRASS: BlockId = 51;
pub const ROSE: BlockId = 52;
pub const DANDELION: BlockId = 53;
pub const DEAD_BUSH: BlockId = 54;
pub const SAND: BlockId = 55;

pub const ALL_ITEMS: [BlockId; 55] = [
    STONE,
    GRANITE,
    DIORITE,
    ANDESITE,
    COBBLESTONE,
    MOSSY_COBBLESTONE,
    DIRT,
    GRASS,
    SAND,
    SANDSTONE,
    WOOD,
    OAK_PLANKS,
    BIRCH_PLANKS,
    SPRUCE_PLANKS,
    JUNGLE_PLANKS,
    ACACIA_PLANKS,
    DARK_OAK_PLANKS,
    BOOKSHELF,
    OAK_LEAVES,
    TALL_GRASS,
    ROSE,
    DANDELION,
    DEAD_BUSH,
    OBSIDIAN,
    BEDROCK,
    COAL_ORE,
    COPPER_ORE,
    TIN_ORE,
    IRON_ORE,
    COPPER_INGOT,
    TIN_INGOT,
    IRON_INGOT,
    BRONZE_INGOT,
    STEEL_INGOT,
    WORKBENCH,
    CHEST,
    SIMPLE_FURNACE,
    CABLE,
    COAL_GENERATOR,
    BATBOX,
    ELECTRIC_FURNACE,
    ALLOY_SMELTER,
    TORCH,
    PURPLE_STAINED_GLASS,
    BLACK_STAINED_GLASS,
    RED_STAINED_GLASS,
    BLUE_STAINED_GLASS,
    GREEN_STAINED_GLASS,
    YELLOW_STAINED_GLASS,
    WATER,
    LAVA,
    OIL,
    LIGHTER,
    TNT,
    FIRE,
];

/// High bit used in mesher snapshots to mark a machine block that is currently active/lit.
pub const ACTIVE_MACHINE_FLAG: BlockId = 0x8000;

/// Extracts the base block/item ID (`0..=255`), stripping fluid level, facing, and active flags.
#[inline]
pub fn base_block_id(id: BlockId) -> BlockId {
    id & 0x00FF
}

#[inline]
pub fn is_active_block(id: BlockId) -> bool {
    (id & ACTIVE_MACHINE_FLAG) != 0
}

/// Packs a horizontal facing (`0 = North (-Z), 1 = East (+X), 2 = South (+Z), 3 = West (-X)`)
/// into bits `12..=13` of a snapshot `BlockId`.
#[inline]
pub fn with_facing(id: BlockId, facing: u8) -> BlockId {
    (id & !0x3000) | (((facing & 0x03) as u16) << 12)
}

/// Reads the horizontal facing (`0..=3`) from a snapshot `BlockId`.
#[inline]
pub fn facing_from_id(id: BlockId) -> u8 {
    ((id >> 12) & 0x03) as u8
}

/// Packs a fluid decay level (`1..=8`) into bits `8..=11` of a snapshot `BlockId`.
#[inline]
pub fn with_fluid_level(id: BlockId, level: u8) -> BlockId {
    base_block_id(id) | ((level.clamp(1, 8) as u16) << 8)
}

/// Reads the fluid level (`1..=8`, defaulting to `8` if unset) from a snapshot `BlockId`.
#[inline]
pub fn fluid_level_from_id(id: BlockId) -> u8 {
    let lvl = ((id >> 8) & 0x0F) as u8;
    if lvl == 0 {
        8
    } else {
        lvl.min(8)
    }
}

/// True if `id` is a liquid block (`Water`, `Lava`, or `Oil`).
#[inline]
pub fn is_fluid(id: BlockId) -> bool {
    matches!(base_block_id(id), WATER | LAVA | OIL)
}

/// True if `id` is a cross-mesh foliage block (`TallGrass`, `Rose`, `Dandelion`, `DeadBush`).
#[inline]
pub fn is_cross_foliage(id: BlockId) -> bool {
    matches!(
        base_block_id(id),
        TALL_GRASS | ROSE | DANDELION | DEAD_BUSH
    )
}

/// True if `id` is any colored stained glass block.
#[inline]
pub fn is_stained_glass(id: BlockId) -> bool {
    matches!(
        base_block_id(id),
        PURPLE_STAINED_GLASS
            | BLACK_STAINED_GLASS
            | RED_STAINED_GLASS
            | BLUE_STAINED_GLASS
            | GREEN_STAINED_GLASS
            | YELLOW_STAINED_GLASS
    )
}

/// True if `id` is rendered in the translucent alpha-blended pass (`Water`, `Fire`, Stained Glass).
#[inline]
pub fn is_translucent(id: BlockId) -> bool {
    matches!(base_block_id(id), WATER | FIRE) || is_stained_glass(id)
}

/// True if `id` blocks physical movement (AABB collisions).
/// Air, fluids (`Water`, `Lava`, `Oil`), `Fire`, `Torch`, and cross-mesh foliage are passable.
#[inline]
pub fn is_solid_block(id: BlockId) -> bool {
    let base = base_block_id(id);
    base != AIR && !is_fluid(base) && base != FIRE && base != TORCH && !is_cross_foliage(base)
}

/// True if `id` resists TNT explosions (`Bedrock` and `Obsidian`).
#[inline]
pub fn is_blast_resistant(id: BlockId) -> bool {
    matches!(base_block_id(id), BEDROCK | OBSIDIAN)
}

/// True if `id` can catch fire or be consumed by neighboring `Fire`.
#[inline]
pub fn is_flammable(id: BlockId) -> bool {
    matches!(
        base_block_id(id),
        WOOD | OAK_PLANKS
            | BIRCH_PLANKS
            | SPRUCE_PLANKS
            | JUNGLE_PLANKS
            | ACACIA_PLANKS
            | DARK_OAK_PLANKS
            | BOOKSHELF
            | OAK_LEAVES
            | OIL
            | TALL_GRASS
            | DEAD_BUSH
    )
}

/// True if `id` can be placed in the world as a voxel block.
pub fn is_placeable(id: BlockId) -> bool {
    matches!(
        base_block_id(id),
        STONE
            | GRANITE
            | DIORITE
            | ANDESITE
            | COBBLESTONE
            | MOSSY_COBBLESTONE
            | DIRT
            | GRASS
            | SAND
            | SANDSTONE
            | WOOD
            | OAK_PLANKS
            | BIRCH_PLANKS
            | SPRUCE_PLANKS
            | JUNGLE_PLANKS
            | ACACIA_PLANKS
            | DARK_OAK_PLANKS
            | BOOKSHELF
            | OAK_LEAVES
            | TALL_GRASS
            | ROSE
            | DANDELION
            | DEAD_BUSH
            | OBSIDIAN
            | BEDROCK
            | COAL_ORE
            | COPPER_ORE
            | TIN_ORE
            | IRON_ORE
            | CABLE
            | COAL_GENERATOR
            | ELECTRIC_FURNACE
            | ALLOY_SMELTER
            | BATBOX
            | CHEST
            | WORKBENCH
            | SIMPLE_FURNACE
            | TORCH
            | PURPLE_STAINED_GLASS
            | BLACK_STAINED_GLASS
            | RED_STAINED_GLASS
            | BLUE_STAINED_GLASS
            | GREEN_STAINED_GLASS
            | YELLOW_STAINED_GLASS
            | WATER
            | LAVA
            | OIL
            | TNT
            | FIRE
    )
}

/// True if the block occupies an opaque full 1x1x1 voxel cell (occludes neighbour faces).
/// Cables, torches, variable-height fluids, cross foliage, and translucent blocks do not occlude solid faces.
#[inline]
pub fn is_full_cube(id: BlockId) -> bool {
    let base = base_block_id(id);
    base != AIR
        && base != CABLE
        && base != TORCH
        && !is_fluid(base)
        && !is_translucent(base)
        && !is_cross_foliage(base)
}

/// True if a cable arm should visually and electrically connect to `id`.
#[inline]
pub fn is_energy_connectable(id: BlockId) -> bool {
    matches!(
        base_block_id(id),
        CABLE | COAL_GENERATOR | ELECTRIC_FURNACE | ALLOY_SMELTER | BATBOX
    )
}

pub const CABLE_MIN: f32 = 0.35;
pub const CABLE_MAX: f32 = 0.65;

pub fn block_name(id: BlockId) -> &'static str {
    match base_block_id(id) {
        AIR => "Air",
        STONE => "Stone",
        DIRT => "Dirt",
        GRASS => "Grass",
        COAL_ORE => "Coal Ore",
        COPPER_ORE => "Copper Ore",
        CABLE => "Cable",
        COAL_GENERATOR => "Coal Generator",
        ELECTRIC_FURNACE => "Electric Furnace",
        COPPER_INGOT => "Copper Ingot",
        ALLOY_SMELTER => "Alloy Smelter",
        BRONZE_INGOT => "Bronze Ingot",
        BATBOX => "BatBox (40k EU)",
        CHEST => "Storage Chest",
        WORKBENCH => "Crafting Workbench",
        SIMPLE_FURNACE => "Stone Furnace",
        TIN_ORE => "Tin Ore",
        TIN_INGOT => "Tin Ingot",
        IRON_ORE => "Iron Ore",
        IRON_INGOT => "Iron Ingot",
        STEEL_INGOT => "Steel Ingot",
        WATER => "Water",
        LAVA => "Lava",
        OIL => "Crude Oil",
        OBSIDIAN => "Obsidian",
        COBBLESTONE => "Cobblestone",
        TNT => "TNT (Dynamite)",
        FIRE => "Fire",
        OAK_LEAVES => "Oak Leaves",
        TORCH => "Torch",
        LIGHTER => "Lighter (Flint & Steel)",
        PURPLE_STAINED_GLASS => "Purple Stained Glass",
        WOOD => "Oak Wood Log",
        BEDROCK => "Bedrock",
        GRANITE => "Granite",
        DIORITE => "Diorite",
        ANDESITE => "Andesite",
        OAK_PLANKS => "Oak Planks",
        BIRCH_PLANKS => "Birch Planks",
        SPRUCE_PLANKS => "Spruce Planks",
        JUNGLE_PLANKS => "Jungle Planks",
        ACACIA_PLANKS => "Acacia Planks",
        DARK_OAK_PLANKS => "Dark Oak Planks",
        SANDSTONE => "Sandstone",
        BOOKSHELF => "Bookshelf",
        MOSSY_COBBLESTONE => "Mossy Cobblestone",
        BLACK_STAINED_GLASS => "Black Stained Glass",
        RED_STAINED_GLASS => "Red Stained Glass",
        BLUE_STAINED_GLASS => "Blue Stained Glass",
        GREEN_STAINED_GLASS => "Green Stained Glass",
        YELLOW_STAINED_GLASS => "Yellow Stained Glass",
        TALL_GRASS => "Tall Grass",
        ROSE => "Rose",
        DANDELION => "Dandelion",
        DEAD_BUSH => "Dead Bush",
        SAND => "Sand",
        _ => "Unknown",
    }
}

/// Base (sRGB) color of a block or item (inactive state for machines).
pub fn block_color(id: BlockId) -> [f32; 3] {
    match base_block_id(id) {
        STONE => [0.50, 0.50, 0.52],
        DIRT => [0.45, 0.30, 0.18],
        GRASS => [0.30, 0.65, 0.20],
        COAL_ORE => [0.14, 0.14, 0.16],
        COPPER_ORE => [0.82, 0.46, 0.22],
        CABLE => [0.90, 0.62, 0.12],
        COAL_GENERATOR => [0.36, 0.30, 0.30],       // dark cast-iron red-grey
        ELECTRIC_FURNACE => [0.28, 0.36, 0.48],     // industrial steel-blue
        COPPER_INGOT => [0.98, 0.58, 0.28],         // bright copper ingot
        ALLOY_SMELTER => [0.46, 0.32, 0.22],        // heavy bronze-steel casing
        BRONZE_INGOT => [0.86, 0.62, 0.20],         // golden-brown bronze ingot
        BATBOX => [0.22, 0.46, 0.38],               // battery pack teal-green
        CHEST => [0.68, 0.48, 0.22],                // oak wood chest
        WORKBENCH => [0.58, 0.36, 0.20],            // crafting table wood
        SIMPLE_FURNACE => [0.42, 0.42, 0.44],       // cobblestone furnace
        TIN_ORE => [0.68, 0.74, 0.78],              // pale silvery-blue tin ore
        TIN_INGOT => [0.82, 0.88, 0.92],            // bright silver-white tin ingot
        IRON_ORE => [0.64, 0.50, 0.42],             // rusty ochre iron ore
        IRON_INGOT => [0.76, 0.78, 0.80],           // metallic iron ingot
        STEEL_INGOT => [0.38, 0.42, 0.48],          // dark tempered carbon-steel ingot
        WATER => [0.16, 0.54, 0.94],                // translucent azure water
        LAVA => [0.98, 0.38, 0.08],                 // molten orange-red lava
        OIL => [0.14, 0.07, 0.20],                  // dense dark violet-black crude oil
        OBSIDIAN => [0.16, 0.10, 0.26],             // volcanic dark purple-black glass
        COBBLESTONE => [0.40, 0.40, 0.42],          // rough fractured stone
        TNT => [0.88, 0.18, 0.16],                  // bright dynamite red
        FIRE => [0.98, 0.68, 0.14],                 // bright flame orange-gold
        OAK_LEAVES => [0.16, 0.48, 0.14],           // lush deep forest green leaves
        TORCH => [1.00, 0.82, 0.32],                // warm glowing torch
        LIGHTER => [0.92, 0.74, 0.32],              // flint & steel striker
        PURPLE_STAINED_GLASS => [0.65, 0.16, 0.88], // rich purple stained glass
        WOOD => [0.54, 0.36, 0.18],                 // timber brown log
        BEDROCK => [0.15, 0.15, 0.17],              // unbreakable dark bedrock
        GRANITE => [0.62, 0.44, 0.38],
        DIORITE => [0.78, 0.78, 0.78],
        ANDESITE => [0.56, 0.56, 0.58],
        OAK_PLANKS => [0.68, 0.52, 0.30],
        BIRCH_PLANKS => [0.80, 0.74, 0.52],
        SPRUCE_PLANKS => [0.45, 0.32, 0.19],
        JUNGLE_PLANKS => [0.66, 0.48, 0.34],
        ACACIA_PLANKS => [0.72, 0.40, 0.22],
        DARK_OAK_PLANKS => [0.28, 0.19, 0.11],
        SANDSTONE => [0.85, 0.80, 0.62],
        BOOKSHELF => [0.60, 0.42, 0.24],
        MOSSY_COBBLESTONE => [0.36, 0.48, 0.34],
        BLACK_STAINED_GLASS => [0.15, 0.15, 0.18],
        RED_STAINED_GLASS => [0.82, 0.18, 0.18],
        BLUE_STAINED_GLASS => [0.20, 0.32, 0.85],
        GREEN_STAINED_GLASS => [0.26, 0.60, 0.22],
        YELLOW_STAINED_GLASS => [0.88, 0.82, 0.22],
        TALL_GRASS => [0.34, 0.68, 0.20],
        ROSE => [0.86, 0.18, 0.18],
        DANDELION => [0.95, 0.85, 0.18],
        DEAD_BUSH => [0.58, 0.42, 0.22],
        SAND => [0.88, 0.84, 0.64],
        _ => [1.0, 0.0, 1.0],
    }
}

/// Face color used by the mesher. Values > 1.0 trigger emissive glow in the fragment shader
/// (active machines, molten `LAVA`, burning `FIRE`, and `TORCH` glow in the dark at night).
pub fn block_face_color(raw_id: BlockId, axis: usize) -> [f32; 3] {
    let base = base_block_id(raw_id);
    let is_side = axis != 1;

    // Always-emissive blocks:
    if base == LAVA {
        return [1.75, 0.50, 0.10];
    }
    if base == FIRE {
        return [1.90, 0.78, 0.18];
    }
    if base == TORCH {
        return if is_side {
            [1.65, 0.72, 0.24]
        } else {
            [1.95, 0.90, 0.42]
        };
    }

    if !is_active_block(raw_id) {
        if base == WORKBENCH && !is_side {
            return [0.72, 0.52, 0.30]; // lighter crafting grid top
        }
        if base == TNT {
            return if is_side {
                [0.90, 0.20, 0.18] // red dynamite sticks with white band
            } else {
                [0.92, 0.88, 0.84] // white-labelled fuse top/bottom
            };
        }
        if base == WOOD && !is_side {
            return [0.68, 0.50, 0.28]; // lighter tree-ring top/bottom
        }
        return block_color(base);
    }

    match base {
        COAL_GENERATOR | SIMPLE_FURNACE => {
            if is_side {
                [1.65, 0.55, 0.12] // fiery burning coal firebox
            } else {
                [0.62, 0.34, 0.26] // warm lit casing
            }
        }
        ELECTRIC_FURNACE => {
            if is_side {
                [1.70, 0.72, 0.18] // glowing molten viewport
            } else {
                [0.42, 0.56, 0.78] // energized steel-blue top
            }
        }
        ALLOY_SMELTER => {
            if is_side {
                [1.80, 0.82, 0.22] // white-hot induction alloy glow
            } else {
                [0.72, 0.48, 0.28] // hot bronze casing
            }
        }
        BATBOX => {
            if is_side {
                [0.30, 1.45, 0.95] // glowing electric charge indicator
            } else {
                [0.32, 0.65, 0.52]
            }
        }
        _ => block_color(base),
    }
}

pub struct Chunk {
    blocks: Box<[BlockId; CHUNK_VOLUME]>,
    /// Per-voxel metadata (`0..=3` in lower 2 bits = Facing: 0=North(-Z), 1=East(+X), 2=South(+Z), 3=West(-X)).
    metadata: Box<[u8; CHUNK_VOLUME]>,
    /// True when the chunk was modified after procedural generation (needs saving to disk).
    pub is_dirty: bool,
}

impl Chunk {
    pub fn new() -> Self {
        // Allocate on the heap directly; never builds a 64 KiB array on the stack.
        let blocks: Box<[BlockId; CHUNK_VOLUME]> = vec![AIR; CHUNK_VOLUME]
            .into_boxed_slice()
            .try_into()
            .expect("chunk size mismatch");
        let metadata: Box<[u8; CHUNK_VOLUME]> = vec![0u8; CHUNK_VOLUME]
            .into_boxed_slice()
            .try_into()
            .expect("chunk metadata size mismatch");
        Self {
            blocks,
            metadata,
            is_dirty: false,
        }
    }

    pub fn from_raw(blocks_vec: Vec<u16>, meta_vec: Vec<u8>) -> Option<Self> {
        if blocks_vec.len() != CHUNK_VOLUME {
            return None;
        }
        let blocks: Box<[BlockId; CHUNK_VOLUME]> = blocks_vec.into_boxed_slice().try_into().ok()?;
        let metadata: Box<[u8; CHUNK_VOLUME]> = if meta_vec.len() == CHUNK_VOLUME {
            meta_vec.into_boxed_slice().try_into().ok()?
        } else {
            vec![0u8; CHUNK_VOLUME].into_boxed_slice().try_into().ok()?
        };
        Some(Self {
            blocks,
            metadata,
            is_dirty: true,
        })
    }

    pub fn blocks_slice(&self) -> &[BlockId] {
        &*self.blocks
    }

    pub fn metadata_slice(&self) -> &[u8] {
        &*self.metadata
    }

    #[inline]
    fn index(x: usize, y: usize, z: usize) -> usize {
        x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE
    }

    #[inline]
    fn in_bounds(x: i32, y: i32, z: i32) -> bool {
        let n = CHUNK_SIZE as i32;
        (0..n).contains(&x) && (0..n).contains(&y) && (0..n).contains(&z)
    }

    /// Returns `AIR` for out-of-bounds coordinates.
    pub fn get_block(&self, x: i32, y: i32, z: i32) -> BlockId {
        if Self::in_bounds(x, y, z) {
            self.blocks[Self::index(x as usize, y as usize, z as usize)]
        } else {
            AIR
        }
    }

    /// Out-of-bounds writes are ignored.
    pub fn set_block(&mut self, x: i32, y: i32, z: i32, id: BlockId) {
        if Self::in_bounds(x, y, z) {
            self.blocks[Self::index(x as usize, y as usize, z as usize)] = id;
        }
    }

    /// Returns `0` for out-of-bounds coordinates.
    pub fn get_meta(&self, x: i32, y: i32, z: i32) -> u8 {
        if Self::in_bounds(x, y, z) {
            self.metadata[Self::index(x as usize, y as usize, z as usize)]
        } else {
            0
        }
    }

    /// Out-of-bounds writes are ignored.
    pub fn set_meta(&mut self, x: i32, y: i32, z: i32, meta: u8) {
        if Self::in_bounds(x, y, z) {
            self.metadata[Self::index(x as usize, y as usize, z as usize)] = meta;
        }
    }
}
