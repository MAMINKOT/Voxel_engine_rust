// Universal Recipe Registry supporting 2x2 / 3x3 Shaped & Shapeless Crafting,
// Smelting (Stone Furnace & Electric Furnace), and Dual-Input Alloying (Alloy Smelter).

use crate::inventory::ItemStack;
use crate::voxel::chunk::{
    ALLOY_SMELTER, BATBOX, BRONZE_INGOT, CABLE, CHEST, COAL_GENERATOR, COAL_ORE, COBBLESTONE,
    COPPER_INGOT, COPPER_ORE, DIRT, ELECTRIC_FURNACE, GRASS, IRON_INGOT, IRON_ORE, LIGHTER,
    PURPLE_STAINED_GLASS, SIMPLE_FURNACE, STEEL_INGOT, STONE, TIN_INGOT, TIN_ORE, TNT, TORCH,
    WOOD, WORKBENCH,
};

#[derive(Clone, Debug, PartialEq)]
pub enum Recipe {
    CraftingShaped {
        width: usize,
        height: usize,
        inputs: Vec<Option<u16>>, // Row-major item IDs in the trimmed pattern
        output: ItemStack,
    },
    CraftingShapeless {
        inputs: Vec<u16>,
        output: ItemStack,
    },
    Smelting {
        input_id: u16,
        output: ItemStack,
        cook_time_ticks: u16,
    },
    Alloy {
        input_a: u16,
        input_b: u16,
        count_a: u16,
        count_b: u16,
        output: ItemStack,
        cook_time_ticks: u16,
    },
}

impl Recipe {
    pub fn output_stack(&self) -> ItemStack {
        match self {
            Recipe::CraftingShaped { output, .. }
            | Recipe::CraftingShapeless { output, .. }
            | Recipe::Smelting { output, .. }
            | Recipe::Alloy { output, .. } => *output,
        }
    }

    pub fn uses_input(&self, id: u16) -> bool {
        match self {
            Recipe::CraftingShaped { inputs, .. } => inputs.iter().flatten().any(|&x| x == id),
            Recipe::CraftingShapeless { inputs, .. } => inputs.contains(&id),
            Recipe::Smelting { input_id, .. } => *input_id == id,
            Recipe::Alloy { input_a, input_b, .. } => *input_a == id || *input_b == id,
        }
    }
}

pub struct RecipeRegistry {
    recipes: Vec<Recipe>,
}

impl Default for RecipeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl RecipeRegistry {
    pub fn new() -> Self {
        let recipes = vec![
            // ── 2x2 Shaped Crafting (can be crafted in inventory or workbench) ──
            // 4 Wood -> 1 Workbench
            Recipe::CraftingShaped {
                width: 2,
                height: 2,
                inputs: vec![Some(WOOD), Some(WOOD), Some(WOOD), Some(WOOD)],
                output: ItemStack { id: WORKBENCH, count: 1 },
            },
            // 4 Stone -> 1 Workbench
            Recipe::CraftingShaped {
                width: 2,
                height: 2,
                inputs: vec![Some(STONE), Some(STONE), Some(STONE), Some(STONE)],
                output: ItemStack { id: WORKBENCH, count: 1 },
            },
            // 2 Coal + 2 Dirt checkerboard -> 2 TNT
            Recipe::CraftingShaped {
                width: 2,
                height: 2,
                inputs: vec![
                    Some(COAL_ORE), Some(DIRT),
                    Some(DIRT),     Some(COAL_ORE),
                ],
                output: ItemStack { id: TNT, count: 2 },
            },
            // 2 Dirt + 2 Stone -> 1 Storage Chest (compact 2x2 recipe)
            Recipe::CraftingShaped {
                width: 2,
                height: 2,
                inputs: vec![Some(DIRT), Some(DIRT), Some(STONE), Some(STONE)],
                output: ItemStack { id: CHEST, count: 1 },
            },
            // ── 3x3 Shaped Crafting (Workbench) ─────────────────────────────
            // 8 Dirt ring -> 1 Storage Chest
            Recipe::CraftingShaped {
                width: 3,
                height: 3,
                inputs: vec![
                    Some(DIRT), Some(DIRT), Some(DIRT),
                    Some(DIRT), None,       Some(DIRT),
                    Some(DIRT), Some(DIRT), Some(DIRT),
                ],
                output: ItemStack { id: CHEST, count: 1 },
            },
            // 8 Stone ring -> 1 Stone Furnace (Simple Furnace)
            Recipe::CraftingShaped {
                width: 3,
                height: 3,
                inputs: vec![
                    Some(STONE), Some(STONE), Some(STONE),
                    Some(STONE), None,        Some(STONE),
                    Some(STONE), Some(STONE), Some(STONE),
                ],
                output: ItemStack { id: SIMPLE_FURNACE, count: 1 },
            },
            // 3 Copper Ingots in a row -> 6 Copper Cables
            Recipe::CraftingShaped {
                width: 3,
                height: 1,
                inputs: vec![Some(COPPER_INGOT), Some(COPPER_INGOT), Some(COPPER_INGOT)],
                output: ItemStack { id: CABLE, count: 6 },
            },
            // Coal Generator: Stone corners, Copper sides, Coal top, Stone Furnace center, Cable bottom
            Recipe::CraftingShaped {
                width: 3,
                height: 3,
                inputs: vec![
                    Some(STONE),        Some(COAL_ORE),       Some(STONE),
                    Some(COPPER_INGOT), Some(SIMPLE_FURNACE), Some(COPPER_INGOT),
                    Some(STONE),        Some(CABLE),          Some(STONE),
                ],
                output: ItemStack { id: COAL_GENERATOR, count: 1 },
            },
            // Electric Furnace: Copper Ingots + Cables around a Stone Furnace
            Recipe::CraftingShaped {
                width: 3,
                height: 3,
                inputs: vec![
                    Some(COPPER_INGOT), Some(COPPER_INGOT),   Some(COPPER_INGOT),
                    Some(CABLE),        Some(SIMPLE_FURNACE), Some(CABLE),
                    Some(COPPER_INGOT), Some(CABLE),          Some(COPPER_INGOT),
                ],
                output: ItemStack { id: ELECTRIC_FURNACE, count: 1 },
            },
            // Alloy Smelter: Bronze/Copper + Electric Furnace + Stone Furnace + Cables
            Recipe::CraftingShaped {
                width: 3,
                height: 3,
                inputs: vec![
                    Some(COPPER_INGOT), Some(ELECTRIC_FURNACE), Some(COPPER_INGOT),
                    Some(COPPER_INGOT), Some(SIMPLE_FURNACE),   Some(COPPER_INGOT),
                    Some(CABLE),        Some(CABLE),            Some(CABLE),
                ],
                output: ItemStack { id: ALLOY_SMELTER, count: 1 },
            },
            // BatBox (40,000 EU): Cables + Copper Ingots + Coal + Stone casing
            Recipe::CraftingShaped {
                width: 3,
                height: 3,
                inputs: vec![
                    Some(STONE),        Some(CABLE),    Some(STONE),
                    Some(COPPER_INGOT), Some(COAL_ORE), Some(COPPER_INGOT),
                    Some(STONE),        Some(CABLE),    Some(STONE),
                ],
                output: ItemStack { id: BATBOX, count: 1 },
            },
            // ── Shapeless Crafting ──────────────────────────────────────────
            Recipe::CraftingShapeless {
                inputs: vec![COAL_ORE, WOOD],
                output: ItemStack { id: TORCH, count: 4 },
            },
            Recipe::CraftingShapeless {
                inputs: vec![STONE, STONE, COAL_ORE],
                output: ItemStack { id: PURPLE_STAINED_GLASS, count: 4 },
            },
            Recipe::CraftingShapeless {
                inputs: vec![IRON_INGOT, STONE],
                output: ItemStack { id: LIGHTER, count: 1 },
            },
            Recipe::CraftingShapeless {
                inputs: vec![DIRT, GRASS],
                output: ItemStack { id: GRASS, count: 2 },
            },
            Recipe::CraftingShapeless {
                inputs: vec![BRONZE_INGOT],
                output: ItemStack { id: CABLE, count: 4 },
            },
            // ── Smelting (Stone Furnace & Electric Furnace) ─────────────────
            Recipe::Smelting {
                input_id: COPPER_ORE,
                output: ItemStack { id: COPPER_INGOT, count: 1 },
                cook_time_ticks: 100,
            },
            Recipe::Smelting {
                input_id: TIN_ORE,
                output: ItemStack { id: TIN_INGOT, count: 1 },
                cook_time_ticks: 100,
            },
            Recipe::Smelting {
                input_id: IRON_ORE,
                output: ItemStack { id: IRON_INGOT, count: 1 },
                cook_time_ticks: 120,
            },
            Recipe::Smelting {
                input_id: COBBLESTONE,
                output: ItemStack { id: STONE, count: 1 },
                cook_time_ticks: 80,
            },
            // ── Alloy Smelter ───────────────────────────────────────────────
            Recipe::Alloy {
                input_a: COPPER_INGOT,
                input_b: TIN_INGOT,
                count_a: 3,
                count_b: 1,
                output: ItemStack { id: BRONZE_INGOT, count: 4 },
                cook_time_ticks: 200,
            },
            Recipe::Alloy {
                input_a: IRON_INGOT,
                input_b: COAL_ORE,
                count_a: 1,
                count_b: 2,
                output: ItemStack { id: STEEL_INGOT, count: 1 },
                cook_time_ticks: 300,
            },
        ];

        Self { recipes }
    }

    pub fn is_alloy_ingredient(&self, id: u16) -> bool {
        self.recipes.iter().any(|r| {
            matches!(
                r,
                Recipe::Alloy { input_a, input_b, .. } if *input_a == id || *input_b == id
            )
        })
    }

    pub fn recipes_for_output(&self, id: u16) -> Vec<&Recipe> {
        self.recipes
            .iter()
            .filter(|r| r.output_stack().id == id)
            .collect()
    }

    pub fn recipes_using_input(&self, id: u16) -> Vec<&Recipe> {
        self.recipes.iter().filter(|r| r.uses_input(id)).collect()
    }

    /// Matches a 2D crafting grid (`grid_w × grid_h`) against all shaped and shapeless recipes.
    pub fn match_crafting(
        &self,
        grid: &[Option<ItemStack>],
        grid_w: usize,
        grid_h: usize,
    ) -> Option<ItemStack> {
        let (tw, th, trimmed) = trim_grid(grid, grid_w, grid_h)?;

        for recipe in &self.recipes {
            match recipe {
                Recipe::CraftingShaped { width, height, inputs, output } => {
                    if *width == tw && *height == th && inputs == &trimmed {
                        return Some(*output);
                    }
                }
                Recipe::CraftingShapeless { inputs, output } => {
                    let mut present: Vec<u16> =
                        grid.iter().flatten().map(|stack| stack.id).collect();
                    if present.len() == inputs.len() {
                        let mut expected = inputs.clone();
                        present.sort_unstable();
                        expected.sort_unstable();
                        if present == expected {
                            return Some(*output);
                        }
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// Looks up a single-input smelting recipe (`SimpleFurnace` / `ElectricFurnace`).
    pub fn match_smelting(&self, input_id: u16) -> Option<(ItemStack, u16)> {
        for recipe in &self.recipes {
            if let Recipe::Smelting {
                input_id: rec_in,
                output,
                cook_time_ticks,
            } = recipe
            {
                if *rec_in == input_id {
                    return Some((*output, *cook_time_ticks));
                }
            }
        }
        None
    }

    /// Looks up a dual-input alloy recipe (matches `(A, B)` in either slot order).
    /// Returns `(consume_from_slot_a, consume_from_slot_b, output_stack, cook_time_ticks)`.
    pub fn match_alloy(
        &self,
        slot_a: Option<ItemStack>,
        slot_b: Option<ItemStack>,
    ) -> Option<(u16, u16, ItemStack, u16)> {
        let (a, b) = (slot_a?, slot_b?);
        for recipe in &self.recipes {
            if let Recipe::Alloy {
                input_a,
                input_b,
                count_a,
                count_b,
                output,
                cook_time_ticks,
            } = recipe
            {
                if a.id == *input_a && a.count >= *count_a && b.id == *input_b && b.count >= *count_b
                {
                    return Some((*count_a, *count_b, *output, *cook_time_ticks));
                }
                if a.id == *input_b && a.count >= *count_b && b.id == *input_a && b.count >= *count_a
                {
                    return Some((*count_b, *count_a, *output, *cook_time_ticks));
                }
            }
        }
        None
    }
}

/// Decrements every non-empty cell in `grid` by 1 item after a craft output is taken.
pub fn consume_crafting_ingredients(grid: &mut [Option<ItemStack>]) {
    for cell in grid.iter_mut() {
        if let Some(stack) = cell {
            stack.count = stack.count.saturating_sub(1);
            if stack.count == 0 {
                *cell = None;
            }
        }
    }
}

/// Trims empty outer rows and columns from a `w × h` crafting grid so shaped patterns match
/// regardless of where they are placed inside the grid.
fn trim_grid(
    grid: &[Option<ItemStack>],
    w: usize,
    h: usize,
) -> Option<(usize, usize, Vec<Option<u16>>)> {
    let mut min_x = w;
    let mut max_x = 0usize;
    let mut min_y = h;
    let mut max_y = 0usize;
    let mut any = false;

    for y in 0..h {
        for x in 0..w {
            if grid[y * w + x].is_some() {
                any = true;
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
            }
        }
    }
    if !any {
        return None;
    }

    let tw = max_x - min_x + 1;
    let th = max_y - min_y + 1;
    let mut out = Vec::with_capacity(tw * th);
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            out.push(grid[y * w + x].map(|s| s.id));
        }
    }
    Some((tw, th, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shaped_2x2_workbench_matches_anywhere_in_3x3_grid() {
        let reg = RecipeRegistry::new();
        let mut grid = [None; 9];
        // Place 2x2 stone in bottom-right corner of a 3x3 workbench grid.
        for idx in [4, 5, 7, 8] {
            grid[idx] = Some(ItemStack { id: STONE, count: 1 });
        }
        let out = reg.match_crafting(&grid, 3, 3).unwrap();
        assert_eq!(out, ItemStack { id: WORKBENCH, count: 1 });
    }

    #[test]
    fn stone_ring_3x3_crafts_simple_furnace() {
        let reg = RecipeRegistry::new();
        let mut grid = [Some(ItemStack { id: STONE, count: 1 }); 9];
        grid[4] = None;
        let out = reg.match_crafting(&grid, 3, 3).unwrap();
        assert_eq!(out, ItemStack { id: SIMPLE_FURNACE, count: 1 });
    }

    #[test]
    fn shapeless_and_alloy_lookup_work() {
        let reg = RecipeRegistry::new();
        let grid = [
            Some(ItemStack { id: GRASS, count: 1 }),
            None,
            None,
            Some(ItemStack { id: DIRT, count: 1 }),
        ];
        assert_eq!(
            reg.match_crafting(&grid, 2, 2),
            Some(ItemStack { id: GRASS, count: 2 })
        );

        let bronze = reg
            .match_alloy(
                Some(ItemStack { id: TIN_INGOT, count: 2 }),
                Some(ItemStack { id: COPPER_INGOT, count: 5 }),
            )
            .unwrap();
        assert_eq!(bronze.0, 1);
        assert_eq!(bronze.1, 3);
        assert_eq!(bronze.2, ItemStack { id: BRONZE_INGOT, count: 4 });

        let steel = reg
            .match_alloy(
                Some(ItemStack { id: IRON_INGOT, count: 2 }),
                Some(ItemStack { id: COAL_ORE, count: 4 }),
            )
            .unwrap();
        assert_eq!(steel.0, 1);
        assert_eq!(steel.1, 2);
        assert_eq!(steel.2, ItemStack { id: STEEL_INGOT, count: 1 });
        assert_eq!(steel.3, 300);
    }
}
