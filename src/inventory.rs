// Classical player inventory: 9-slot hotbar + 3x9 (27-slot) main backpack + 2x2 / 3x3 crafting
// buffers + cursor stack.

use crate::voxel::chunk::{
    block_name, BlockId, ALLOY_SMELTER, BATBOX, CABLE, CHEST, COAL_GENERATOR, COAL_ORE,
    COPPER_INGOT, COPPER_ORE, ELECTRIC_FURNACE, IRON_INGOT, IRON_ORE, LAVA, LIGHTER, OIL,
    PURPLE_STAINED_GLASS, SIMPLE_FURNACE, STONE, TIN_INGOT, TIN_ORE, TNT, TORCH, WATER, WOOD,
    WORKBENCH,
};

pub const HOTBAR_SIZE: usize = 9;
pub const MAIN_INV_ROWS: usize = 3;
pub const MAIN_INV_COLS: usize = 9;
pub const MAIN_INV_SIZE: usize = MAIN_INV_ROWS * MAIN_INV_COLS;
pub const MAX_STACK: u16 = 64;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ItemStack {
    pub id: BlockId,
    pub count: u16,
}

pub struct PlayerInventory {
    pub hotbar: [Option<ItemStack>; HOTBAR_SIZE],
    pub main: [Option<ItemStack>; MAIN_INV_SIZE],
    /// 2x2 hand-crafting grid inside the player inventory window (`E`).
    pub craft_2x2: [Option<ItemStack>; 4],
    /// 3x3 crafting grid used while a `Workbench` window is open.
    pub craft_3x3: [Option<ItemStack>; 9],
    pub selected_slot: usize,
    /// Item stack currently held on the mouse cursor while the UI is open.
    pub cursor_item: Option<ItemStack>,
}

impl PlayerInventory {
    pub fn new() -> Self {
        let mut hotbar = [None; HOTBAR_SIZE];
        hotbar[0] = Some(ItemStack { id: LIGHTER, count: 1 });
        hotbar[1] = Some(ItemStack { id: TNT, count: 32 });
        hotbar[2] = Some(ItemStack { id: TORCH, count: 32 });
        hotbar[3] = Some(ItemStack { id: PURPLE_STAINED_GLASS, count: 32 });
        hotbar[4] = Some(ItemStack { id: WATER, count: 32 });
        hotbar[5] = Some(ItemStack { id: LAVA, count: 32 });
        hotbar[6] = Some(ItemStack { id: WORKBENCH, count: 2 });
        hotbar[7] = Some(ItemStack { id: ALLOY_SMELTER, count: 2 });
        hotbar[8] = Some(ItemStack { id: COAL_GENERATOR, count: 4 });

        let mut main = [None; MAIN_INV_SIZE];
        main[0] = Some(ItemStack { id: CABLE, count: 32 });
        main[1] = Some(ItemStack { id: BATBOX, count: 2 });
        main[2] = Some(ItemStack { id: ELECTRIC_FURNACE, count: 2 });
        main[3] = Some(ItemStack { id: SIMPLE_FURNACE, count: 2 });
        main[4] = Some(ItemStack { id: CHEST, count: 2 });
        main[5] = Some(ItemStack { id: OIL, count: 16 });
        main[6] = Some(ItemStack { id: COAL_ORE, count: 32 });
        main[7] = Some(ItemStack { id: COPPER_ORE, count: 32 });
        main[8] = Some(ItemStack { id: TIN_ORE, count: 32 });
        main[9] = Some(ItemStack { id: IRON_ORE, count: 32 });
        main[10] = Some(ItemStack { id: COPPER_INGOT, count: 24 });
        main[11] = Some(ItemStack { id: TIN_INGOT, count: 16 });
        main[12] = Some(ItemStack { id: IRON_INGOT, count: 16 });
        main[13] = Some(ItemStack { id: WOOD, count: 32 });
        main[14] = Some(ItemStack { id: STONE, count: 32 });

        Self {
            hotbar,
            main,
            craft_2x2: [None; 4],
            craft_3x3: [None; 9],
            selected_slot: 0,
            cursor_item: None,
        }
    }

    pub fn select(&mut self, slot: usize) {
        if slot < HOTBAR_SIZE {
            self.selected_slot = slot;
        }
    }

    /// Moves the hotbar selection by `delta` slots, wrapping around.
    pub fn cycle(&mut self, delta: i32) {
        let n = HOTBAR_SIZE as i32;
        self.selected_slot = (self.selected_slot as i32 + delta).rem_euclid(n) as usize;
    }

    pub fn selected_stack(&self) -> Option<ItemStack> {
        self.hotbar[self.selected_slot]
    }

    /// Removes one item from the selected hotbar slot (clearing it when it reaches 0).
    pub fn take_one_selected(&mut self) -> Option<BlockId> {
        let stack = self.hotbar[self.selected_slot].as_mut()?;
        let id = stack.id;
        stack.count -= 1;
        if stack.count == 0 {
            self.hotbar[self.selected_slot] = None;
        }
        Some(id)
    }

    pub fn can_add(&self, id: BlockId) -> bool {
        self.hotbar
            .iter()
            .chain(self.main.iter())
            .any(|s| s.map_or(true, |s| s.id == id && s.count < MAX_STACK))
    }

    /// Adds items into the hotbar first, then the 3x9 main grid, stacking onto existing
    /// stacks before filling empty slots. Returns how many items did not fit.
    pub fn add(&mut self, id: BlockId, mut count: u16) -> u16 {
        for slot in self.hotbar.iter_mut().chain(self.main.iter_mut()).flatten() {
            if slot.id == id && slot.count < MAX_STACK {
                let moved = count.min(MAX_STACK - slot.count);
                slot.count += moved;
                count -= moved;
                if count == 0 {
                    return 0;
                }
            }
        }
        for slot in self.hotbar.iter_mut().chain(self.main.iter_mut()) {
            if count == 0 {
                break;
            }
            if slot.is_none() {
                let moved = count.min(MAX_STACK);
                *slot = Some(ItemStack { id, count: moved });
                count -= moved;
            }
        }
        count
    }

    /// Returns any item held on the cursor or left inside the 2x2 / 3x3 crafting grids
    /// back into the player's inventory when the UI closes. Returns overflow stacks.
    pub fn return_cursor_and_crafting(&mut self) -> Vec<ItemStack> {
        let mut overflow = Vec::new();
        let mut to_return: Vec<ItemStack> = Vec::new();
        if let Some(held) = self.cursor_item.take() {
            to_return.push(held);
        }
        for slot in self.craft_2x2.iter_mut().chain(self.craft_3x3.iter_mut()) {
            if let Some(s) = slot.take() {
                to_return.push(s);
            }
        }
        for item in to_return {
            let rem = self.add(item.id, item.count);
            if rem > 0 {
                overflow.push(ItemStack { id: item.id, count: rem });
            }
        }
        overflow
    }

    /// e.g. `[Slot 1: Stone x14]`
    pub fn status_line(&self) -> String {
        match self.selected_stack() {
            Some(s) => format!(
                "[Slot {}: {} x{}]",
                self.selected_slot + 1,
                block_name(s.id),
                s.count
            ),
            None => format!("[Slot {}: empty]", self.selected_slot + 1),
        }
    }
}

/// Standard slot interaction (left click = swap/merge all, right click = split/place one).
/// If `filter` is provided, only items accepted by `filter(id)` may be placed into `slot`.
pub fn interact_slot<F: Fn(BlockId) -> bool>(
    slot: &mut Option<ItemStack>,
    cursor: &mut Option<ItemStack>,
    right_click: bool,
    filter: F,
) {
    if !right_click {
        match (*slot, *cursor) {
            (None, None) => {}
            (Some(_), None) => {
                *cursor = slot.take();
            }
            (None, Some(held)) => {
                if filter(held.id) {
                    *slot = cursor.take();
                }
            }
            (Some(mut s), Some(mut held)) => {
                if s.id == held.id && filter(held.id) {
                    let space = MAX_STACK.saturating_sub(s.count);
                    let moved = held.count.min(space);
                    s.count += moved;
                    held.count -= moved;
                    *slot = Some(s);
                    *cursor = if held.count > 0 { Some(held) } else { None };
                } else if filter(held.id) {
                    *slot = Some(held);
                    *cursor = Some(s);
                }
            }
        }
    } else {
        match (*slot, *cursor) {
            (None, None) => {}
            (Some(mut s), None) => {
                let take = (s.count + 1) / 2;
                s.count -= take;
                *cursor = Some(ItemStack { id: s.id, count: take });
                *slot = if s.count > 0 { Some(s) } else { None };
            }
            (None, Some(mut held)) => {
                if filter(held.id) {
                    *slot = Some(ItemStack { id: held.id, count: 1 });
                    held.count -= 1;
                    *cursor = if held.count > 0 { Some(held) } else { None };
                }
            }
            (Some(mut s), Some(mut held)) => {
                if s.id == held.id && s.count < MAX_STACK && filter(held.id) {
                    s.count += 1;
                    held.count -= 1;
                    *slot = Some(s);
                    *cursor = if held.count > 0 { Some(held) } else { None };
                }
            }
        }
    }
}
