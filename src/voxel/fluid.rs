// Cellular automaton fluid simulation (rate-limited: once every 4 logic ticks = 5 Hz)
// with a `VecDeque<IVec3>` active coordinate queue, decay levels `1..=8`, ledge drop
// reset (`level = 8` on vertical fall), and vector current physics (`flow_vector_at`).
//
// Rules:
//   • Source block has `level = 8` (`MAX_FLUID_LEVEL`) and `is_source = true`.
//   • Ledge reset on drop: when fluid falls into `pos - Y` (or receives inflow from `pos + Y`),
//     the lower voxel receives `level = 8` (`MAX_FLUID_LEVEL`), restoring its full 7-block
//     horizontal spread budget on the new ledge below.
//   • Horizontal spread & decay: when resting on a solid block, each horizontal step
//     (`+X, -X, +Z, -Z`) receives `level - 1` (for Lava/Oil: `level - 2`). At `level <= 1`
//     the fluid stops spreading and dries up if its upstream feeder disappears.
//   • Water + Lava crystallization:
//       - `WATER` touching a `LAVA` source (`is_source == true`) -> `OBSIDIAN` (ID 24).
//       - `WATER` touching flowing `LAVA` (`is_source == false`) -> `COBBLESTONE` (ID 25).

use std::collections::{HashMap, HashSet, VecDeque};

use glam::{IVec3, Vec3};

use crate::voxel::chunk::{
    is_fluid, is_solid_block, BlockId, AIR, COBBLESTONE, FIRE, LAVA, OBSIDIAN, OIL, WATER,
};

pub const MAX_FLUID_LEVEL: u8 = 8;
/// Fluids update once every 4 logic ticks (5 updates/sec at 20 TPS).
pub const FLUID_TICK_INTERVAL: u32 = 4;

const HORIZONTAL_DIRS: [IVec3; 4] = [
    IVec3::new(1, 0, 0),
    IVec3::new(-1, 0, 0),
    IVec3::new(0, 0, 1),
    IVec3::new(0, 0, -1),
];

const ALL_DIRS: [IVec3; 6] = [
    IVec3::new(1, 0, 0),
    IVec3::new(-1, 0, 0),
    IVec3::new(0, 1, 0),
    IVec3::new(0, -1, 0),
    IVec3::new(0, 0, 1),
    IVec3::new(0, 0, -1),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FluidCell {
    pub kind: BlockId,
    pub level: u8,
    pub is_source: bool,
}

pub struct FluidSimulator {
    cells: HashMap<IVec3, FluidCell>,
    queue: VecDeque<IVec3>,
    queued: HashSet<IVec3>,
    tick_counter: u32,
}

impl FluidSimulator {
    pub fn new() -> Self {
        Self {
            cells: HashMap::new(),
            queue: VecDeque::new(),
            queued: HashSet::new(),
            tick_counter: 0,
        }
    }

    pub fn loaded_fluid_count(&self) -> usize {
        self.cells.len()
    }

    /// Returns the fluid decay level (`1..=8`) at `pos`, used by the mesher for proportional height.
    pub fn level_at(&self, pos: IVec3) -> Option<u8> {
        self.cells.get(&pos).map(|c| c.level)
    }

    /// Computes the 3D current vector at `pos`:
    ///   `D = sum((level_self - level_neighbor) * dir)`
    /// plus a downward pull when the fluid is falling into a lower cell.
    pub fn flow_vector_at(&self, pos: IVec3) -> Vec3 {
        let Some(&self_cell) = self.cells.get(&pos) else {
            return Vec3::ZERO;
        };
        let self_lvl = self_cell.level as f32;
        let mut dir_sum = Vec3::ZERO;

        for d in HORIZONTAL_DIRS {
            let np = pos + d;
            if let Some(nb) = self.cells.get(&np) {
                if nb.kind == self_cell.kind {
                    let diff = self_lvl - (nb.level as f32);
                    dir_sum += d.as_vec3() * diff;
                }
            } else if self.cells.contains_key(&(np - IVec3::Y)) {
                // Neighbor is a waterfall drop-off edge: pull toward the drop-off
                dir_sum += d.as_vec3() * self_lvl;
            }
        }

        // Downward waterfall current if fluid continues below or is fed from above
        if self.cells.contains_key(&(pos - IVec3::Y))
            || self.cells.contains_key(&(pos + IVec3::Y))
        {
            dir_sum.y -= 3.5;
        }

        if dir_sum.length_squared() > 1e-4 {
            dir_sum.normalize()
        } else {
            Vec3::ZERO
        }
    }

    /// Registers or removes a fluid cell when a block in the world is edited by the player/world.
    pub fn on_block_changed(&mut self, pos: IVec3, new_id: BlockId) {
        if is_fluid(new_id) {
            self.cells.entry(pos).or_insert(FluidCell {
                kind: new_id,
                level: MAX_FLUID_LEVEL,
                is_source: true,
            });
            self.enqueue_with_neighbors(pos);
        } else {
            self.cells.remove(&pos);
            self.enqueue_with_neighbors(pos);
        }
    }

    /// Inserts a fluid cell with an explicit level (`1..=8`).
    #[cfg(test)]
    pub fn set_fluid_with_level(&mut self, pos: IVec3, kind: BlockId, level: u8) {
        if is_fluid(kind) && level > 0 {
            let lvl = level.min(MAX_FLUID_LEVEL);
            self.cells.insert(
                pos,
                FluidCell {
                    kind,
                    level: lvl,
                    is_source: lvl == MAX_FLUID_LEVEL,
                },
            );
            self.enqueue_with_neighbors(pos);
        }
    }

    fn enqueue(&mut self, pos: IVec3) {
        if self.queued.insert(pos) {
            self.queue.push_back(pos);
        }
    }

    fn enqueue_with_neighbors(&mut self, pos: IVec3) {
        if self.cells.contains_key(&pos) {
            self.enqueue(pos);
        }
        for d in ALL_DIRS {
            let np = pos + d;
            if self.cells.contains_key(&np) {
                self.enqueue(np);
            }
        }
    }

    /// Checks whether a non-source fluid cell still has an upstream feeder
    /// (either the same fluid directly above it, or a horizontal neighbor with a higher level).
    fn has_upstream_feeder(&self, pos: IVec3, cell: FluidCell) -> bool {
        if cell.is_source {
            return true;
        }
        if let Some(above) = self.cells.get(&(pos + IVec3::Y)) {
            if above.kind == cell.kind {
                return true;
            }
        }
        for d in HORIZONTAL_DIRS {
            if let Some(nb) = self.cells.get(&(pos + d)) {
                if nb.kind == cell.kind && nb.level > cell.level {
                    return true;
                }
            }
        }
        false
    }

    /// Advances the fluid cellular automaton once every `FLUID_TICK_INTERVAL` (4) logic ticks.
    /// Processes only the currently queued coordinates from `self.queue`.
    pub fn tick<F: Fn(IVec3) -> BlockId>(&mut self, get_block: F) -> Vec<(IVec3, BlockId)> {
        self.tick_counter = self.tick_counter.wrapping_add(1);
        if self.tick_counter % FLUID_TICK_INTERVAL != 0 || self.queue.is_empty() {
            return Vec::new();
        }

        let batch_len = self.queue.len();
        let mut active = Vec::with_capacity(batch_len);
        for _ in 0..batch_len {
            if let Some(pos) = self.queue.pop_front() {
                self.queued.remove(&pos);
                active.push(pos);
            }
        }

        let mut block_edits: Vec<(IVec3, BlockId)> = Vec::new();
        let mut next_cells: Vec<(IVec3, FluidCell)> = Vec::new();
        let mut dried_up: Vec<IVec3> = Vec::new();
        let mut crystallized: HashSet<IVec3> = HashSet::new();

        for pos in active {
            if crystallized.contains(&pos) {
                continue;
            }
            let Some(mut cell) = self.cells.get(&pos).copied() else {
                continue;
            };
            let world_id = get_block(pos);
            if world_id != cell.kind {
                self.cells.remove(&pos);
                continue;
            }

            // 1. Check Water + Lava contact with any of the 6 orthogonal neighbors.
            if cell.kind == WATER || cell.kind == LAVA {
                let mut lava_turned = false;
                for d in ALL_DIRS {
                    let np = pos + d;
                    if crystallized.contains(&np) {
                        continue;
                    }
                    let nb = get_block(np);
                    if cell.kind == WATER && nb == LAVA {
                        let is_src = self.cells.get(&np).map(|c| c.is_source).unwrap_or(true);
                        let stone_id = if is_src { OBSIDIAN } else { COBBLESTONE };
                        crystallized.insert(np);
                        block_edits.push((np, stone_id));
                    } else if cell.kind == LAVA && nb == WATER {
                        let stone_id = if cell.is_source {
                            OBSIDIAN
                        } else {
                            COBBLESTONE
                        };
                        crystallized.insert(pos);
                        block_edits.push((pos, stone_id));
                        lava_turned = true;
                        break;
                    }
                }
                if lava_turned {
                    continue;
                }
            }

            // 2. Decay check for non-source cells whose feeder was removed.
            if !self.has_upstream_feeder(pos, cell) {
                if cell.level <= 1 {
                    dried_up.push(pos);
                } else {
                    next_cells.push((
                        pos,
                        FluidCell {
                            kind: cell.kind,
                            level: cell.level - 1,
                            is_source: false,
                        },
                    ));
                }
                continue;
            }

            // If receiving inflow from above (`pos + Y`), reset this cell's level to MAX_FLUID_LEVEL (8)
            // so when it lands on a ledge, it gets a fresh 7-block horizontal spread range!
            if let Some(above) = self.cells.get(&(pos + IVec3::Y)) {
                if above.kind == cell.kind && cell.level < MAX_FLUID_LEVEL {
                    cell.level = MAX_FLUID_LEVEL;
                    next_cells.push((pos, cell));
                }
            }

            // 3. Strict downward flow (`pos - Y`): resets level to `MAX_FLUID_LEVEL` (8) on drop!
            if pos.y > 1 {
                let below = pos - IVec3::Y;
                if !crystallized.contains(&below) {
                    let b_below = get_block(below);
                    if b_below == AIR || b_below == FIRE {
                        next_cells.push((
                            below,
                            FluidCell {
                                kind: cell.kind,
                                level: MAX_FLUID_LEVEL,
                                is_source: false,
                            },
                        ));
                        continue;
                    } else if b_below == cell.kind {
                        let below_lvl = self.cells.get(&below).map(|c| c.level).unwrap_or(0);
                        if below_lvl < MAX_FLUID_LEVEL {
                            next_cells.push((
                                below,
                                FluidCell {
                                    kind: cell.kind,
                                    level: MAX_FLUID_LEVEL,
                                    is_source: false,
                                },
                            ));
                        }
                        // Falling into a pool of the same fluid below: do not spread sideways mid-air.
                        continue;
                    } else if (cell.kind == WATER && b_below == LAVA)
                        || (cell.kind == LAVA && b_below == WATER)
                    {
                        let target_lava = if b_below == LAVA { below } else { pos };
                        let is_src = self
                            .cells
                            .get(&target_lava)
                            .map(|c| c.is_source)
                            .unwrap_or(true);
                        let stone_id = if is_src { OBSIDIAN } else { COBBLESTONE };
                        crystallized.insert(target_lava);
                        block_edits.push((target_lava, stone_id));
                        continue;
                    }
                }
            }

            // 4. Horizontal spreading (`+X, -X, +Z, -Z`) when supported by a solid block below.
            let below = pos - IVec3::Y;
            let b_below = get_block(below);
            if !is_solid_block(b_below) || cell.level <= 1 {
                continue;
            }

            let step_drop = match cell.kind {
                LAVA | OIL => 2,
                _ => 1,
            };
            if cell.level <= step_drop {
                continue;
            }
            let new_level = cell.level - step_drop;

            for d in HORIZONTAL_DIRS {
                let np = pos + d;
                if crystallized.contains(&np) {
                    continue;
                }
                let nb = get_block(np);
                if nb == AIR || nb == FIRE {
                    next_cells.push((
                        np,
                        FluidCell {
                            kind: cell.kind,
                            level: new_level,
                            is_source: false,
                        },
                    ));
                } else if nb == cell.kind {
                    let existing_lvl = self.cells.get(&np).map(|c| c.level).unwrap_or(0);
                    if existing_lvl < new_level {
                        let keep_src = self.cells.get(&np).map(|c| c.is_source).unwrap_or(false);
                        next_cells.push((
                            np,
                            FluidCell {
                                kind: cell.kind,
                                level: new_level,
                                is_source: keep_src,
                            },
                        ));
                    }
                } else if (cell.kind == WATER && nb == LAVA) || (cell.kind == LAVA && nb == WATER)
                {
                    let target_lava = if nb == LAVA { np } else { pos };
                    let is_src = self
                        .cells
                        .get(&target_lava)
                        .map(|c| c.is_source)
                        .unwrap_or(true);
                    let stone_id = if is_src { OBSIDIAN } else { COBBLESTONE };
                    crystallized.insert(target_lava);
                    block_edits.push((target_lava, stone_id));
                }
            }
        }

        for pos in crystallized {
            self.cells.remove(&pos);
            self.enqueue_with_neighbors(pos);
        }

        for pos in dried_up {
            self.cells.remove(&pos);
            block_edits.push((pos, AIR));
            self.enqueue_with_neighbors(pos);
        }

        for (pos, cell) in next_cells {
            let changed = match self.cells.get(&pos) {
                Some(prev) => prev.kind != cell.kind || prev.level != cell.level,
                None => true,
            };
            if changed {
                self.cells.insert(pos, cell);
                block_edits.push((pos, cell.kind));
                self.enqueue_with_neighbors(pos);
            }
        }

        block_edits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::chunk::STONE;

    fn step_fluid_once(sim: &mut FluidSimulator, world_map: &mut HashMap<IVec3, BlockId>) {
        for _ in 0..FLUID_TICK_INTERVAL {
            let edits = sim.tick(|p| *world_map.get(&p).unwrap_or(&AIR));
            for (p, id) in edits {
                if id == AIR {
                    world_map.remove(&p);
                } else {
                    world_map.insert(p, id);
                }
            }
        }
    }

    #[test]
    fn water_falls_over_ledge_resets_to_level_8_and_flows_another_7_blocks() {
        let mut world_map: HashMap<IVec3, BlockId> = HashMap::new();
        // Upper ledge at y = 3 for x in 0..=4, lower floor at y = 1 for x in 5..=15
        for x in 0..=4 {
            world_map.insert(IVec3::new(x, 3, 0), STONE);
        }
        for x in 5..=15 {
            world_map.insert(IVec3::new(x, 1, 0), STONE);
        }
        // Side walls so water flows along +X
        let mut sim = FluidSimulator::new();
        let src = IVec3::new(0, 4, 0);
        world_map.insert(src, WATER);
        sim.on_block_changed(src, WATER);

        for _ in 0..16 {
            step_fluid_once(&mut sim, &mut world_map);
        }

        // At x = 4, y = 4 (edge of upper ledge), level decayed from 8 to 4.
        assert_eq!(sim.level_at(IVec3::new(4, 4, 0)), Some(4));
        // At x = 5, y = 2 (dropped onto lower ledge at y = 1), level reset to 8!
        assert_eq!(sim.level_at(IVec3::new(5, 2, 0)), Some(8));
        // And it continues flowing on the lower ledge up to x = 12 (7 blocks past x = 5)!
        assert_eq!(sim.level_at(IVec3::new(12, 2, 0)), Some(1));

        // Check current flow vector at x = 2, y = 4 points along +X
        let flow = sim.flow_vector_at(IVec3::new(2, 4, 0));
        assert!(flow.x > 0.5);
    }

    #[test]
    fn water_touching_lava_source_creates_obsidian_and_flowing_lava_creates_cobblestone() {
        let mut world_map: HashMap<IVec3, BlockId> = HashMap::new();
        let mut sim = FluidSimulator::new();

        // Case 1: Water adjacent to Lava source (level 8) -> Obsidian
        let water_pos = IVec3::new(0, 5, 0);
        let lava_src = IVec3::new(1, 5, 0);
        world_map.insert(water_pos, WATER);
        world_map.insert(lava_src, LAVA);
        sim.on_block_changed(water_pos, WATER);
        sim.on_block_changed(lava_src, LAVA);

        // Case 2: Water adjacent to flowing Lava (level 4) -> Cobblestone
        let water_pos2 = IVec3::new(0, 8, 0);
        let lava_flow = IVec3::new(1, 8, 0);
        world_map.insert(water_pos2, WATER);
        world_map.insert(lava_flow, LAVA);
        sim.on_block_changed(water_pos2, WATER);
        sim.set_fluid_with_level(lava_flow, LAVA, 4);

        step_fluid_once(&mut sim, &mut world_map);

        assert_eq!(world_map.get(&lava_src), Some(&OBSIDIAN));
        assert_eq!(world_map.get(&lava_flow), Some(&COBBLESTONE));
    }
}
