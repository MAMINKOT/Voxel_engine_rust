// Industrial energy grid & block-entity simulation running at a fixed 20 TPS.
//
// Manages:
//   • Graph-based electrical network (`Cable`, `CoalGenerator`, `ElectricFurnace`,
//     `AlloySmelter`, `BatBox`).
//   • Standalone block entities (`Chest` with 27 slots, `SimpleFurnace` burning coal).
//   • Active visual state (`is_active`) per machine/furnace so `World` re-meshes chunks
//     when a furnace or generator lights up or turns off.

use std::collections::{HashMap, HashSet, VecDeque};

use glam::IVec3;
use serde::{Deserialize, Serialize};

use crate::crafting::RecipeRegistry;
use crate::inventory::{ItemStack, MAIN_INV_SIZE};
use crate::voxel::chunk::{
    BlockId, ALLOY_SMELTER, BATBOX, CABLE, CHEST, COAL_GENERATOR, COAL_ORE, ELECTRIC_FURNACE,
    SIMPLE_FURNACE, WORKBENCH,
};

pub const TICKS_PER_SECOND: u32 = 20;
/// Duration of one logic tick in seconds.
pub const TICK_DT: f32 = 1.0 / TICKS_PER_SECOND as f32;

// ─── Machine constants ───────────────────────────────────────────────────────

pub const GEN_MAX_EU: u32 = 4000;
pub const GEN_GENERATE_EU_PER_TICK: u32 = 10;
pub const GEN_MAX_OUT_EU_PER_TICK: u32 = 32;
pub const COAL_BURN_TICKS: u32 = 1600;

pub const FURNACE_MAX_EU: u32 = 1200;
pub const FURNACE_MAX_IN_EU_PER_TICK: u32 = 32;
pub const FURNACE_RECIPE_EU_PER_TICK: u32 = 3;

pub const ALLOY_MAX_EU: u32 = 2400;
pub const ALLOY_MAX_IN_EU_PER_TICK: u32 = 64;
pub const ALLOY_RECIPE_EU_PER_TICK: u32 = 12;

pub const BATBOX_MAX_EU: u32 = 40_000;
pub const BATBOX_MAX_IO_EU_PER_TICK: u32 = 32;

const MAX_ITEMS: u16 = 64;

const NEIGHBORS: [IVec3; 6] = [
    IVec3::new(1, 0, 0),
    IVec3::new(-1, 0, 0),
    IVec3::new(0, 1, 0),
    IVec3::new(0, -1, 0),
    IVec3::new(0, 0, 1),
    IVec3::new(0, 0, -1),
];

// ─── Machines & Block Entities ───────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
pub struct Generator {
    pub fuel: u16,
    pub burn_ticks: u32,
    pub stored_eu: u32,
    pub max_eu: u32,
    pub is_active: bool,
}

impl Default for Generator {
    fn default() -> Self {
        Self {
            fuel: 0,
            burn_ticks: 0,
            stored_eu: 0,
            max_eu: GEN_MAX_EU,
            is_active: false,
        }
    }
}

impl Generator {
    fn step_burn(&mut self) {
        if self.burn_ticks == 0 && self.stored_eu < self.max_eu && self.fuel > 0 {
            self.fuel -= 1;
            self.burn_ticks = COAL_BURN_TICKS;
        }
        if self.burn_ticks > 0 {
            self.burn_ticks -= 1;
            self.stored_eu = (self.stored_eu + GEN_GENERATE_EU_PER_TICK).min(self.max_eu);
        }
        self.is_active = self.burn_ticks > 0;
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Furnace {
    pub input: Option<ItemStack>,
    pub output: Option<ItemStack>,
    pub cook_progress: u32,
    pub cook_total: u32,
    pub stored_eu: u32,
    pub max_eu: u32,
    pub is_active: bool,
}

impl Default for Furnace {
    fn default() -> Self {
        Self {
            input: None,
            output: None,
            cook_progress: 0,
            cook_total: 100,
            stored_eu: 0,
            max_eu: FURNACE_MAX_EU,
            is_active: false,
        }
    }
}

impl Furnace {
    fn step_smelt(&mut self, recipes: &RecipeRegistry) {
        let Some(in_stack) = self.input else {
            self.cook_progress = 0;
            self.is_active = false;
            return;
        };
        let Some((recipe_out, cook_ticks)) = recipes.match_smelting(in_stack.id) else {
            self.cook_progress = 0;
            self.is_active = false;
            return;
        };
        let can_fit = match self.output {
            None => true,
            Some(out) => out.id == recipe_out.id && out.count + recipe_out.count <= MAX_ITEMS,
        };
        if !can_fit {
            self.cook_progress = 0;
            self.is_active = false;
            return;
        }
        self.cook_total = cook_ticks as u32;

        if self.stored_eu >= FURNACE_RECIPE_EU_PER_TICK {
            self.stored_eu -= FURNACE_RECIPE_EU_PER_TICK;
            self.cook_progress += 1;
            self.is_active = true;
            if self.cook_progress >= self.cook_total {
                let mut inp = in_stack;
                inp.count -= 1;
                self.input = (inp.count > 0).then_some(inp);
                match &mut self.output {
                    Some(out) => out.count += recipe_out.count,
                    None => self.output = Some(recipe_out),
                }
                self.cook_progress = 0;
            }
        } else {
            self.is_active = false;
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AlloySmelter {
    pub input_a: Option<ItemStack>,
    pub input_b: Option<ItemStack>,
    pub output: Option<ItemStack>,
    pub cook_progress: u32,
    pub cook_total: u32,
    pub stored_eu: u32,
    pub max_eu: u32,
    pub is_active: bool,
}

impl Default for AlloySmelter {
    fn default() -> Self {
        Self {
            input_a: None,
            input_b: None,
            output: None,
            cook_progress: 0,
            cook_total: 200,
            stored_eu: 0,
            max_eu: ALLOY_MAX_EU,
            is_active: false,
        }
    }
}

impl AlloySmelter {
    fn step_smelt(&mut self, recipes: &RecipeRegistry) {
        let Some((use_a, use_b, recipe_out, cook_ticks)) =
            recipes.match_alloy(self.input_a, self.input_b)
        else {
            self.cook_progress = 0;
            self.is_active = false;
            return;
        };
        let can_fit = match self.output {
            None => true,
            Some(out) => out.id == recipe_out.id && out.count + recipe_out.count <= MAX_ITEMS,
        };
        if !can_fit {
            self.cook_progress = 0;
            self.is_active = false;
            return;
        }
        self.cook_total = cook_ticks as u32;

        if self.stored_eu >= ALLOY_RECIPE_EU_PER_TICK {
            self.stored_eu -= ALLOY_RECIPE_EU_PER_TICK;
            self.cook_progress += 1;
            self.is_active = true;
            if self.cook_progress >= self.cook_total {
                if let Some(mut a) = self.input_a {
                    a.count = a.count.saturating_sub(use_a);
                    self.input_a = (a.count > 0).then_some(a);
                }
                if let Some(mut b) = self.input_b {
                    b.count = b.count.saturating_sub(use_b);
                    self.input_b = (b.count > 0).then_some(b);
                }
                match &mut self.output {
                    Some(out) => out.count += recipe_out.count,
                    None => self.output = Some(recipe_out),
                }
                self.cook_progress = 0;
            }
        } else {
            self.is_active = false;
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct BatBox {
    pub stored_eu: u32,
    pub max_eu: u32,
    pub charge_slot: Option<ItemStack>,
    pub is_active: bool,
}

impl Default for BatBox {
    fn default() -> Self {
        Self {
            stored_eu: 0,
            max_eu: BATBOX_MAX_EU,
            charge_slot: None,
            is_active: false,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SimpleFurnace {
    pub input: Option<ItemStack>,
    pub fuel: u16,
    pub output: Option<ItemStack>,
    pub burn_ticks: u32,
    pub cook_progress: u32,
    pub cook_total: u32,
    pub is_active: bool,
}

impl Default for SimpleFurnace {
    fn default() -> Self {
        Self {
            input: None,
            fuel: 0,
            output: None,
            burn_ticks: 0,
            cook_progress: 0,
            cook_total: 100,
            is_active: false,
        }
    }
}

impl SimpleFurnace {
    fn step(&mut self, recipes: &RecipeRegistry) {
        let valid_recipe = self.input.and_then(|inp| {
            let (out, ticks) = recipes.match_smelting(inp.id)?;
            let fits = match self.output {
                None => true,
                Some(cur) => cur.id == out.id && cur.count + out.count <= MAX_ITEMS,
            };
            fits.then_some((out, ticks as u32))
        });

        if valid_recipe.is_some() && self.burn_ticks == 0 && self.fuel > 0 {
            self.fuel -= 1;
            self.burn_ticks = COAL_BURN_TICKS;
        }

        if self.burn_ticks > 0 {
            self.burn_ticks -= 1;
            if let Some((recipe_out, cook_ticks)) = valid_recipe {
                self.cook_total = cook_ticks;
                self.cook_progress += 1;
                if self.cook_progress >= self.cook_total {
                    if let Some(mut inp) = self.input {
                        inp.count -= 1;
                        self.input = (inp.count > 0).then_some(inp);
                    }
                    match &mut self.output {
                        Some(out) => out.count += recipe_out.count,
                        None => self.output = Some(recipe_out),
                    }
                    self.cook_progress = 0;
                }
            } else {
                self.cook_progress = 0;
            }
        } else {
            self.cook_progress = 0;
        }

        self.is_active = self.burn_ticks > 0;
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Chest {
    pub slots: [Option<ItemStack>; MAIN_INV_SIZE],
}

impl Default for Chest {
    fn default() -> Self {
        Self {
            slots: [None; MAIN_INV_SIZE],
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
enum Machine {
    Generator(Generator),
    Furnace(Furnace),
    AlloySmelter(AlloySmelter),
    BatBox(BatBox),
}

#[derive(Default, Serialize, Deserialize)]
pub struct SavedEnergyState {
    machines: Vec<([i32; 3], Machine)>,
    simple_furnaces: Vec<([i32; 3], SimpleFurnace)>,
    chests: Vec<([i32; 3], Vec<Option<ItemStack>>)>,
    workbenches: Vec<[i32; 3]>,
    cables: Vec<[i32; 3]>,
}

impl Machine {
    fn is_active(&self) -> bool {
        match self {
            Machine::Generator(g) => g.is_active,
            Machine::Furnace(f) => f.is_active,
            Machine::AlloySmelter(a) => a.is_active,
            Machine::BatBox(b) => b.is_active,
        }
    }
}

// ─── Graph ───────────────────────────────────────────────────────────────────

#[derive(Default)]
struct Network {
    generators: Vec<IVec3>,
    batboxes: Vec<IVec3>,
    consumers: Vec<IVec3>,
    cables: usize,
}

pub struct EnergyNetwork {
    machines: HashMap<IVec3, Machine>,
    simple_furnaces: HashMap<IVec3, SimpleFurnace>,
    chests: HashMap<IVec3, Chest>,
    workbenches: HashSet<IVec3>,
    cables: HashSet<IVec3>,
    networks: Vec<Network>,
    net_of: HashMap<IVec3, usize>,
    dirty: bool,
    pub recipes: RecipeRegistry,
}

fn is_grid_block(id: BlockId) -> bool {
    matches!(
        id,
        CABLE | COAL_GENERATOR | ELECTRIC_FURNACE | ALLOY_SMELTER | BATBOX
    )
}

impl EnergyNetwork {
    pub fn new() -> Self {
        Self {
            machines: HashMap::new(),
            simple_furnaces: HashMap::new(),
            chests: HashMap::new(),
            workbenches: HashSet::new(),
            cables: HashSet::new(),
            networks: Vec::new(),
            net_of: HashMap::new(),
            dirty: false,
            recipes: RecipeRegistry::new(),
        }
    }

    pub fn export_state(&self) -> SavedEnergyState {
        SavedEnergyState {
            machines: self
                .machines
                .iter()
                .map(|(p, m)| (p.to_array(), m.clone()))
                .collect(),
            simple_furnaces: self
                .simple_furnaces
                .iter()
                .map(|(p, f)| (p.to_array(), f.clone()))
                .collect(),
            chests: self
                .chests
                .iter()
                .map(|(p, c)| (p.to_array(), c.slots.to_vec()))
                .collect(),
            workbenches: self.workbenches.iter().map(|p| p.to_array()).collect(),
            cables: self.cables.iter().map(|p| p.to_array()).collect(),
        }
    }

    pub fn import_state(&mut self, state: SavedEnergyState) {
        self.machines.clear();
        self.simple_furnaces.clear();
        self.chests.clear();
        self.workbenches.clear();
        self.cables.clear();
        for (p, m) in state.machines {
            self.machines.insert(IVec3::from_array(p), m);
        }
        for (p, f) in state.simple_furnaces {
            self.simple_furnaces.insert(IVec3::from_array(p), f);
        }
        for (p, slots_vec) in state.chests {
            let mut slots = [None; MAIN_INV_SIZE];
            for (i, s) in slots_vec.into_iter().take(MAIN_INV_SIZE).enumerate() {
                slots[i] = s;
            }
            self.chests.insert(IVec3::from_array(p), Chest { slots });
        }
        for p in state.workbenches {
            self.workbenches.insert(IVec3::from_array(p));
        }
        for p in state.cables {
            self.cables.insert(IVec3::from_array(p));
        }
        self.dirty = true;
    }

    /// Must be called for every block change in the world.
    pub fn on_block_changed(&mut self, pos: IVec3, old: BlockId, new: BlockId) {
        match old {
            CABLE => {
                self.cables.remove(&pos);
            }
            COAL_GENERATOR | ELECTRIC_FURNACE | ALLOY_SMELTER | BATBOX => {
                self.machines.remove(&pos);
            }
            SIMPLE_FURNACE => {
                self.simple_furnaces.remove(&pos);
            }
            CHEST => {
                self.chests.remove(&pos);
            }
            WORKBENCH => {
                self.workbenches.remove(&pos);
            }
            _ => {}
        }
        match new {
            CABLE => {
                self.cables.insert(pos);
            }
            COAL_GENERATOR => {
                self.machines.insert(pos, Machine::Generator(Generator::default()));
            }
            ELECTRIC_FURNACE => {
                self.machines.insert(pos, Machine::Furnace(Furnace::default()));
            }
            ALLOY_SMELTER => {
                self.machines
                    .insert(pos, Machine::AlloySmelter(AlloySmelter::default()));
            }
            BATBOX => {
                self.machines.insert(pos, Machine::BatBox(BatBox::default()));
            }
            SIMPLE_FURNACE => {
                self.simple_furnaces.insert(pos, SimpleFurnace::default());
            }
            CHEST => {
                self.chests.insert(pos, Chest::default());
            }
            WORKBENCH => {
                self.workbenches.insert(pos);
            }
            _ => {}
        }
        if is_grid_block(old) || is_grid_block(new) {
            self.dirty = true;
        }
    }

    fn is_node(&self, p: &IVec3) -> bool {
        self.cables.contains(p) || self.machines.contains_key(p)
    }

    /// Flood fill (BFS) over all cables and machines, producing connected networks.
    fn rebuild(&mut self) {
        let mut nodes: Vec<IVec3> = self
            .cables
            .iter()
            .chain(self.machines.keys())
            .copied()
            .collect();
        nodes.sort_by_key(|p| (p.x, p.y, p.z));

        let mut networks: Vec<Network> = Vec::new();
        let mut net_of: HashMap<IVec3, usize> = HashMap::new();

        for &start in &nodes {
            if net_of.contains_key(&start) {
                continue;
            }
            let index = networks.len();
            let mut net = Network::default();
            let mut queue = VecDeque::from([start]);
            net_of.insert(start, index);

            while let Some(p) = queue.pop_front() {
                if self.cables.contains(&p) {
                    net.cables += 1;
                } else {
                    match self.machines.get(&p) {
                        Some(Machine::Generator(_)) => net.generators.push(p),
                        Some(Machine::BatBox(_)) => net.batboxes.push(p),
                        Some(Machine::Furnace(_) | Machine::AlloySmelter(_)) => {
                            net.consumers.push(p);
                        }
                        None => {}
                    }
                }
                for offset in NEIGHBORS {
                    let q = p + offset;
                    if self.is_node(&q) && !net_of.contains_key(&q) {
                        net_of.insert(q, index);
                        queue.push_back(q);
                    }
                }
            }
            networks.push(net);
        }

        log::info!(
            "Energy graph rebuilt: {} network(s), {} cable(s), {} machine(s)",
            networks.len(),
            self.cables.len(),
            self.machines.len()
        );
        self.networks = networks;
        self.net_of = net_of;
        self.dirty = false;
    }

    /// One logic tick (20 TPS). Returns the coordinates of any blocks whose `is_active`
    /// visual state changed this tick so the caller can re-mesh their chunks.
    pub fn tick(&mut self) -> Vec<IVec3> {
        if self.dirty {
            self.rebuild();
        }

        let mut prev_active: HashMap<IVec3, bool> = self
            .machines
            .iter()
            .map(|(&p, m)| (p, m.is_active()))
            .collect();
        for (&p, sf) in &self.simple_furnaces {
            prev_active.insert(p, sf.is_active);
        }

        let networks = std::mem::take(&mut self.networks);
        for net in &networks {
            Self::tick_network(&mut self.machines, net, &self.recipes);
        }
        self.networks = networks;

        for sf in self.simple_furnaces.values_mut() {
            sf.step(&self.recipes);
        }

        let mut changed = Vec::new();
        for (&pos, machine) in &self.machines {
            let was = prev_active.get(&pos).copied().unwrap_or(false);
            if machine.is_active() != was {
                changed.push(pos);
            }
        }
        for (&pos, sf) in &self.simple_furnaces {
            let was = prev_active.get(&pos).copied().unwrap_or(false);
            if sf.is_active != was {
                changed.push(pos);
            }
        }
        changed
    }

    fn tick_network(
        machines: &mut HashMap<IVec3, Machine>,
        net: &Network,
        recipes: &RecipeRegistry,
    ) {
        // 1. Step all generators (burn coal -> fill internal `stored_eu` up to 4000 EU).
        let mut gen_push_budget: Vec<(IVec3, u32)> = Vec::with_capacity(net.generators.len());
        for &p in &net.generators {
            if let Some(Machine::Generator(g)) = machines.get_mut(&p) {
                g.step_burn();
                let budget = g.stored_eu.min(GEN_MAX_OUT_EU_PER_TICK);
                if budget > 0 {
                    gen_push_budget.push((p, budget));
                }
            }
        }

        // 2. Consumers pull energy into their internal `stored_eu` buffers:
        //    first from generators, then any deficit from BatBoxes.
        for &c_pos in &net.consumers {
            let want = match machines.get(&c_pos) {
                Some(Machine::Furnace(f)) => {
                    (f.max_eu.saturating_sub(f.stored_eu)).min(FURNACE_MAX_IN_EU_PER_TICK)
                }
                Some(Machine::AlloySmelter(a)) => {
                    (a.max_eu.saturating_sub(a.stored_eu)).min(ALLOY_MAX_IN_EU_PER_TICK)
                }
                _ => 0,
            };
            if want == 0 {
                continue;
            }

            let mut needed = want;
            for (g_pos, budget) in &mut gen_push_budget {
                if needed == 0 {
                    break;
                }
                let take = (*budget).min(needed);
                if take > 0 {
                    if let Some(Machine::Generator(g)) = machines.get_mut(g_pos) {
                        g.stored_eu -= take;
                        *budget -= take;
                        needed -= take;
                    }
                }
            }

            if needed > 0 {
                for &b_pos in &net.batboxes {
                    if needed == 0 {
                        break;
                    }
                    if let Some(Machine::BatBox(b)) = machines.get_mut(&b_pos) {
                        let avail = b.stored_eu.min(BATBOX_MAX_IO_EU_PER_TICK);
                        let take = avail.min(needed);
                        if take > 0 {
                            b.stored_eu -= take;
                            needed -= take;
                        }
                    }
                }
            }

            let received = want - needed;
            if received > 0 {
                match machines.get_mut(&c_pos) {
                    Some(Machine::Furnace(f)) => f.stored_eu += received,
                    Some(Machine::AlloySmelter(a)) => a.stored_eu += received,
                    _ => {}
                }
            }
        }

        // 3. Surplus generator output charges BatBoxes (up to 32 EU/t per BatBox).
        for &b_pos in &net.batboxes {
            let cap = match machines.get(&b_pos) {
                Some(Machine::BatBox(b)) => {
                    (b.max_eu.saturating_sub(b.stored_eu)).min(BATBOX_MAX_IO_EU_PER_TICK)
                }
                _ => 0,
            };
            if cap == 0 {
                continue;
            }
            let mut room = cap;
            for (g_pos, budget) in &mut gen_push_budget {
                if room == 0 {
                    break;
                }
                let take = (*budget).min(room);
                if take > 0 {
                    if let Some(Machine::Generator(g)) = machines.get_mut(g_pos) {
                        g.stored_eu -= take;
                        *budget -= take;
                        room -= take;
                    }
                }
            }
            let added = cap - room;
            if added > 0 {
                if let Some(Machine::BatBox(b)) = machines.get_mut(&b_pos) {
                    b.stored_eu += added;
                }
            }
        }

        for &b_pos in &net.batboxes {
            if let Some(Machine::BatBox(b)) = machines.get_mut(&b_pos) {
                b.is_active = b.stored_eu > 0;
            }
        }

        // 4. Step smelting processes using each consumer's internal `stored_eu`.
        for &c_pos in &net.consumers {
            match machines.get_mut(&c_pos) {
                Some(Machine::Furnace(f)) => f.step_smelt(recipes),
                Some(Machine::AlloySmelter(a)) => a.step_smelt(recipes),
                _ => {}
            }
        }
    }

    // ── Queries & UI accessors ────────────────────────────────────────────

    /// True if right-clicking `pos` should open a block UI (machine, chest, workbench, furnace).
    pub fn is_interactive(&self, pos: IVec3) -> bool {
        self.machines.contains_key(&pos)
            || self.simple_furnaces.contains_key(&pos)
            || self.chests.contains_key(&pos)
            || self.workbenches.contains(&pos)
    }

    pub fn is_machine_active(&self, pos: IVec3) -> bool {
        if let Some(m) = self.machines.get(&pos) {
            return m.is_active();
        }
        if let Some(sf) = self.simple_furnaces.get(&pos) {
            return sf.is_active;
        }
        false
    }

    /// Items stored inside a machine/chest (dropped into the world when the block is broken).
    pub fn contents(&self, pos: IVec3) -> Vec<(BlockId, u16)> {
        let mut items: Vec<(BlockId, u16)> = Vec::new();
        if let Some(m) = self.machines.get(&pos) {
            match m {
                Machine::Generator(g) => items.push((COAL_ORE, g.fuel)),
                Machine::Furnace(f) => {
                    if let Some(s) = f.input {
                        items.push((s.id, s.count));
                    }
                    if let Some(s) = f.output {
                        items.push((s.id, s.count));
                    }
                }
                Machine::AlloySmelter(a) => {
                    if let Some(s) = a.input_a {
                        items.push((s.id, s.count));
                    }
                    if let Some(s) = a.input_b {
                        items.push((s.id, s.count));
                    }
                    if let Some(s) = a.output {
                        items.push((s.id, s.count));
                    }
                }
                Machine::BatBox(b) => {
                    if let Some(s) = b.charge_slot {
                        items.push((s.id, s.count));
                    }
                }
            }
        } else if let Some(sf) = self.simple_furnaces.get(&pos) {
            if let Some(s) = sf.input {
                items.push((s.id, s.count));
            }
            items.push((COAL_ORE, sf.fuel));
            if let Some(s) = sf.output {
                items.push((s.id, s.count));
            }
        } else if let Some(chest) = self.chests.get(&pos) {
            for s in chest.slots.iter().flatten() {
                items.push((s.id, s.count));
            }
        }
        items.into_iter().filter(|&(_, n)| n > 0).collect()
    }

    /// Returns `(fuel_stack, burn_ticks, max_burn, stored_eu, max_eu)`.
    pub fn generator_ui_data(
        &self,
        pos: IVec3,
    ) -> Option<(Option<ItemStack>, u32, u32, u32, u32)> {
        match self.machines.get(&pos)? {
            Machine::Generator(g) => {
                let fuel = (g.fuel > 0).then_some(ItemStack { id: COAL_ORE, count: g.fuel });
                Some((fuel, g.burn_ticks, COAL_BURN_TICKS, g.stored_eu, g.max_eu))
            }
            _ => None,
        }
    }

    pub fn set_generator_fuel(&mut self, pos: IVec3, fuel: Option<ItemStack>) {
        if let Some(Machine::Generator(g)) = self.machines.get_mut(&pos) {
            g.fuel = fuel.filter(|s| s.id == COAL_ORE).map_or(0, |s| s.count);
        }
    }

    /// Returns `(input_stack, output_stack, cook_progress, cook_total, stored_eu, max_eu)`.
    pub fn furnace_ui_data(
        &self,
        pos: IVec3,
    ) -> Option<(Option<ItemStack>, Option<ItemStack>, u32, u32, u32, u32)> {
        match self.machines.get(&pos)? {
            Machine::Furnace(f) => Some((
                f.input,
                f.output,
                f.cook_progress,
                f.cook_total.max(1),
                f.stored_eu,
                f.max_eu,
            )),
            _ => None,
        }
    }

    pub fn set_furnace_slots(
        &mut self,
        pos: IVec3,
        input: Option<ItemStack>,
        output: Option<ItemStack>,
    ) {
        if let Some(Machine::Furnace(f)) = self.machines.get_mut(&pos) {
            if f.input.map(|s| s.id) != input.map(|s| s.id) || input.is_none() {
                f.cook_progress = 0;
            }
            f.input = input;
            f.output = output;
        }
    }

    /// Returns `(input_a, input_b, output_stack, cook_progress, cook_total, stored_eu, max_eu)`.
    pub fn alloy_ui_data(
        &self,
        pos: IVec3,
    ) -> Option<(
        Option<ItemStack>,
        Option<ItemStack>,
        Option<ItemStack>,
        u32,
        u32,
        u32,
        u32,
    )> {
        match self.machines.get(&pos)? {
            Machine::AlloySmelter(a) => Some((
                a.input_a,
                a.input_b,
                a.output,
                a.cook_progress,
                a.cook_total.max(1),
                a.stored_eu,
                a.max_eu,
            )),
            _ => None,
        }
    }

    pub fn set_alloy_slots(
        &mut self,
        pos: IVec3,
        input_a: Option<ItemStack>,
        input_b: Option<ItemStack>,
        output: Option<ItemStack>,
    ) {
        let valid = self.recipes.match_alloy(input_a, input_b).is_some();
        if let Some(Machine::AlloySmelter(a)) = self.machines.get_mut(&pos) {
            a.input_a = input_a;
            a.input_b = input_b;
            a.output = output;
            if !valid {
                a.cook_progress = 0;
            }
        }
    }

    /// Returns `(charge_slot, stored_eu, max_eu)`.
    pub fn batbox_ui_data(&self, pos: IVec3) -> Option<(Option<ItemStack>, u32, u32)> {
        match self.machines.get(&pos)? {
            Machine::BatBox(b) => Some((b.charge_slot, b.stored_eu, b.max_eu)),
            _ => None,
        }
    }

    pub fn set_batbox_slot(&mut self, pos: IVec3, slot: Option<ItemStack>) {
        if let Some(Machine::BatBox(b)) = self.machines.get_mut(&pos) {
            b.charge_slot = slot;
        }
    }

    /// Returns `(input, fuel_stack, output, burn_ticks, max_burn, cook_progress, cook_total)`.
    pub fn simple_furnace_ui_data(
        &self,
        pos: IVec3,
    ) -> Option<(
        Option<ItemStack>,
        Option<ItemStack>,
        Option<ItemStack>,
        u32,
        u32,
        u32,
        u32,
    )> {
        let sf = self.simple_furnaces.get(&pos)?;
        let fuel = (sf.fuel > 0).then_some(ItemStack { id: COAL_ORE, count: sf.fuel });
        Some((
            sf.input,
            fuel,
            sf.output,
            sf.burn_ticks,
            COAL_BURN_TICKS,
            sf.cook_progress,
            sf.cook_total.max(1),
        ))
    }

    pub fn set_simple_furnace_slots(
        &mut self,
        pos: IVec3,
        input: Option<ItemStack>,
        fuel: Option<ItemStack>,
        output: Option<ItemStack>,
    ) {
        if let Some(sf) = self.simple_furnaces.get_mut(&pos) {
            if sf.input.map(|s| s.id) != input.map(|s| s.id) || input.is_none() {
                sf.cook_progress = 0;
            }
            sf.input = input;
            sf.fuel = fuel.filter(|s| s.id == COAL_ORE).map_or(0, |s| s.count);
            sf.output = output;
        }
    }

    pub fn chest_slots_mut(&mut self, pos: IVec3) -> Option<&mut [Option<ItemStack>; MAIN_INV_SIZE]> {
        self.chests.get_mut(&pos).map(|c| &mut c.slots)
    }

    /// Returns `(energy_nets_count, machines_count, cables_count)` for the F3 debug screen.
    pub fn network_stats(&self) -> (usize, usize, usize) {
        (
            self.networks.len(),
            self.machines.len() + self.simple_furnaces.len(),
            self.cables.len(),
        )
    }

    /// Human-readable status for the window title bar / console log.
    pub fn status(&self, pos: IVec3) -> Option<String> {
        if let Some(machine) = self.machines.get(&pos) {
            return Some(match machine {
                Machine::Generator(g) => format!(
                    "Coal Generator: {}/{} EU | burn: {}t | fuel: {}",
                    g.stored_eu, g.max_eu, g.burn_ticks, g.fuel
                ),
                Machine::Furnace(f) => format!(
                    "Electric Furnace: {}/{} EU | smelt: {}/{}",
                    f.stored_eu, f.max_eu, f.cook_progress, f.cook_total
                ),
                Machine::AlloySmelter(a) => format!(
                    "Alloy Smelter: {}/{} EU | alloy: {}/{}",
                    a.stored_eu, a.max_eu, a.cook_progress, a.cook_total
                ),
                Machine::BatBox(b) => format!("BatBox: {}/{} EU", b.stored_eu, b.max_eu),
            });
        }
        if let Some(sf) = self.simple_furnaces.get(&pos) {
            return Some(format!(
                "Stone Furnace: burn {}t | smelt: {}/{} | fuel: {}",
                sf.burn_ticks, sf.cook_progress, sf.cook_total, sf.fuel
            ));
        }
        if let Some(chest) = self.chests.get(&pos) {
            let used = chest.slots.iter().filter(|s| s.is_some()).count();
            return Some(format!("Storage Chest: {}/{} slots used", used, MAIN_INV_SIZE));
        }
        if self.workbenches.contains(&pos) {
            return Some("Crafting Workbench (3x3)".to_string());
        }
        None
    }

    #[cfg(test)]
    pub fn insert_item(&mut self, pos: IVec3, item: BlockId) -> bool {
        match self.machines.get_mut(&pos) {
            Some(Machine::Generator(g)) if item == COAL_ORE && g.fuel < MAX_ITEMS => {
                g.fuel += 1;
                true
            }
            Some(Machine::Furnace(f)) => {
                match &mut f.input {
                    None => {
                        f.input = Some(ItemStack { id: item, count: 1 });
                        true
                    }
                    Some(s) if s.id == item && s.count < MAX_ITEMS => {
                        s.count += 1;
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    #[cfg(test)]
    pub fn take_output(&mut self, pos: IVec3) -> u16 {
        match self.machines.get_mut(&pos) {
            Some(Machine::Furnace(f)) => f.output.take().map_or(0, |s| s.count),
            Some(Machine::AlloySmelter(a)) => a.output.take().map_or(0, |s| s.count),
            _ => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::chunk::{COPPER_INGOT, COPPER_ORE, TIN_INGOT};

    fn place(net: &mut EnergyNetwork, p: IVec3, id: BlockId) {
        net.on_block_changed(p, 0, id);
    }

    fn chain() -> EnergyNetwork {
        let mut net = EnergyNetwork::new();
        place(&mut net, IVec3::new(0, 0, 0), COAL_GENERATOR);
        place(&mut net, IVec3::new(1, 0, 0), CABLE);
        place(&mut net, IVec3::new(2, 0, 0), ELECTRIC_FURNACE);
        net
    }

    #[test]
    fn furnace_smelts_through_cable() {
        let mut net = chain();
        assert!(net.insert_item(IVec3::new(0, 0, 0), COAL_ORE));
        assert!(net.insert_item(IVec3::new(2, 0, 0), COPPER_ORE));
        for _ in 0..105 {
            net.tick();
        }
        assert_eq!(net.networks.len(), 1);
        assert_eq!(net.take_output(IVec3::new(2, 0, 0)), 1);
    }

    #[test]
    fn disconnected_furnace_gets_no_power_and_cut_cable_splits_graph() {
        let mut net = chain();
        net.insert_item(IVec3::new(0, 0, 0), COAL_ORE);
        net.insert_item(IVec3::new(2, 0, 0), COPPER_ORE);
        net.on_block_changed(IVec3::new(1, 0, 0), CABLE, 0);
        for _ in 0..200 {
            net.tick();
        }
        assert_eq!(net.networks.len(), 2);
        assert_eq!(net.take_output(IVec3::new(2, 0, 0)), 0);
    }

    #[test]
    fn generator_fills_internal_buffer_and_charges_batbox() {
        let mut net = EnergyNetwork::new();
        place(&mut net, IVec3::new(0, 0, 0), COAL_GENERATOR);
        place(&mut net, IVec3::new(1, 0, 0), CABLE);
        place(&mut net, IVec3::new(2, 0, 0), BATBOX);
        net.insert_item(IVec3::new(0, 0, 0), COAL_ORE);

        for _ in 0..50 {
            net.tick();
        }
        let (_, stored, _) = net.batbox_ui_data(IVec3::new(2, 0, 0)).unwrap();
        assert_eq!(stored, 50 * GEN_GENERATE_EU_PER_TICK);
    }

    #[test]
    fn alloy_smelter_produces_four_bronze_ingots() {
        let mut net = EnergyNetwork::new();
        place(&mut net, IVec3::new(0, 0, 0), COAL_GENERATOR);
        place(&mut net, IVec3::new(1, 0, 0), COAL_GENERATOR);
        place(&mut net, IVec3::new(2, 0, 0), CABLE);
        place(&mut net, IVec3::new(3, 0, 0), ALLOY_SMELTER);
        net.insert_item(IVec3::new(0, 0, 0), COAL_ORE);
        net.insert_item(IVec3::new(1, 0, 0), COAL_ORE);
        net.set_alloy_slots(
            IVec3::new(3, 0, 0),
            Some(ItemStack { id: COPPER_INGOT, count: 3 }),
            Some(ItemStack { id: TIN_INGOT, count: 1 }),
            None,
        );

        for _ in 0..210 {
            net.tick();
        }
        assert_eq!(net.take_output(IVec3::new(3, 0, 0)), 4);
    }

    #[test]
    fn simple_stone_furnace_smelts_copper_on_coal() {
        let mut net = EnergyNetwork::new();
        let p = IVec3::new(5, 0, 5);
        place(&mut net, p, SIMPLE_FURNACE);
        net.set_simple_furnace_slots(
            p,
            Some(ItemStack { id: COPPER_ORE, count: 1 }),
            Some(ItemStack { id: COAL_ORE, count: 1 }),
            None,
        );
        for _ in 0..105 {
            net.tick();
        }
        let (_, _, out, _, _, _, _) = net.simple_furnace_ui_data(p).unwrap();
        assert_eq!(out, Some(ItemStack { id: COPPER_INGOT, count: 1 }));
    }
}
