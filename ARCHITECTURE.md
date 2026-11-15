# Voxel Industrial Engine — Architecture

> Data-Oriented Design | Rust 2021 | wgpu 0.19 | winit 0.29 | glam | egui 0.27 | rayon 1.10 | sysinfo 0.30

## Current state — Step 7.5: Sloped Translucent Water, Underwater Fog, Sun/Moon Sky Discs, Oak Canopy Trees, Drop Merging & RAM Monitor

| Step | Content | Status |
|---|---|---|
| 1 | wgpu window, depth buffer, FPS camera, FPS in title | done |
| 2 | 32³ chunks, greedy meshing, procedural terrain, sun lighting | done |
| 3 | AABB physics, walk/fly, DDA raycast, block highlight, break/place | done |
| 4 | 3D item drops + magnet pickup, hotbar, 20 TPS graph energy network | done |
| 5 | Internal EU buffers, `AlloySmelter` (Bronze), `BatBox` (40k EU), active 3D glow & full Machine UIs | done |
| 6 | `RecipeRegistry` (2×2/3×3 shaped & shapeless, smelting, alloying), `Chest`, `Workbench`, `SimpleFurnace`, NEI/JEI sidebar | done |
| 7 | Creative Fly AABB collisions, 16×16 Rayon parallel world, `F3` Debug HUD, Tin/Iron/Steel, Day/Night cycle, Fluids (Water/Lava/Oil), Lighter/Fire & TNT Explosions | done |
| 7.5 | Rate-limited `VecDeque` fluid levels (`height = level/8.0`), translucent water pass + underwater tint/fog, Sun & Moon sky discs + Moon night relief, `OakLeaves` tree domes, drop stack merging & `sysinfo` RAM monitor | done |

### Controls

| Input | Action |
|---|---|
| Click on window | capture mouse (mouse look); `Esc` closes open UI or releases mouse |
| `W A S D` | move |
| `Space` | jump (walking) / swim up (in fluids) / fly up (flying) |
| double-tap `Space` (< 250 ms) | toggle Walking ↔ Flying (with full AABB wall/floor collisions) |
| `LShift` | down (flying); **LShift + RMB** places a block/cable directly onto an interactive block face |
| `LCtrl` | run ×1.5 (walking) / turbo ×3 (flying) |
| `F3` | toggle top-left / top-right monospace diagnostic Debug Overlay |
| `T` | step Day/Night cycle forward by +150s (Sunrise → Day → Sunset → Night) |
| `1`–`9` or mouse wheel | switch active hotbar slot (0..8) |
| `E` / `I` / `Tab` | open/close full Player Inventory (`2×2` hand-crafting + `3×9` backpack + `1×9` hotbar + NEI sidebar) |
| LMB (in world) | break block (spawns spinning 3D item drop + drops machine/chest contents) |
| RMB (in world) | open interactive UI, ignite `TNT` / place `Fire` when holding `Lighter`, or place block/fluid |
| NEI Sidebar (in UI) | LMB on item: view Recipes; RMB on item: view Usages; MMB on item: give ×16 stack |

## Module structure

```
src/
├── main.rs              entry point, env_logger
├── camera.rs            Camera (view/proj), CameraUniform (view_proj + sun_dir + sun_color), CameraController
├── crafting/mod.rs      RecipeRegistry: CraftingShaped, CraftingShapeless, Smelting (Cu/Sn/Fe/Stone), Alloy (Bronze & Steel)
├── inventory.rs         PlayerInventory: 9-slot hotbar + 3×9 main + 2×2/3×3 crafting buffers + cursor stack
├── player.rs            Player: AABB physics for both Walking & Flying, fluid buoyancy/drag, explosion impulse knockback
├── engine/mod.rs        window, wgpu state, 20 TPS fixed accumulator, F3 & T keys, Lighter/TNT & UI modal routing
├── ui/
│   ├── mod.rs           UI module root
│   └── debug.rs         F3 Debug Overlay (FPS, frame time, XYZ, chunk, cardinal facing, raycast target, active chunks, fluids & nets)
├── render/
│   ├── pipeline.rs      world pipeline + WGSL (dynamic sun direction/color + night emissive glow), depth texture
│   ├── overlay.rs       wireframe AABB box (LineList) + crosshair pipelines
│   ├── sky.rs           SkyState: 1200s Day/Night cycle, sun orbit, sky clear color & sun/ambient interpolation
│   └── ui.rs            egui HUD hotbar, 2×2/3×3 crafting UI, Chest/Furnace/Generator/Alloy/BatBox UI & NEI browser
├── sim/
│   ├── energy_net.rs    EnergyNetwork: BFS graph solver, CoalGenerator, ElectricFurnace, SimpleFurnace, AlloySmelter, BatBox, Chest, Workbench
│   └── explosions.rs    ExplosionSystem: PrimedTnt (60t white-flashing fuse, R=4.5 crater, 40% drops, shockwave), Fire spread
└── voxel/
    ├── chunk.rs         Chunk: Box<[u16; 32768]>, block/item IDs (0..30) + ACTIVE_MACHINE_FLAG, sub-block & fluid helpers
    ├── drops.rs         ItemDrop physics, magnet pickup, 0.25³ spinning/levitating cube mesh
    ├── fluid.rs         FluidSimulator: 10 TPS cellular automaton for Water, Lava, Oil + Obsidian/Cobblestone crystallization
    ├── mesher.rs        Pass 1: greedy meshing for full cubes; Pass 2: dynamic 6-way cable sub-blocks
    ├── raycast.rs       DDA voxel traversal (skips fluids) + slab Ray-AABB sub-block intersection for cables
    └── world.rs         16×16 (256 chunks) Rayon parallel generation & meshing, block edits, fluids, explosions & energy net
```
