// Binary world persistence (`serde` + `bincode`) under `saves/<world_name>/`:
//   • `level.dat`: `WorldMeta` (world name, seed, time_of_day, game mode, player pose & inventory)
//   • `chunks/c.{cx}.{cz}.bin`: `SavedChunk` (ONLY modified `is_dirty == true` chunks)
//   • `machines.dat`: `SavedMachines` (chests, energy network EU buffers, furnaces, active fluids)

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use glam::{IVec3, Vec3};
use serde::{Deserialize, Serialize};

use crate::inventory::{ItemStack, PlayerInventory, HOTBAR_SIZE, MAIN_INV_SIZE};
use crate::sim::energy_net::SavedEnergyState;
use crate::voxel::chunk::Chunk;
use crate::voxel::fluid::FluidCell;
use crate::voxel::world::World;

pub const SAVES_ROOT: &str = "saves";

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct WorldMeta {
    pub world_name: String,
    pub seed: u64,
    pub time_of_day: f32,
    pub game_mode_creative: bool,
    pub player_pos: [f32; 3],
    pub player_yaw: f32,
    pub player_pitch: f32,
    pub player_inventory: Vec<Option<ItemStack>>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SavedChunk {
    pub chunk_x: i32,
    pub chunk_z: i32,
    pub blocks: Vec<u16>,
    pub metadata: Vec<u8>,
}

#[derive(Serialize, Deserialize, Default)]
pub struct SavedMachines {
    pub energy: SavedEnergyState,
    pub fluids: Vec<([i32; 3], FluidCell)>,
}

#[derive(Clone, Debug)]
pub struct SavedWorldSummary {
    pub folder_name: String,
    pub world_name: String,
    pub seed: u64,
    pub game_mode_creative: bool,
    pub size_bytes: u64,
    pub last_saved_secs: u64,
}

/// Converts a user seed string into a `u64` (numeric parse or deterministic FNV-1a hash).
pub fn hash_seed_str(input: &str) -> u64 {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() ^ (d.subsec_nanos() as u64))
            .unwrap_or(42);
    }
    if let Ok(num) = trimmed.parse::<u64>() {
        return num;
    }
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in trimmed.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

pub fn sanitize_folder_name(name: &str) -> String {
    let cleaned: String = name
        .trim()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == ' ' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let trimmed = cleaned.trim();
    if trimmed.is_empty() {
        "New_World".to_string()
    } else {
        trimmed.to_string()
    }
}

pub fn world_dir(folder_name: &str) -> PathBuf {
    Path::new(SAVES_ROOT).join(folder_name)
}

fn dir_size_bytes(path: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    let mut total = 0u64;
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            total += dir_size_bytes(&p);
        } else if let Ok(meta) = entry.metadata() {
            total += meta.len();
        }
    }
    total
}

/// Scans `saves/` and returns summaries of all valid saved worlds sorted by most recently saved.
pub fn list_saved_worlds() -> Vec<SavedWorldSummary> {
    let root = Path::new(SAVES_ROOT);
    if !root.exists() {
        return Vec::new();
    }
    let Ok(entries) = fs::read_dir(root) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let level_path = path.join("level.dat");
        let Ok(bytes) = fs::read(&level_path) else {
            continue;
        };
        let Ok(meta) = bincode::deserialize::<WorldMeta>(&bytes) else {
            continue;
        };
        let folder_name = entry.file_name().to_string_lossy().to_string();
        let last_saved_secs = fs::metadata(&level_path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let size_bytes = dir_size_bytes(&path);
        out.push(SavedWorldSummary {
            folder_name,
            world_name: meta.world_name,
            seed: meta.seed,
            game_mode_creative: meta.game_mode_creative,
            size_bytes,
            last_saved_secs,
        });
    }
    out.sort_by(|a, b| b.last_saved_secs.cmp(&a.last_saved_secs));
    out
}

pub fn pack_inventory(inv: &PlayerInventory) -> Vec<Option<ItemStack>> {
    let mut out = Vec::with_capacity(HOTBAR_SIZE + MAIN_INV_SIZE);
    out.extend_from_slice(&inv.hotbar);
    out.extend_from_slice(&inv.main);
    out
}

pub fn unpack_inventory(data: &[Option<ItemStack>], inv: &mut PlayerInventory) {
    if data.len() < HOTBAR_SIZE {
        return;
    }
    for (i, slot) in inv.hotbar.iter_mut().enumerate() {
        *slot = data[i];
    }
    for (i, slot) in inv.main.iter_mut().enumerate() {
        *slot = data.get(HOTBAR_SIZE + i).copied().flatten();
    }
    inv.cursor_item = None;
}

/// Saves `level.dat`, dirty chunks (`chunks/c.{cx}.{cz}.bin`), and `machines.dat`.
pub fn save_world(
    folder_name: &str,
    meta: &WorldMeta,
    world: &World,
) -> Result<(), Box<dyn std::error::Error>> {
    let base = world_dir(folder_name);
    let chunks_dir = base.join("chunks");
    fs::create_dir_all(&chunks_dir)?;

    // 1. level.dat
    let meta_bytes = bincode::serialize(meta)?;
    fs::write(base.join("level.dat"), meta_bytes)?;

    // 2. Dirty chunks only
    let mut saved_chunks = 0usize;
    for (&coord, chunk) in world.chunks() {
        if !chunk.is_dirty {
            continue;
        }
        let sc = SavedChunk {
            chunk_x: coord.x,
            chunk_z: coord.z,
            blocks: chunk.blocks_slice().to_vec(),
            metadata: chunk.metadata_slice().to_vec(),
        };
        let chunk_path = chunks_dir.join(format!("c.{}.{}.bin", coord.x, coord.z));
        let bytes = bincode::serialize(&sc)?;
        fs::write(chunk_path, bytes)?;
        saved_chunks += 1;
    }

    // 3. machines.dat
    let machines = SavedMachines {
        energy: world.energy.export_state(),
        fluids: world.fluids.export_cells(),
    };
    let machines_bytes = bincode::serialize(&machines)?;
    fs::write(base.join("machines.dat"), machines_bytes)?;

    log::info!(
        "[Save] Saved world '{}' ({} dirty chunks) to {}",
        meta.world_name,
        saved_chunks,
        base.display()
    );
    Ok(())
}

pub fn load_world_meta(folder_name: &str) -> Result<WorldMeta, Box<dyn std::error::Error>> {
    let bytes = fs::read(world_dir(folder_name).join("level.dat"))?;
    let meta = bincode::deserialize::<WorldMeta>(&bytes)?;
    Ok(meta)
}

/// Applies all saved `c.{cx}.{cz}.bin` chunks and `machines.dat` onto a freshly seeded `World`.
pub fn load_world_overrides(
    folder_name: &str,
    device: &wgpu::Device,
    world: &mut World,
) -> Result<(), Box<dyn std::error::Error>> {
    let base = world_dir(folder_name);
    let chunks_dir = base.join("chunks");
    let mut loaded_chunks = 0usize;

    if chunks_dir.exists() {
        for entry in fs::read_dir(&chunks_dir)?.flatten() {
            let p = entry.path();
            if p.extension().and_then(|e| e.to_str()) != Some("bin") {
                continue;
            }
            let Ok(bytes) = fs::read(&p) else {
                continue;
            };
            let Ok(sc) = bincode::deserialize::<SavedChunk>(&bytes) else {
                continue;
            };
            if let Some(chunk) = Chunk::from_raw(sc.blocks, sc.metadata) {
                world.insert_loaded_chunk(IVec3::new(sc.chunk_x, 0, sc.chunk_z), chunk);
                loaded_chunks += 1;
            }
        }
    }

    let machines_path = base.join("machines.dat");
    if machines_path.exists() {
        if let Ok(bytes) = fs::read(&machines_path) {
            if let Ok(machines) = bincode::deserialize::<SavedMachines>(&bytes) {
                world.energy.import_state(machines.energy);
                world.fluids.import_cells(machines.fluids);
            }
        }
    }

    if loaded_chunks > 0 {
        world.remesh_all(device);
    }
    log::info!(
        "[Load] Restored {} modified chunks from {}",
        loaded_chunks,
        base.display()
    );
    Ok(())
}

pub fn delete_world(folder_name: &str) -> io::Result<()> {
    let dir = world_dir(folder_name);
    if dir.exists() {
        fs::remove_dir_all(dir)?;
    }
    Ok(())
}

pub fn default_world_meta(name: String, seed: u64, creative: bool, spawn: Vec3, inv: &PlayerInventory) -> WorldMeta {
    WorldMeta {
        world_name: name,
        seed,
        time_of_day: 0.25,
        game_mode_creative: creative,
        player_pos: spawn.to_array(),
        player_yaw: 0.785,
        player_pitch: -0.25,
        player_inventory: pack_inventory(inv),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::chunk::{GRANITE, STONE};

    #[test]
    fn bincode_roundtrip_world_meta_and_saved_chunk() {
        let inv = PlayerInventory::new();
        let meta = WorldMeta {
            world_name: "Test Industrial".to_string(),
            seed: hash_seed_str("industrial-42"),
            time_of_day: 0.35,
            game_mode_creative: true,
            player_pos: [12.5, 22.0, -4.0],
            player_yaw: 1.2,
            player_pitch: -0.3,
            player_inventory: pack_inventory(&inv),
        };
        let bytes = bincode::serialize(&meta).expect("serialize meta");
        let decoded: WorldMeta = bincode::deserialize(&bytes).expect("deserialize meta");
        assert_eq!(decoded.world_name, "Test Industrial");
        assert_eq!(decoded.seed, hash_seed_str("industrial-42"));
        assert!(decoded.game_mode_creative);

        let mut c = Chunk::new();
        c.set_block(3, 10, 5, GRANITE);
        c.set_block(4, 10, 5, STONE);
        c.set_meta(3, 10, 5, 2);
        let sc = SavedChunk {
            chunk_x: -1,
            chunk_z: 2,
            blocks: c.blocks_slice().to_vec(),
            metadata: c.metadata_slice().to_vec(),
        };
        let c_bytes = bincode::serialize(&sc).expect("serialize chunk");
        let sc2: SavedChunk = bincode::deserialize(&c_bytes).expect("deserialize chunk");
        let restored = Chunk::from_raw(sc2.blocks, sc2.metadata).expect("restore chunk");
        assert_eq!(restored.get_block(3, 10, 5), GRANITE);
        assert_eq!(restored.get_meta(3, 10, 5), 2);
    }
}
