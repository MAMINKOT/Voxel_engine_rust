// Non-RTX RGB + Sunlight Voxel Lighting (`LightData (u16)`) with BFS Flood-Fill
// and Stained Glass spectral filtering.
//
// Bit layout of `LightData (u16)`:
//   • Bits 0..3   (4 bits): Sunlight (`0..15`)
//   • Bits 4..7   (4 bits): Red      (`0..15`)
//   • Bits 8..11  (4 bits): Green    (`0..15`)
//   • Bits 12..15 (4 bits): Blue     (`0..15`)

use std::collections::VecDeque;

use crate::voxel::chunk::{
    base_block_id, is_active_block, is_full_cube, BlockId, FIRE, LAVA, OAK_LEAVES,
    PURPLE_STAINED_GLASS, TORCH, WATER,
};

pub type LightData = u16;

#[inline]
pub const fn pack_light(sun: u8, r: u8, g: u8, b: u8) -> LightData {
    ((sun & 0xF) as u16)
        | (((r & 0xF) as u16) << 4)
        | (((g & 0xF) as u16) << 8)
        | (((b & 0xF) as u16) << 12)
}

#[inline]
pub const fn sun_channel(l: LightData) -> u8 {
    (l & 0xF) as u8
}

#[inline]
pub const fn red_channel(l: LightData) -> u8 {
    ((l >> 4) & 0xF) as u8
}

#[inline]
pub const fn green_channel(l: LightData) -> u8 {
    ((l >> 8) & 0xF) as u8
}

#[inline]
pub const fn blue_channel(l: LightData) -> u8 {
    ((l >> 12) & 0xF) as u8
}

/// Converts packed `LightData` into `[R, G, B, Sun]` floats (`0.0..=15.0`) for vertex attributes.
#[inline]
pub fn light_to_f32_array(l: LightData) -> [f32; 4] {
    [
        red_channel(l) as f32,
        green_channel(l) as f32,
        blue_channel(l) as f32,
        sun_channel(l) as f32,
    ]
}

/// Intrinsic RGB emission `(R, G, B)` for light-emitting blocks:
///   • `TORCH (29)`          -> warm yellow `(15, 12, 5)`
///   • `LAVA (22)` / `FIRE`  -> fiery orange-red `(15, 8, 2)`
///   • Active Machine        -> industrial amber `(14, 7, 1)`
#[inline]
pub fn block_emission(raw: BlockId) -> Option<(u8, u8, u8)> {
    let base = base_block_id(raw);
    if base == TORCH {
        Some((15, 12, 5))
    } else if base == LAVA || base == FIRE {
        Some((15, 8, 2))
    } else if is_active_block(raw) {
        Some((14, 7, 1))
    } else {
        None
    }
}

/// Computes the 3D `LightData` grid for a padded chunk volume (`pad × pad × pad`, typically `34³`).
///
/// Algorithm:
///   1. **Vertical Sunlight Cast:** From `y = pad - 1` down to `0` for each `(x, z)` column.
///      - Passes unhindered (`15`) through `AIR`, `TORCH`, `CABLE`, `FIRE`.
///      - Passing through `PURPLE_STAINED_GLASS` filters out the Green spectrum (`G = 0`) and
///        converts the beam into a purple RGB ray `(R: 14, G: 0, B: 14)` that continues downward!
///      - Passing through `WATER` or `OAK_LEAVES` attenuates sunlight by `2` per block.
///      - Stops (`0`) upon hitting any opaque full-cube block.
///   2. **Seed Artificial Emitters:** `TORCH`, `LAVA`, `FIRE`, and active machines seed their
///      voxel and enqueue into the BFS queue.
///   3. **6-Neighbor BFS Flood-Fill:** Propagates both Sunlight and RGB channels through non-opaque
///      voxels (`!is_full_cube`), decrementing by `1` per step, and applying an 80% Green-channel
///      attenuation (`g / 5`) whenever light traverses `PURPLE_STAINED_GLASS`.
#[allow(dead_code)]
pub fn compute_padded_light(cells: &[BlockId], pad: i32) -> Vec<LightData> {
    compute_padded_light_rect(cells, pad, pad, pad)
}

/// Computes padded per-voxel lighting across a rectangular `(pad_x, pad_y, pad_z)` neighborhood
/// so torches, lava, fire, and sunlight propagate seamlessly across 32x32 chunk borders.
pub fn compute_padded_light_rect(
    cells: &[BlockId],
    pad_x: i32,
    pad_y: i32,
    pad_z: i32,
) -> Vec<LightData> {
    let total = (pad_x * pad_y * pad_z) as usize;
    let mut light = vec![0u16; total];
    let mut queue: VecDeque<(i32, i32, i32)> = VecDeque::with_capacity(4096);

    let idx = |x: i32, y: i32, z: i32| -> usize { (x + y * pad_x + z * pad_x * pad_y) as usize };

    // 1. Vertical Sunlight rays from top of volume (`y = pad_y - 1` down to `0`)
    for z in 0..pad_z {
        for x in 0..pad_x {
            let mut sun: u8 = 15;
            let mut purple_beam: u8 = 0;

            for y in (0..pad_y).rev() {
                let raw = cells[idx(x, y, z)];
                let base = base_block_id(raw);

                if is_full_cube(raw) {
                    sun = 0;
                    purple_beam = 0;
                    continue;
                }

                if base == PURPLE_STAINED_GLASS {
                    // Sunlight or existing beam hitting purple stained glass turns into purple RGB light!
                    let incoming = sun.max(purple_beam);
                    if incoming > 1 {
                        purple_beam = incoming - 1;
                    }
                    sun = 0;
                    let i = idx(x, y, z);
                    let r = red_channel(light[i]).max(purple_beam);
                    let g = green_channel(light[i]) / 5; // 80% Green attenuation
                    let b = blue_channel(light[i]).max(purple_beam);
                    light[i] = pack_light(0, r, g, b);
                    if purple_beam > 1 {
                        queue.push_back((x, y, z));
                    }
                    continue;
                }

                if base == WATER || base == OAK_LEAVES {
                    sun = sun.saturating_sub(2);
                    purple_beam = purple_beam.saturating_sub(2);
                }

                let i = idx(x, y, z);
                let cur_sun = sun_channel(light[i]).max(sun);
                let cur_r = red_channel(light[i]).max(purple_beam);
                let cur_g = green_channel(light[i]);
                let cur_b = blue_channel(light[i]).max(purple_beam);
                light[i] = pack_light(cur_sun, cur_r, cur_g, cur_b);

                if cur_sun > 1 || purple_beam > 1 {
                    queue.push_back((x, y, z));
                }
            }
        }
    }

    // 2. Seed artificial RGB light sources (Torches, Lava, Fire, Active Machines)
    for z in 0..pad_z {
        for y in 0..pad_y {
            for x in 0..pad_x {
                let raw = cells[idx(x, y, z)];
                if let Some((er, eg, eb)) = block_emission(raw) {
                    let i = idx(x, y, z);
                    let cur_sun = sun_channel(light[i]);
                    let nr = red_channel(light[i]).max(er);
                    let ng = green_channel(light[i]).max(eg);
                    let nb = blue_channel(light[i]).max(eb);
                    light[i] = pack_light(cur_sun, nr, ng, nb);
                    queue.push_back((x, y, z));
                }
            }
        }
    }

    // 3. Multi-channel 6-way BFS Flood-Fill
    const DIRS: [(i32, i32, i32); 6] = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ];

    while let Some((x, y, z)) = queue.pop_front() {
        let cur = light[idx(x, y, z)];
        let s = sun_channel(cur);
        let r = red_channel(cur);
        let g = green_channel(cur);
        let b = blue_channel(cur);

        if s <= 1 && r <= 1 && g <= 1 && b <= 1 {
            continue;
        }

        for (dx, dy, dz) in DIRS {
            let nx = x + dx;
            let ny = y + dy;
            let nz = z + dz;
            if nx < 0 || nx >= pad_x || ny < 0 || ny >= pad_y || nz < 0 || nz >= pad_z {
                continue;
            }

            let n_raw = cells[idx(nx, ny, nz)];
            if is_full_cube(n_raw) {
                continue;
            }

            let n_base = base_block_id(n_raw);
            let step_cost: u8 = if n_base == WATER || n_base == OAK_LEAVES {
                2
            } else {
                1
            };

            let (prop_s, prop_r, prop_g, prop_b) = if n_base == PURPLE_STAINED_GLASS {
                // Passing through Purple Stained Glass converts Sunlight into R+B and cuts Green by 80%!
                let from_sun = s.saturating_sub(1);
                let nr = r.saturating_sub(1).max(from_sun);
                let ng = (g.saturating_sub(1)) / 5;
                let nb = b.saturating_sub(1).max(from_sun);
                (0, nr, ng, nb)
            } else {
                (
                    s.saturating_sub(step_cost),
                    r.saturating_sub(step_cost),
                    g.saturating_sub(step_cost),
                    b.saturating_sub(step_cost),
                )
            };

            let ni = idx(nx, ny, nz);
            let old = light[ni];
            let os = sun_channel(old);
            let or = red_channel(old);
            let og = green_channel(old);
            let ob = blue_channel(old);

            let ns = os.max(prop_s);
            let nr = or.max(prop_r);
            let ng = og.max(prop_g);
            let nb = ob.max(prop_b);

            let updated = pack_light(ns, nr, ng, nb);
            if updated != old {
                light[ni] = updated;
                queue.push_back((nx, ny, nz));
            }
        }
    }

    light
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::chunk::{AIR, STONE};

    #[test]
    fn light_data_bit_packing_roundtrip() {
        let packed = pack_light(15, 14, 7, 3);
        assert_eq!(sun_channel(packed), 15);
        assert_eq!(red_channel(packed), 14);
        assert_eq!(green_channel(packed), 7);
        assert_eq!(blue_channel(packed), 3);
    }

    #[test]
    fn torch_and_purple_stained_glass_filter_green_channel() {
        let pad = 10i32;
        let mut cells = vec![AIR; (pad * pad * pad) as usize];
        let idx = |x: i32, y: i32, z: i32| -> usize { (x + y * pad + z * pad * pad) as usize };

        // Enclose a small dark room with STONE roof at y = 8 and a STONE dividing wall at x = 5
        for z in 0..pad {
            for x in 0..pad {
                cells[idx(x, 8, z)] = STONE;
            }
            for y in 0..pad {
                cells[idx(5, y, z)] = STONE;
            }
        }

        // Place a TORCH at (4, 4, 4) -> emits (R: 15, G: 12, B: 5)
        cells[idx(4, 4, 4)] = TORCH;
        // Place a PURPLE_STAINED_GLASS window in the wall at (5, 4, 4)
        cells[idx(5, 4, 4)] = PURPLE_STAINED_GLASS;

        let light = compute_padded_light(&cells, pad);

        // At the torch itself: (15, 12, 5)
        let l_torch = light[idx(4, 4, 4)];
        assert_eq!(red_channel(l_torch), 15);
        assert_eq!(green_channel(l_torch), 12);
        assert_eq!(blue_channel(l_torch), 5);

        // Inside the purple stained glass at (5, 4, 4): R and B pass (14 and 4), while G is attenuated by >=80% (<= 2)
        let l_glass = light[idx(5, 4, 4)];
        assert_eq!(red_channel(l_glass), 14);
        assert!(green_channel(l_glass) <= 2);
        assert_eq!(blue_channel(l_glass), 4);

        // Beyond the glass at (6, 4, 4): purple-shifted light continues (R=13, G<=1, B=3)
        let l_after = light[idx(6, 4, 4)];
        assert_eq!(red_channel(l_after), 13);
        assert!(green_channel(l_after) <= 1);
        assert_eq!(blue_channel(l_after), 3);
    }
}
