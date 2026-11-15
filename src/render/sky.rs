// Dynamic Day/Night cycle: sun orbit, sky clear color, directional sun color, and ambient light.

use glam::Vec3;

/// Full day-night cycle length in cycle units (seconds at 1x speed).
pub const DAY_CYCLE_SECS: f32 = 1200.0;

pub struct SkyState {
    /// Current time in `0.0..DAY_CYCLE_SECS`.
    pub day_time: f32,
    /// Multiplier for how fast `day_time` progresses per real second (default 10x so a full
    /// 1200s cycle completes in 2 real minutes; `T` key can also step time manually).
    pub time_scale: f32,
}

impl SkyState {
    pub fn new() -> Self {
        Self {
            // Start in mid-morning (sun well above the horizon).
            day_time: 180.0,
            time_scale: 10.0,
        }
    }

    pub fn update(&mut self, dt_secs: f32) {
        self.day_time = (self.day_time + dt_secs * self.time_scale).rem_euclid(DAY_CYCLE_SECS);
    }

    /// Jumps forward in the cycle (e.g. when pressing `T` to test day/sunset/night).
    pub fn skip(&mut self, delta: f32) {
        self.day_time = (self.day_time + delta).rem_euclid(DAY_CYCLE_SECS);
    }

    pub fn sun_angle(&self) -> f32 {
        (self.day_time / DAY_CYCLE_SECS) * std::f32::consts::TAU
    }

    /// Normalized vector pointing towards the sun: `[cos(a), sin(a), 0.3].normalize()`.
    pub fn sun_direction(&self) -> Vec3 {
        let a = self.sun_angle();
        Vec3::new(a.cos(), a.sin(), 0.3).normalize()
    }

    /// Returns `1.0` when the sun is near the horizon (sunrise/sunset) and `0.0` near zenith/midnight.
    /// Used to enlarge the Sun billboard and shift its disc color to blood-red / orange at dawn & dusk.
    pub fn sun_horizon_factor(&self) -> f32 {
        let elev = self.sun_elevation();
        if elev < -0.22 || elev > 0.40 {
            0.0
        } else {
            (1.0 - (elev.abs() / 0.35)).clamp(0.0, 1.0)
        }
    }

    /// Sine of the sun angle (`-1.0` at midnight, `0.0` at horizon, `+1.0` at noon).
    pub fn sun_elevation(&self) -> f32 {
        self.sun_angle().sin()
    }

    pub fn phase_name(&self) -> &'static str {
        let elev = self.sun_elevation();
        if elev > 0.22 {
            "Day"
        } else if elev >= -0.15 {
            if self.sun_angle().cos() >= 0.0 {
                "Sunrise"
            } else {
                "Sunset"
            }
        } else {
            "Night"
        }
    }

    /// Sky background color in linear RGB:
    ///   • Day:            `[0.45, 0.65, 0.95]` (sRGB)
    ///   • Sunrise/Sunset: `[0.85, 0.45, 0.25]` (sRGB)
    ///   • Night:          `[0.02, 0.02, 0.05]` (sRGB)
    pub fn sky_linear_rgb(&self) -> [f32; 3] {
        let elev = self.sun_elevation();
        let day = Vec3::new(0.45, 0.65, 0.95);
        let dusk = Vec3::new(0.85, 0.45, 0.25);
        let night = Vec3::new(0.02, 0.02, 0.05);

        let rgb = if elev >= 0.25 {
            day
        } else if elev >= 0.0 {
            let t = (elev / 0.25).clamp(0.0, 1.0);
            dusk.lerp(day, t)
        } else if elev >= -0.25 {
            let t = ((elev + 0.25) / 0.25).clamp(0.0, 1.0);
            night.lerp(dusk, t)
        } else {
            night
        };

        [rgb.x.powf(2.2), rgb.y.powf(2.2), rgb.z.powf(2.2)]
    }

    pub fn clear_color(&self) -> wgpu::Color {
        let [r, g, b] = self.sky_linear_rgb();
        wgpu::Color {
            r: r as f64,
            g: g as f64,
            b: b as f64,
            a: 1.0,
        }
    }

    /// Directional sunlight color in linear RGB (fades to zero at night).
    pub fn sun_color(&self) -> [f32; 3] {
        let elev = self.sun_elevation();
        let noon = Vec3::new(1.00, 0.96, 0.88);
        let horizon = Vec3::new(1.00, 0.50, 0.22);

        let c = if elev >= 0.25 {
            noon
        } else if elev >= 0.0 {
            let t = (elev / 0.25).clamp(0.0, 1.0);
            horizon.lerp(noon, t)
        } else if elev >= -0.15 {
            let t = ((elev + 0.15) / 0.15).clamp(0.0, 1.0);
            Vec3::ZERO.lerp(horizon, t)
        } else {
            Vec3::ZERO
        };
        [c.x, c.y, c.z]
    }

    /// Ambient light strength (bright during the day, dim at night).
    pub fn ambient_strength(&self) -> f32 {
        let elev = self.sun_elevation();
        if elev >= 0.25 {
            0.35
        } else if elev >= -0.20 {
            let t = ((elev + 0.20) / 0.45).clamp(0.0, 1.0);
            0.09 + t * (0.35 - 0.09)
        } else {
            0.09
        }
    }
}
