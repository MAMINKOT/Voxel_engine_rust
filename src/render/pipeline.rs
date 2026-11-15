// Render pipeline module (Step 10: Sharp 1:1 HDR `Rgba16Float`, Vertex Corner AO,
// 16-Point Poisson Disk PCF Shadows with Dither & 10% Cascade Blend, Cloud Ground Shadows,
// Strict Emissive-Only Bloom Isolation, and Clean Dynamic Water without Underwater Edge Grids).

use wgpu::util::DeviceExt;

use crate::render::postprocess::HDR_FORMAT;

// ─── Inline WGSL Shaders ─────────────────────────────────────────────────────

const SKY_SHADER_SRC: &str = r#"
struct CameraUniform {
    view_proj:       mat4x4<f32>,
    inv_view_proj:   mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    sun_dir:         vec4<f32>,
    sun_color:       vec4<f32>,
    sky_color:       vec4<f32>,
    cam_pos:         vec4<f32>,
    sun_screen:      vec4<f32>,
    render_params:   vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct SkyVarying {
    @builtin(position) clip_position: vec4<f32>,
    @location(0)       uv:            vec2<f32>,
    @location(1) @interpolate(flat) quad_id: u32,
};

fn quad_corner(corner_idx: u32) -> vec2<f32> {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );
    return corners[corner_idx];
}

@vertex
fn vs_sky(@builtin(vertex_index) idx: u32) -> SkyVarying {
    var out: SkyVarying;

    // Pass A (vertices 0..3): Full-screen sky dome triangle
    if idx < 3u {
        let x = f32(i32(idx & 1u) * 4 - 1);
        let y = f32(i32(idx & 2u) * 2 - 1);
        out.clip_position = vec4<f32>(x, y, 0.9999, 1.0);
        out.uv = vec2<f32>(x, y);
        out.quad_id = 0u;
        return out;
    }

    let sun_direction = normalize(camera.sun_dir.xyz);
    let horizon_factor = camera.sky_color.w;

    // Pass B (vertices 3..9): 3D World-Space Sun Billboard Quad at `camera.pos + sun_direction * 400.0`
    if idx < 9u {
        let c = quad_corner(idx - 3u);
        let right = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), sun_direction));
        let up = normalize(cross(sun_direction, right));
        let half_size = 38.0 * (1.0 + 0.65 * horizon_factor);
        let center = camera.cam_pos.xyz + sun_direction * 400.0;
        let world_pos = center + (right * c.x + up * c.y) * half_size;
        let clip = camera.view_proj * vec4<f32>(world_pos, 1.0);
        out.clip_position = vec4<f32>(clip.xy, clip.w * 0.9998, clip.w);
        out.uv = c;
        out.quad_id = 1u;
        return out;
    }

    // Pass C (vertices 9..15): 3D World-Space Moon Billboard Quad at `camera.pos - sun_direction * 400.0`
    let c = quad_corner(idx - 9u);
    let moon_dir = -sun_direction;
    let right = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), moon_dir));
    let up = normalize(cross(moon_dir, right));
    let half_size = 28.0;
    let center = camera.cam_pos.xyz + moon_dir * 400.0;
    let world_pos = center + (right * c.x + up * c.y) * half_size;
    let clip = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.clip_position = vec4<f32>(clip.xy, clip.w * 0.9998, clip.w);
    out.uv = c;
    out.quad_id = 2u;
    return out;
}

@fragment
fn fs_sky(in: SkyVarying) -> @location(0) vec4<f32> {
    let horizon_factor = camera.sky_color.w;

    // 1. 3D Sun Billboard Quad (`quad_id == 1`) — clamped <= 1.0 so sky does not wash out Bloom
    if in.quad_id == 1u {
        let r = length(in.uv);
        let box_d = max(abs(in.uv.x), abs(in.uv.y));
        if r > 1.0 {
            discard;
        }
        let core_zenith  = vec3<f32>(1.00, 0.99, 0.92);
        let core_horizon = vec3<f32>(1.00, 0.28, 0.06);
        let glow_zenith  = vec3<f32>(1.00, 0.82, 0.35);
        let glow_horizon = vec3<f32>(0.92, 0.16, 0.03);

        let core_col = mix(core_zenith, core_horizon, horizon_factor);
        let glow_col = mix(glow_zenith, glow_horizon, horizon_factor);

        let core_mask = 1.0 - smoothstep(0.36, 0.48, box_d);
        let corona = pow(clamp(1.0 - r, 0.0, 1.0), 1.8);
        let col = clamp(mix(glow_col, core_col, core_mask), vec3<f32>(0.0), vec3<f32>(1.0));
        let alpha = clamp(core_mask + corona * 0.85, 0.0, 1.0);
        return vec4<f32>(col, alpha);
    }

    // 2. 3D Moon Billboard Quad (`quad_id == 2`)
    if in.quad_id == 2u {
        let r = length(in.uv);
        let box_d = max(abs(in.uv.x), abs(in.uv.y));
        if r > 1.0 {
            discard;
        }
        let core_col = vec3<f32>(0.92, 0.96, 1.00);
        let halo_col = vec3<f32>(0.48, 0.64, 0.95);
        let core_mask = 1.0 - smoothstep(0.38, 0.46, box_d);
        let halo = pow(clamp(1.0 - r, 0.0, 1.0), 2.2);
        let col = clamp(mix(halo_col, core_col, core_mask), vec3<f32>(0.0), vec3<f32>(1.0));
        let alpha = clamp(core_mask * 0.95 + halo * 0.55, 0.0, 1.0);
        return vec4<f32>(col, alpha);
    }

    // 3. Sky Dome Gradient (`quad_id == 0`)
    let world_h = camera.inv_view_proj * vec4<f32>(in.uv, 1.0, 1.0);
    let ray_dir = normalize(world_h.xyz / world_h.w);
    let horizon_mix = clamp(ray_dir.y * 0.5 + 0.5, 0.0, 1.0);
    var col = camera.sky_color.rgb * mix(1.14, 0.86, horizon_mix);

    let sun_direction = normalize(camera.sun_dir.xyz);
    let sun_dot = max(dot(ray_dir, sun_direction), 0.0);
    let sunset_haze = pow(sun_dot, 6.0) * horizon_factor * 0.35;
    col = clamp(col + vec3<f32>(0.95, 0.25, 0.05) * sunset_haze, vec3<f32>(0.0), vec3<f32>(1.0));

    var final_color = vec4<f32>(col, 1.0);
    if camera.sun_color.w > 0.5 {
        let uw = pow(vec3<f32>(0.05, 0.25, 0.60), vec3<f32>(2.2));
        final_color = mix(final_color, vec4<f32>(uw, 1.0), 0.72);
    }
    return final_color;
}
"#;

const SHADER_SRC: &str = r#"
// ── Camera & Sky uniform ──────────────────────────────────────────────────────
struct CameraUniform {
    view_proj:       mat4x4<f32>,
    inv_view_proj:   mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    sun_dir:         vec4<f32>,
    sun_color:       vec4<f32>,
    sky_color:       vec4<f32>,
    cam_pos:         vec4<f32>, // xyz = cam_pos, w = time in seconds
    sun_screen:      vec4<f32>,
    render_params:   vec4<f32>, // x = shadow_quality (0/1/2), y = bloom_strength, z = clouds_enabled, w = water_waves
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0) var shadow_map:      texture_depth_2d;
@group(1) @binding(1) var shadow_sampler:  sampler_comparison;
@group(1) @binding(2) var scene_depth_tex: texture_depth_2d;

// ── Vertex stage ──────────────────────────────────────────────────────────────
struct VertexInput {
    @location(0) position:     vec3<f32>,
    @location(1) normal:       vec3<f32>,
    @location(2) color:        vec3<f32>,
    @location(3) light:        vec4<f32>, // [R, G, B, Sun] in 0.0..=15.0
    @location(4) ao_and_flags: vec2<f32>, // [ao_or_shore_mask, mat_flag (0=std, 1=water, 2=glass, 3=fire, 4=emissive)]
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0)       normal:        vec3<f32>,
    @location(1)       color:         vec3<f32>,
    @location(2)       light:         vec4<f32>,
    @location(3)       ao_and_flags:  vec2<f32>,
    @location(4)       world_pos:     vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var pos = in.position;
    var n = in.normal;
    var v_color = in.color;
    var ao_and_flags = in.ao_and_flags;
    let time = camera.cam_pos.w;

    let mat_flag = in.ao_and_flags.y;
    let is_water = abs(mat_flag - 1.0) < 0.25;
    let is_emissive = abs(mat_flag - 3.0) < 0.25 || abs(mat_flag - 4.0) < 0.25;

    // Vertex Wave Displacement for Water (`material_flag == 1.0`) when enabled in Graphics Settings
    if is_water && camera.render_params.w > 0.5 && in.ao_and_flags.x > 0.5 {
        let offset_y = sin(pos.x * 1.5 + time * 3.0) * cos(pos.z * 1.5 + time * 2.5) * 0.04;
        pos.y += offset_y;
    }

    if !is_water {
        // Task 4.1: Procedural vertex color micro-noise (±6% hue/brightness variation from world_pos.xz)
        if !is_emissive {
            let noise = fract(sin(dot(pos.xz, vec2<f32>(12.9898, 78.233))) * 43758.5453);
            v_color = v_color * (0.94 + noise * 0.12);
        }
        // Task 4.2: Smooth Vertex AO interpolation — convert discrete AO level `0..3` at the vertex
        // so the rasterizer bilinearly interpolates `ao_factor` smoothly across every quad without hard bands.
        let ao_level = clamp(u32(round(in.ao_and_flags.x)), 0u, 3u);
        let ao_lut = vec4<f32>(0.42, 0.67, 0.84, 1.00);
        ao_and_flags.x = ao_lut[ao_level];
    }

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(pos, 1.0);
    out.normal        = n;
    out.color         = v_color;
    out.light         = in.light;
    out.ao_and_flags  = ao_and_flags;
    out.world_pos     = pos;
    return out;
}

// ── Cloud Noise for Soft Moving Ground Shadows (`Y = 128.0`) ──────────────────
fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn value_noise2d(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i + vec2<f32>(0.0, 0.0));
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn cloud_fbm(uv: vec2<f32>) -> f32 {
    var val = 0.0;
    val += 0.58 * value_noise2d(uv * 16.0);
    val += 0.28 * value_noise2d(uv * 32.0 + vec2<f32>(17.3, 9.1));
    val += 0.14 * value_noise2d(uv * 64.0 + vec2<f32>(43.7, 28.5));
    return val;
}

fn sample_cloud_shadow(world_pos: vec3<f32>, sun_l: vec3<f32>) -> f32 {
    if camera.render_params.z < 0.5 || sun_l.y <= 0.05 {
        return 1.0;
    }
    let time = camera.cam_pos.w;
    let wind = vec2<f32>(time * 4.5, time * 1.8);
    let ray_t = max(128.0 - world_pos.y, 0.0) / max(sun_l.y, 0.18);
    let cloud_xz = world_pos.xz + sun_l.xz * ray_t;
    let uv = (cloud_xz + wind) * 0.003;
    let n = cloud_fbm(uv);
    if n <= 0.55 {
        return 1.0;
    }
    return mix(1.0, 0.72, smoothstep(0.55, 0.72, n));
}

// ── Task 3: 16-Point Poisson Disk PCF with Rotated Dither & 10% Cascade Blend ──
fn interleaved_gradient_noise(pixel: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(0.06711056 * pixel.x + 0.00583715 * pixel.y));
}

fn sample_sun_shadow(world_pos: vec3<f32>, normal: vec3<f32>, screen_xy: vec2<f32>) -> f32 {
    let quality = camera.render_params.x;
    if quality < 0.5 {
        return 1.0; // Shadows Off
    }

    // Task 2.3: 16-Point Poisson Disk PCF with `normal_bias = 0.02`
    let light_clip = camera.light_view_proj * vec4<f32>(world_pos + normal * 0.02, 1.0);
    let proj = light_clip.xyz / light_clip.w;
    let uv = vec2<f32>(proj.x * 0.5 + 0.5, 0.5 - proj.y * 0.5);

    let edge = max(abs(proj.x), abs(proj.y));
    if edge >= 1.0 || proj.z <= 0.0 || proj.z >= 1.0 {
        return 1.0;
    }

    let ref_z = proj.z - 0.0012;
    let texel = 1.0 / 2048.0;
    var vis = 0.0;

    if quality < 1.5 {
        // Low quality: fast 4-tap PCF
        let s0 = textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2<f32>(-0.75, -0.75) * texel, ref_z);
        let s1 = textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2<f32>( 0.75, -0.75) * texel, ref_z);
        let s2 = textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2<f32>(-0.75,  0.75) * texel, ref_z);
        let s3 = textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2<f32>( 0.75,  0.75) * texel, ref_z);
        vis = (s0 + s1 + s2 + s3) * 0.25;
    } else {
        // High quality: 16-Point Poisson Disk PCF with per-pixel rotated dither
        var poisson = array<vec2<f32>, 16>(
            vec2<f32>(-0.94201624, -0.39906216),
            vec2<f32>( 0.94558609, -0.76890725),
            vec2<f32>(-0.09418410, -0.92938870),
            vec2<f32>( 0.34495938,  0.29387760),
            vec2<f32>(-0.91588581,  0.45771432),
            vec2<f32>(-0.81544232, -0.87912464),
            vec2<f32>(-0.38277543,  0.27676845),
            vec2<f32>( 0.97484398,  0.75648379),
            vec2<f32>( 0.44323325, -0.97511554),
            vec2<f32>( 0.53742981, -0.47373420),
            vec2<f32>(-0.26496911, -0.41893023),
            vec2<f32>( 0.79197514,  0.19090188),
            vec2<f32>(-0.24188840,  0.99706507),
            vec2<f32>(-0.81409955,  0.91437590),
            vec2<f32>( 0.19984126,  0.78641367),
            vec2<f32>( 0.14383161, -0.14100790),
        );

        let angle = interleaved_gradient_noise(screen_xy) * 6.2831853;
        let s = sin(angle);
        let c = cos(angle);
        let spread = texel * 2.1;

        for (var i = 0; i < 16; i = i + 1) {
            let p = poisson[i];
            let rotated = vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c) * spread;
            vis += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + rotated, ref_z);
        }
        vis = vis * (1.0 / 16.0);
    }

    // Task 3.2: Linearly blend the outer 10% (`0.90..1.00`) of the cascade zone to eliminate seam lines
    let cascade_blend = clamp((edge - 0.90) / 0.10, 0.0, 1.0);
    return mix(vis, 1.0, cascade_blend);
}

// ── Shared RGB + Sunlight + Poisson PCF Shadow + Corner AO + Isolated Emissive ──
fn shade_voxel(in: VertexOutput) -> vec3<f32> {
    let n = normalize(in.normal);
    let sun_l = normalize(camera.sun_dir.xyz);
    let moon_l = -sun_l;

    var face_factor = 0.78;
    if n.y > 0.5 {
        face_factor = 1.00;
    } else if n.y < -0.5 {
        face_factor = 0.56;
    } else if abs(n.z) > 0.5 {
        face_factor = 0.86;
    } else {
        face_factor = 0.74;
    }

    // Smoothly interpolated Vertex Corner AO factor (computed at vertices in `vs_main`)
    let ao_factor = clamp(in.ao_and_flags.x, 0.40, 1.00);

    // 16-Point Poisson PCF Shadow + Cloud Ground Shadow
    let shadow_vis = sample_sun_shadow(in.world_pos, n, in.clip_position.xy)
        * sample_cloud_shadow(in.world_pos, sun_l);

    let sun_diff = max(dot(n, sun_l), 0.0) * clamp(sun_l.y + 0.15, 0.0, 1.0) * shadow_vis;
    let moon_diff = max(dot(n, moon_l), 0.0) * clamp(moon_l.y + 0.10, 0.0, 1.0) * shadow_vis;
    let moon_term = (0.24 * moon_diff) * vec3<f32>(0.45, 0.62, 0.95);

    let rgb_light = in.light.xyz * 0.06667;
    let sun_ambient = vec3<f32>(camera.sun_dir.w) * 0.42;
    let sun_direct = camera.sun_color.rgb * (0.95 * sun_diff) + moon_term;
    let sun_light = (in.light.w * 0.06667) * (sun_ambient + sun_direct);

    let ambient_floor = vec3<f32>(0.04, 0.06, 0.12);
    let final_light = (max(rgb_light, sun_light) + ambient_floor) * face_factor * ao_factor;

    // Task 4.1: Combine vertex color noise with subtle world-space voxel micro-variation
    // so even large greedy-merged terrain faces have organic non-plastic texture.
    let mat_flag = in.ao_and_flags.y;
    let is_emissive = abs(mat_flag - 3.0) < 0.25 || abs(mat_flag - 4.0) < 0.25;
    var base_srgb = in.color;
    if !is_emissive {
        let block_xz = floor(in.world_pos.xz - n.xz * 0.02);
        let micro_noise = fract(sin(dot(block_xz, vec2<f32>(12.9898, 78.233))) * 43758.5453);
        let smooth_var = value_noise2d(in.world_pos.xz * 1.35 + vec2<f32>(in.world_pos.y * 0.7));
        let combined_noise = mix(micro_noise, smooth_var, 0.5);
        base_srgb = base_srgb * (0.95 + combined_noise * 0.10);
    }

    let clamped_srgb = clamp(base_srgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let albedo = pow(clamped_srgb, vec3<f32>(2.2));

    // Task 2.1: Strict Bloom Isolation — ONLY `MAT_FIRE (3.0)` and `MAT_EMISSIVE (4.0)`
    // (`Lava`, `Torch` flame, `Fire`, and active `ElectricFurnace` / `CoalGenerator` / furnaces)
    // emit HDR radiance > 1.0! Glass, TNT, and terrain are strictly clamped to [0.0, 1.0].
    if is_emissive {
        return albedo * 3.4;
    }

    return clamp(albedo * final_light, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn linearize_depth(d: f32) -> f32 {
    let z_near = 0.1;
    let z_far = 1000.0;
    return (z_near * z_far) / max(z_far - d * (z_far - z_near), 1e-4);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let lit = shade_voxel(in);
    var final_color = vec4<f32>(lit, 1.0);
    if camera.sun_color.w > 0.5 {
        let uw = pow(vec3<f32>(0.05, 0.25, 0.60), vec3<f32>(2.2));
        final_color = mix(final_color, vec4<f32>(uw, 1.0), 0.5);
    }
    return final_color;
}

@fragment
fn fs_translucent(in: VertexOutput) -> @location(0) vec4<f32> {
    let mat_flag = in.ao_and_flags.y;
    let is_water = abs(mat_flag - 1.0) < 0.25;

    // ── Task 1 & Task 3: Crystal-Clear Sloped Water with Beer-Lambert Depth & Wave Specular ──
    if is_water {
        let base_n = normalize(in.normal);
        let time = camera.cam_pos.w;

        // Task 3.2: Normal Perturbation from two sinusoids + Blinn-Phong solar specular highlight
        var wave_norm = base_n;
        if camera.render_params.w > 0.5 && base_n.y > 0.15 {
            let wn = normalize(vec3<f32>(
                sin(in.world_pos.x * 2.0 + time * 3.0) * 0.05,
                1.0,
                cos(in.world_pos.z * 2.0 + time * 2.0) * 0.05,
            ));
            wave_norm = normalize(vec3<f32>(base_n.x + wn.x, base_n.y * wn.y, base_n.z + wn.z));
        }

        let view_dir = normalize(camera.cam_pos.xyz - in.world_pos);
        let ndotv = clamp(dot(view_dir, wave_norm), 0.0, 1.0);
        let fresnel = pow(1.0 - ndotv, 3.0);

        // Task 3.1: Beer-Lambert Law depth absorption (`1.0 - exp(-water_depth * 0.4)`)
        let dims = vec2<i32>(textureDimensions(scene_depth_tex));
        let px = clamp(vec2<i32>(in.clip_position.xy), vec2<i32>(0, 0), dims - vec2<i32>(1, 1));
        let bg_depth = textureLoad(scene_depth_tex, px, 0);
        let terrain_linear_depth = linearize_depth(bg_depth);
        let water_linear_depth = linearize_depth(in.clip_position.z);
        let water_depth = max(terrain_linear_depth - water_linear_depth, 0.0);

        let beer_absorption = 1.0 - exp(-water_depth * 0.4);
        let shallow_color   = vec3<f32>(0.06, 0.42, 0.58); // Transparent aquamarine shallows
        let deep_navy_color = vec3<f32>(0.008, 0.032, 0.15); // Deep saturated sapphire/navy
        let water_color     = mix(shallow_color, deep_navy_color, beer_absorption);

        let shadow_vis = sample_sun_shadow(in.world_pos, base_n, in.clip_position.xy);
        let sun_l = normalize(camera.sun_dir.xyz);
        let sun_diff = max(dot(wave_norm, sun_l), 0.0) * shadow_vis;
        let light_scale = max(
            in.light.xyz * 0.06667,
            vec3<f32>(in.light.w * 0.06667) * (camera.sun_color.rgb * (0.45 + 0.55 * sun_diff) + vec3<f32>(camera.sun_dir.w) * 0.42)
        ) + vec3<f32>(0.05, 0.07, 0.14);

        var water_col = water_color * light_scale;

        // Sky reflection via Fresnel
        let sky_reflect = clamp(camera.sky_color.rgb * 1.05, vec3<f32>(0.0), vec3<f32>(0.95));
        water_col = mix(water_col, sky_reflect, clamp(fresnel * 0.68, 0.0, 0.82));

        // Blinn-Phong solar specular highlight on wave crests
        let half_vec = normalize(sun_l + view_dir);
        let spec_angle = max(dot(wave_norm, half_vec), 0.0);
        let sun_above = clamp(sun_l.y * 4.0, 0.0, 1.0);
        let specular = pow(spec_angle, 96.0) * 0.65 * shadow_vis * sun_above * camera.sun_color.rgb;
        water_col = clamp(water_col + specular, vec3<f32>(0.0), vec3<f32>(0.98));

        // Task 1.3: Zero white edge lines / zero foam on water block junctions!
        let base_alpha = mix(0.28, 0.90, beer_absorption);
        let alpha = clamp(max(base_alpha, fresnel * 0.78), 0.25, 0.94);
        return vec4<f32>(water_col, alpha);
    }

    let lit = shade_voxel(in);
    let is_fire = abs(mat_flag - 3.0) < 0.25;
    let alpha = select(0.60, 0.88, is_fire);
    // Task 2.1: Stained glass (`mat_flag == 2.0`) is strictly clamped <= 1.0 (emission = 0.0)
    let out_rgb = select(clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0)), lit, is_fire);
    var final_color = vec4<f32>(out_rgb, alpha);
    if camera.sun_color.w > 0.5 {
        let uw = pow(vec3<f32>(0.05, 0.25, 0.60), vec3<f32>(2.2));
        final_color = mix(final_color, vec4<f32>(uw, alpha), 0.5);
    }
    return final_color;
}
"#;

// ─── Depth Texture ───────────────────────────────────────────────────────────

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub struct DepthTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    #[allow(dead_code)]
    pub sampler: wgpu::Sampler,
}

impl DepthTexture {
    pub fn create(device: &wgpu::Device, width: u32, height: u32, label: &str) -> Self {
        let size = wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
        }
    }
}

// ─── Pipeline Builders ────────────────────────────────────────────────────────

/// Vertex layout matching the WGSL `VertexInput`.
/// Position (3×f32) + Normal (3×f32) + Color (3×f32) + Light (4×f32) + AO/Flags (2×f32) = 60 bytes.
pub fn vertex_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRIBS: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x3,
        3 => Float32x4,
        4 => Float32x2
    ];
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<[f32; 15]>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTRIBS,
    }
}

/// Bind group layout `@group(1)` holding the 2048x2048 Shadow Map, comparison sampler,
/// and the opaque scene depth texture copy (used for Water Depth Fade).
pub fn create_scene_env_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("scene_env_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Depth,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Depth,
                },
                count: None,
            },
        ],
    })
}

pub fn create_scene_env_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    shadow_view: &wgpu::TextureView,
    shadow_sampler: &wgpu::Sampler,
    scene_depth_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scene_env_bind_group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(shadow_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(shadow_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(scene_depth_view),
            },
        ],
    })
}

/// Full-screen sky dome + 3D World-Space Sun & Moon Billboard Quads (`draw(0..15, 0..1)`),
/// rendering into the `HDR_FORMAT` (`Rgba16Float`) framebuffer.
pub fn create_sky_pipeline(
    device: &wgpu::Device,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sky_shader"),
        source: wgpu::ShaderSource::Wgsl(SKY_SHADER_SRC.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sky_pipeline_layout"),
        bind_group_layouts: &[camera_bind_group_layout],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("sky_render_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_sky",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_sky",
            targets: &[Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Always,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    })
}

/// Opaque voxel geometry pipeline (`HDR_FORMAT`, `depth_write_enabled: true`, backface culling).
pub fn create_render_pipeline(
    device: &wgpu::Device,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
    scene_env_bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("voxel_shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("render_pipeline_layout"),
        bind_group_layouts: &[camera_bind_group_layout, scene_env_bgl],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("voxel_render_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[vertex_buffer_layout()],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
    })
}

/// Translucent water/fire/stained-glass pipeline (`HDR_FORMAT`, `ALPHA_BLENDING`,
/// `depth_write_enabled: false`, `depth_compare: LessEqual`, `cull_mode: None`).
pub fn create_translucent_pipeline(
    device: &wgpu::Device,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
    scene_env_bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("voxel_translucent_shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("translucent_pipeline_layout"),
        bind_group_layouts: &[camera_bind_group_layout, scene_env_bgl],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("voxel_translucent_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[vertex_buffer_layout()],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_translucent",
            targets: &[Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
    })
}

// ─── Camera Bind Group Layout ─────────────────────────────────────────────────

pub fn create_camera_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("camera_bind_group_layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

pub fn create_camera_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("camera_bind_group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    })
}

pub fn create_camera_buffer(device: &wgpu::Device, data: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("camera_uniform_buffer"),
        contents: data,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}
