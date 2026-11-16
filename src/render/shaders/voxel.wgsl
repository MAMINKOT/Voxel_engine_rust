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
@group(1) @binding(3) var t_diffuse:       texture_2d_array<f32>;
@group(1) @binding(4) var s_diffuse:       sampler;

// ── Vertex stage ──────────────────────────────────────────────────────────────
struct VertexInput {
    @location(0) position:     vec3<f32>,
    @location(1) normal:       vec3<f32>,
    @location(2) color:        vec3<f32>,
    @location(3) light:        vec4<f32>, // [R, G, B, Sun] in 0.0..=15.0
    @location(4) ao_and_flags: vec2<f32>, // [ao_or_wave_flag, mat_flag (0=std, 1=water, 2=glass, 3=fire, 4=emissive)]
    @location(5) uv:           vec2<f32>, // Tiled UVs for AddressMode::Repeat
    @location(6) tex_layer:    u32,       // Layer index in t_diffuse (0xFFFFFFFF = untextured)
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0)       normal:        vec3<f32>,
    @location(1)       color:         vec3<f32>,
    @location(2)       light:         vec4<f32>,
    @location(3)       ao_and_flags:  vec2<f32>,
    @location(4)       world_pos:     vec3<f32>,
    @location(5)       uv:            vec2<f32>,
    @location(6) @interpolate(flat) tex_layer: u32,
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
        // Procedural vertex color micro-noise (±6% hue/brightness variation from world_pos.xz)
        if !is_emissive {
            let noise = fract(sin(dot(pos.xz, vec2<f32>(12.9898, 78.233))) * 43758.5453);
            v_color = v_color * (0.94 + noise * 0.12);
        }
        // Smooth Vertex AO interpolation
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
    out.uv            = in.uv;
    out.tex_layer     = in.tex_layer;
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

// ── 16-Point Poisson Disk PCF with Rotated Dither & 10% Cascade Blend ─────────
fn interleaved_gradient_noise(pixel: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(0.06711056 * pixel.x + 0.00583715 * pixel.y));
}

fn sample_sun_shadow(world_pos: vec3<f32>, normal: vec3<f32>, screen_xy: vec2<f32>) -> f32 {
    let quality = camera.render_params.x;
    if quality < 0.5 {
        return 1.0; // Shadows Off
    }

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
        let s0 = textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2<f32>(-0.75, -0.75) * texel, ref_z);
        let s1 = textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2<f32>( 0.75, -0.75) * texel, ref_z);
        let s2 = textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2<f32>(-0.75,  0.75) * texel, ref_z);
        let s3 = textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2<f32>( 0.75,  0.75) * texel, ref_z);
        vis = (s0 + s1 + s2 + s3) * 0.25;
    } else {
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

    // Smooth shadow fade between 75.0m and 95.0m from camera + outer 10% frustum blend
    let cam_dist = length(world_pos.xz - camera.cam_pos.xz);
    let dist_fade = clamp((cam_dist - 75.0) / 20.0, 0.0, 1.0);
    let cascade_blend = clamp((edge - 0.90) / 0.10, 0.0, 1.0);
    return mix(vis, 1.0, max(dist_fade, cascade_blend));
}

// ── Shared Textured + RGB + Sunlight + Poisson PCF Shadow + Smooth AO ─────────
fn shade_voxel(in: VertexOutput) -> vec4<f32> {
    // Animated Lava (`tex_layer == 28`, 20 frames @ 32..52, slow 10-tick interval)
    // & Fire (`tex_layer == 30`, 32 frames @ 52..84, 4-tick interval)
    let global_ticks = u32(max(camera.cam_pos.w, 0.0) * 20.0);
    var sampled_layer = in.tex_layer;
    if in.tex_layer == 28u {
        sampled_layer = 32u + ((global_ticks / 10u) % 20u);
    } else if in.tex_layer == 30u {
        sampled_layer = 52u + ((global_ticks / 4u) % 32u);
    }

    // World-Aligned UV for Greedy-Meshed blocks (`in.uv.x < -0.5`) so every 1x1x1 voxel
    // gets a crisp unstretched 16x16 square tile regardless of Greedy Quad size.
    var uv = in.uv;
    if in.uv.x < -0.5 {
        if abs(in.normal.y) > 0.5 {
            uv = in.world_pos.xz;
        } else if abs(in.normal.x) > 0.5 {
            uv = vec2<f32>(in.world_pos.z, -in.world_pos.y);
        } else {
            uv = vec2<f32>(in.world_pos.x, -in.world_pos.y);
        }
    }

    // Sample unconditionally in uniform control flow, then apply if `tex_layer != 0xFFFFFFFFu`
    let safe_layer = min(sampled_layer, 108u);
    let tex_color = textureSample(t_diffuse, s_diffuse, uv, i32(safe_layer));
    let has_texture = in.tex_layer != 0xFFFFFFFFu;

    if has_texture && tex_color.a < 0.1 {
        discard;
    }

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

    let ao_factor = clamp(in.ao_and_flags.x, 0.40, 1.00);

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

    let tint_linear = pow(clamp(base_srgb, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(2.2));
    var albedo = select(tint_linear, tex_color.rgb * tint_linear, has_texture);
    let alpha = select(1.0, tex_color.a, has_texture);

    // Biome Noise Tint on Grass Top (`tex_layer == 2u`)
    if in.tex_layer == 2u {
        let biome_tint = sin(in.world_pos.x * 0.04) * cos(in.world_pos.z * 0.04) * 0.08 + 0.92;
        albedo = albedo * vec3<f32>(biome_tint * 0.95, biome_tint * 1.05, biome_tint * 0.90);
    }

    // PrimedTNT white flash overlay (`mat_flag == 5.0`)
    if abs(mat_flag - 5.0) < 0.25 {
        albedo = mix(albedo, vec3<f32>(1.0, 1.0, 1.0), 0.7);
    }

    if is_emissive {
        return vec4<f32>(albedo * 3.4, alpha);
    }

    return vec4<f32>(clamp(albedo * final_light, vec3<f32>(0.0), vec3<f32>(1.0)), alpha);
}

fn linearize_depth(d: f32) -> f32 {
    let z_near = 0.1;
    let z_far = 1000.0;
    return (z_near * z_far) / max(z_far - d * (z_far - z_near), 1e-4);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let lit = shade_voxel(in);
    var final_color = vec4<f32>(lit.rgb, 1.0);
    if camera.sun_color.w > 0.5 {
        let uw = pow(vec3<f32>(0.05, 0.25, 0.60), vec3<f32>(2.2));
        final_color = mix(final_color, vec4<f32>(uw, 1.0), 0.5);
    } else {
        // Distance Fog (`80m..180m`) blending smoothly into `sky_color`
        let dist = length(in.world_pos.xz - camera.cam_pos.xz);
        let fog_factor = clamp((dist - 80.0) / (180.0 - 80.0), 0.0, 1.0);
        final_color = vec4<f32>(
            mix(final_color.rgb, camera.sky_color.rgb, fog_factor * fog_factor),
            final_color.a
        );
    }
    return final_color;
}

@fragment
fn fs_translucent(in: VertexOutput) -> @location(0) vec4<f32> {
    let mat_flag = in.ao_and_flags.y;
    let is_water = abs(mat_flag - 1.0) < 0.25;

    if is_water {
        let base_n = normalize(in.normal);
        let time = camera.cam_pos.w;

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

        let dims = vec2<i32>(textureDimensions(scene_depth_tex));
        let px = clamp(vec2<i32>(in.clip_position.xy), vec2<i32>(0, 0), dims - vec2<i32>(1, 1));
        let bg_depth = textureLoad(scene_depth_tex, px, 0);
        let terrain_linear_depth = linearize_depth(bg_depth);
        let water_linear_depth = linearize_depth(in.clip_position.z);
        let water_depth = max(terrain_linear_depth - water_linear_depth, 0.0);

        let beer_absorption = 1.0 - exp(-water_depth * 0.4);
        let shallow_color   = vec3<f32>(0.06, 0.42, 0.58);
        let deep_navy_color = vec3<f32>(0.008, 0.032, 0.15);
        let water_color     = mix(shallow_color, deep_navy_color, beer_absorption);

        let shadow_vis = sample_sun_shadow(in.world_pos, base_n, in.clip_position.xy);
        let sun_l = normalize(camera.sun_dir.xyz);
        let sun_diff = max(dot(wave_norm, sun_l), 0.0) * shadow_vis;
        let light_scale = max(
            in.light.xyz * 0.06667,
            vec3<f32>(in.light.w * 0.06667) * (camera.sun_color.rgb * (0.45 + 0.55 * sun_diff) + vec3<f32>(camera.sun_dir.w) * 0.42)
        ) + vec3<f32>(0.05, 0.07, 0.14);

        var water_col = water_color * light_scale;

        let sky_reflect = clamp(camera.sky_color.rgb * 1.05, vec3<f32>(0.0), vec3<f32>(0.95));
        water_col = mix(water_col, sky_reflect, clamp(fresnel * 0.68, 0.0, 0.82));

        let half_vec = normalize(sun_l + view_dir);
        let spec_angle = max(dot(wave_norm, half_vec), 0.0);
        let sun_above = clamp(sun_l.y * 4.0, 0.0, 1.0);
        let specular = pow(spec_angle, 96.0) * 0.65 * shadow_vis * sun_above * camera.sun_color.rgb;
        water_col = clamp(water_col + specular, vec3<f32>(0.0), vec3<f32>(0.98));

        let base_alpha = mix(0.28, 0.90, beer_absorption);
        let alpha = clamp(max(base_alpha, fresnel * 0.78), 0.25, 0.94);
        return vec4<f32>(water_col, alpha);
    }

    let lit = shade_voxel(in);
    let is_fire = abs(mat_flag - 3.0) < 0.25;
    let alpha = select(max(lit.a, 0.55), 0.92, is_fire);
    let out_rgb = select(clamp(lit.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), lit.rgb, is_fire);
    var final_color = vec4<f32>(out_rgb, alpha);
    if camera.sun_color.w > 0.5 {
        let uw = pow(vec3<f32>(0.05, 0.25, 0.60), vec3<f32>(2.2));
        final_color = mix(final_color, vec4<f32>(uw, alpha), 0.5);
    } else {
        let dist = length(in.world_pos.xz - camera.cam_pos.xz);
        let fog_factor = clamp((dist - 80.0) / (180.0 - 80.0), 0.0, 1.0);
        final_color = vec4<f32>(
            mix(final_color.rgb, camera.sky_color.rgb, fog_factor * fog_factor),
            final_color.a
        );
    }
    return final_color;
}
