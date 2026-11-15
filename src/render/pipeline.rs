// Render pipeline module (Step 9: HDR `Rgba16Float`, Vertex Corner AO, 3x3 PCF Directional
// Shadows, and Dynamic Water with Vertex Waves, Fresnel Reflection, Depth Fade & Shore Foam).

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

    // 1. 3D Sun Billboard Quad (`quad_id == 1`) — HDR emission > 1.0 for natural solar Bloom
    if in.quad_id == 1u {
        let r = length(in.uv);
        let box_d = max(abs(in.uv.x), abs(in.uv.y));
        if r > 1.0 {
            discard;
        }
        let core_zenith  = vec3<f32>(4.20, 4.00, 3.40);
        let core_horizon = vec3<f32>(3.80, 0.85, 0.18);
        let glow_zenith  = vec3<f32>(1.80, 1.35, 0.50);
        let glow_horizon = vec3<f32>(1.75, 0.28, 0.04);

        let core_col = mix(core_zenith, core_horizon, horizon_factor);
        let glow_col = mix(glow_zenith, glow_horizon, horizon_factor);

        let core_mask = 1.0 - smoothstep(0.36, 0.48, box_d);
        let corona = pow(clamp(1.0 - r, 0.0, 1.0), 1.8);
        let col = mix(glow_col, core_col, core_mask);
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
        let core_col = vec3<f32>(1.65, 1.75, 1.95);
        let halo_col = vec3<f32>(0.55, 0.72, 1.10);
        let core_mask = 1.0 - smoothstep(0.38, 0.46, box_d);
        let halo = pow(clamp(1.0 - r, 0.0, 1.0), 2.2);
        let col = mix(halo_col, core_col, core_mask);
        let alpha = clamp(core_mask * 0.95 + halo * 0.55, 0.0, 1.0);
        return vec4<f32>(col, alpha);
    }

    // 3. Sky Dome Gradient (`quad_id == 0`)
    let world_h = camera.inv_view_proj * vec4<f32>(in.uv, 1.0, 1.0);
    let ray_dir = normalize(world_h.xyz / world_h.w);
    let horizon_mix = clamp(ray_dir.y * 0.5 + 0.5, 0.0, 1.0);
    var col = camera.sky_color.rgb * mix(1.18, 0.86, horizon_mix);

    let sun_direction = normalize(camera.sun_dir.xyz);
    let sun_dot = max(dot(ray_dir, sun_direction), 0.0);
    let sunset_haze = pow(sun_dot, 6.0) * horizon_factor * 0.45;
    col = col + vec3<f32>(1.15, 0.30, 0.06) * sunset_haze;

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
    @location(4) ao_and_flags: vec2<f32>, // [ao_level (0..3), material_flag (0=std, 1=water, 2=glass, 3=fire)]
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
    let time = camera.cam_pos.w;

    // Task 5.1: Vertex Wave Displacement for Water (`material_flag == 1.0`)
    let is_water = abs(in.ao_and_flags.y - 1.0) < 0.25;
    if is_water && (n.y > 0.5 || fract(pos.y) > 0.02) {
        let offset_y = sin(pos.x * 1.5 + time * 3.0) * cos(pos.z * 1.5 + time * 2.5) * 0.06;
        pos.y += offset_y;
        if n.y > 0.5 {
            // Analytical normal perturbation from wave slope
            let dydx = 1.5 * cos(in.position.x * 1.5 + time * 3.0) * cos(in.position.z * 1.5 + time * 2.5) * 0.06;
            let dydz = -1.5 * sin(in.position.x * 1.5 + time * 3.0) * sin(in.position.z * 1.5 + time * 2.5) * 0.06;
            n = normalize(vec3<f32>(-dydx, 1.0, -dydz));
        }
    }

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(pos, 1.0);
    out.normal        = n;
    out.color         = in.color;
    out.light         = in.light;
    out.ao_and_flags  = in.ao_and_flags;
    out.world_pos     = pos;
    return out;
}

// ── Task 2.3: 3x3 PCF Directional Shadow Sampling ────────────────────────────
fn sample_sun_shadow(world_pos: vec3<f32>, normal: vec3<f32>) -> f32 {
    let light_clip = camera.light_view_proj * vec4<f32>(world_pos + normal * 0.035, 1.0);
    let proj = light_clip.xyz / light_clip.w;
    let uv = vec2<f32>(proj.x * 0.5 + 0.5, 0.5 - proj.y * 0.5);

    if uv.x <= 0.002 || uv.x >= 0.998 || uv.y <= 0.002 || uv.y >= 0.998 || proj.z <= 0.0 || proj.z >= 1.0 {
        return 1.0;
    }

    let texel = 1.0 / 2048.0;
    let ref_z = proj.z - 0.0015;
    var vis = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let offset = vec2<f32>(f32(dx), f32(dy)) * texel;
            vis += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + offset, ref_z);
        }
    }
    vis = vis * (1.0 / 9.0);

    // Smooth fade out at the boundary of the 80-block shadow frustum
    let edge = max(abs(proj.x), abs(proj.y));
    let fade = smoothstep(0.88, 0.99, edge);
    return mix(vis, 1.0, fade);
}

// ── Shared RGB + Sunlight + PCF Shadow + Corner AO + HDR Emissive Shading ────
fn shade_voxel(in: VertexOutput) -> vec3<f32> {
    let n = normalize(in.normal);
    let sun_l = normalize(camera.sun_dir.xyz);
    let moon_l = -sun_l;

    // Directional face contrast
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

    // Task 1.2: Vertex Corner AO factor `vec4(0.4, 0.65, 0.82, 1.0)[ao_level]`
    let ao_level = clamp(u32(round(in.ao_and_flags.x)), 0u, 3u);
    let ao_lut = vec4<f32>(0.40, 0.65, 0.82, 1.00);
    let ao_factor = ao_lut[ao_level];

    // Task 2.3: 3x3 PCF Shadow visibility (if shadowed, direct sun_diff drops to 0 -> ambient level)
    let shadow_vis = sample_sun_shadow(in.world_pos, n);

    let sun_diff = max(dot(n, sun_l), 0.0) * clamp(sun_l.y + 0.15, 0.0, 1.0) * shadow_vis;
    let moon_diff = max(dot(n, moon_l), 0.0) * clamp(moon_l.y + 0.10, 0.0, 1.0) * shadow_vis;
    let moon_term = (0.24 * moon_diff) * vec3<f32>(0.45, 0.62, 0.95);

    // Voxel RGB artificial/filtered light (0..15 -> 0.0..1.0) & Sunlight channel (0..15 -> 0.0..1.0)
    let rgb_light = in.light.xyz * 0.06667;
    let sun_ambient = vec3<f32>(camera.sun_dir.w) * 0.42;
    let sun_direct = camera.sun_color.rgb * (0.95 * sun_diff) + moon_term;
    let sun_light = (in.light.w * 0.06667) * (sun_ambient + sun_direct);

    // Unfading night Ambient Floor (`vec3(0.04, 0.06, 0.12)`)
    let ambient_floor = vec3<f32>(0.04, 0.06, 0.12);
    let final_light = (max(rgb_light, sun_light) + ambient_floor) * face_factor * ao_factor;

    let clamped_srgb = min(in.color, vec3<f32>(1.0));
    let albedo = pow(clamped_srgb, vec3<f32>(2.2));

    // Task 3.1: Emissive blocks (Lava, Torch, Fire, active Alloy Smelter / Furnace / Generator)
    // output HDR radiance `base_color * emission_intensity` (from 2.0 to 6.0) for Bloom!
    let peak = max(in.color.r, max(in.color.g, in.color.b));
    if peak > 1.02 {
        let emission_intensity = 2.2 + clamp((peak - 1.0) * 3.8, 0.0, 3.8); // 2.2 .. 6.0
        return albedo * emission_intensity;
    }

    return albedo * final_light;
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
    let is_water = abs(in.ao_and_flags.y - 1.0) < 0.25;

    // ── Task 5: Dynamic Water (Fresnel Reflection + Depth Fade + Shoreline Foam) ──
    if is_water {
        let n = normalize(in.normal);
        let view_dir = normalize(camera.cam_pos.xyz - in.world_pos);
        let ndotv = clamp(dot(view_dir, n), 0.0, 1.0);
        // Task 5.2: Fresnel `pow(1.0 - dot(view_dir, normal), 3.0)`
        let fresnel = pow(1.0 - ndotv, 3.0);

        // Task 5.3: Depth Fade against opaque riverbed in `scene_depth_tex`
        let dims = vec2<i32>(textureDimensions(scene_depth_tex));
        let px = clamp(vec2<i32>(in.clip_position.xy), vec2<i32>(0, 0), dims - vec2<i32>(1, 1));
        let bg_depth = textureLoad(scene_depth_tex, px, 0);
        let bg_dist = linearize_depth(bg_depth);
        let water_dist = linearize_depth(in.clip_position.z);
        let depth_diff = max(bg_dist - water_dist, 0.0);

        // Shallow water near shore is crystal clear; deeper than 3 blocks becomes rich deep ocean blue
        let depth_factor = clamp(depth_diff / 3.2, 0.0, 1.0);
        let shallow_col = vec3<f32>(0.06, 0.34, 0.58);
        let deep_col    = vec3<f32>(0.012, 0.055, 0.20);
        var water_albedo = mix(shallow_col, deep_col, depth_factor);

        // Lighting on water
        let shadow_vis = sample_sun_shadow(in.world_pos, n);
        let sun_l = normalize(camera.sun_dir.xyz);
        let sun_diff = max(dot(n, sun_l), 0.0) * shadow_vis;
        let light_scale = max(in.light.xyz * 0.06667, vec3<f32>(in.light.w * 0.06667) * (camera.sun_color.rgb * (0.4 + 0.6 * sun_diff) + vec3<f32>(camera.sun_dir.w) * 0.4))
            + vec3<f32>(0.05, 0.07, 0.14);
        var water_col = water_albedo * light_scale;

        // Sky & solar specular reflection via Fresnel
        let reflect_dir = reflect(-view_dir, n);
        let sun_spec = pow(max(dot(reflect_dir, sun_l), 0.0), 48.0) * shadow_vis;
        let sky_reflect = camera.sky_color.rgb * 1.15 + camera.sun_color.rgb * sun_spec * 2.4;
        water_col = mix(water_col, sky_reflect, clamp(fresnel * 0.78, 0.0, 0.92));

        // Thin white shoreline foam where `depth_diff` is small (`0.0 .. 0.22` blocks)
        let foam = (1.0 - smoothstep(0.02, 0.22, depth_diff)) * step(0.001, depth_diff);
        let foam_col = vec3<f32>(0.92, 0.96, 1.00) * max(light_scale, vec3<f32>(0.35));
        water_col = mix(water_col, foam_col, foam * 0.82);

        // Alpha: crystal-transparent near shore & straight-down view, opaque in deep water or grazing Fresnel
        let base_alpha = mix(0.28, 0.88, depth_factor);
        let alpha = clamp(max(base_alpha, fresnel * 0.85 + foam * 0.75), 0.25, 0.94);
        return vec4<f32>(water_col, alpha);
    }

    let lit = shade_voxel(in);
    let peak = max(in.color.r, max(in.color.g, in.color.b));
    let alpha = select(0.60, 0.86, peak > 1.05);
    var final_color = vec4<f32>(lit, alpha);
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
/// and the opaque scene depth texture copy (used for Water Depth Fade & Shoreline Foam).
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
