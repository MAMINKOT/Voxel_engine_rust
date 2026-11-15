// Procedural Layered Voxel/2D Clouds at altitude `Y = 128.0..132.0` (4 blocks thick).
//
// Uses 3-octave 2D Value/Simplex noise scrolling along `X/Z` with wind:
//   `uv = (pos.xz + time * wind_speed) * 0.003`
// When `noise > 0.55`, renders a stylized fluffy white cloud with a shaded grey underside
// and casts soft moving shadows onto the terrain below.

use crate::render::pipeline::DEPTH_FORMAT;
use crate::render::postprocess::HDR_FORMAT;

const CLOUD_SHADER_SRC: &str = r#"
struct CameraUniform {
    view_proj:       mat4x4<f32>,
    inv_view_proj:   mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    sun_dir:         vec4<f32>,
    sun_color:       vec4<f32>,
    sky_color:       vec4<f32>,
    cam_pos:         vec4<f32>, // xyz = cam_pos, w = time
    sun_screen:      vec4<f32>,
    render_params:   vec4<f32>, // x = shadow_quality, y = bloom_strength, z = clouds_enabled, w = water_waves
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct CloudVarying {
    @builtin(position) clip_position: vec4<f32>,
    @location(0)       world_pos:     vec3<f32>,
    @location(1)       layer_frac:    f32,
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

// 2D hash & 3-octave Value Noise matching the ground cloud-shadow function in `pipeline.rs`
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

@vertex
fn vs_clouds(@builtin(vertex_index) idx: u32) -> CloudVarying {
    // 4 horizontal slices (24 vertices total = 4 layers × 6 vertices) spanning Y = 128.0 .. 132.0
    let layer = idx / 6u;
    let corner = quad_corner(idx % 6u);
    let layer_frac = f32(layer) / 3.0; // 0.0 at bottom (Y=128), 1.0 at top (Y=132)
    let y = 128.0 + layer_frac * 4.0;

    let radius = 420.0;
    let world_x = camera.cam_pos.x + corner.x * radius;
    let world_z = camera.cam_pos.z + corner.y * radius;
    let world_pos = vec3<f32>(world_x, y, world_z);

    var out: CloudVarying;
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.layer_frac = layer_frac;
    return out;
}

@fragment
fn fs_clouds(in: CloudVarying) -> @location(0) vec4<f32> {
    if camera.render_params.z < 0.5 {
        discard;
    }

    let time = camera.cam_pos.w;
    let wind = vec2<f32>(time * 4.5, time * 1.8);
    // Blocky/stylized voxel-cloud quantization (2m grid steps smoothed with fbm)
    let snapped_xz = floor(in.world_pos.xz * 0.5) * 2.0;
    let uv = (snapped_xz + wind) * 0.003;
    let n = cloud_fbm(uv);

    // Slightly tighter threshold near the top/bottom of the 4-block slab for a rounded profile
    let profile = 1.0 - abs(in.layer_frac * 2.0 - 1.0) * 0.06;
    let density = n * profile;
    if density <= 0.55 {
        discard;
    }

    // Distance fade towards the horizon so clouds blend seamlessly into the sky
    let dist_xz = length(in.world_pos.xz - camera.cam_pos.xz);
    let horizon_fade = 1.0 - smoothstep(260.0, 410.0, dist_xz);
    if horizon_fade <= 0.01 {
        discard;
    }

    // Shaded grey underside (`layer_frac == 0.0`) transitioning to fluffy white top (`layer_frac == 1.0`)
    let bottom_grey = vec3<f32>(0.62, 0.66, 0.74);
    let top_white   = vec3<f32>(0.98, 0.99, 1.00);
    let base_cloud  = mix(bottom_grey, top_white, clamp(in.layer_frac * 0.85 + (density - 0.55) * 1.2, 0.0, 1.0));

    // Modulate cloud brightness by day/sunset/night lighting (strictly <= 1.0 so clouds don't bloom)
    let day_light = clamp(camera.sun_color.rgb * 0.75 + vec3<f32>(camera.sun_dir.w) * 0.85, vec3<f32>(0.12, 0.14, 0.22), vec3<f32>(1.0));
    let lit_cloud = clamp(base_cloud * day_light, vec3<f32>(0.0), vec3<f32>(0.98));

    let edge_alpha = smoothstep(0.55, 0.64, density);
    let alpha = clamp(edge_alpha * 0.68 * horizon_fade, 0.0, 0.88);
    return vec4<f32>(lit_cloud, alpha);
}
"#;

pub struct CloudRenderer {
    pub pipeline: wgpu::RenderPipeline,
}

impl CloudRenderer {
    pub fn new(device: &wgpu::Device, camera_bgl: &wgpu::BindGroupLayout) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("clouds_shader"),
            source: wgpu::ShaderSource::Wgsl(CLOUD_SHADER_SRC.into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("clouds_pipeline_layout"),
            bind_group_layouts: &[camera_bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("clouds_render_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_clouds",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_clouds",
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self { pipeline }
    }

    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>, camera_bg: &'a wgpu::BindGroup) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, camera_bg, &[]);
        // 4 horizontal layers × 6 vertices per quad = 24 vertices
        pass.draw(0..24, 0..1);
    }
}
