// Directional Shadow Mapping (`2048x2048` `Depth32Float`) for dynamic sun/terrain shadows
// with 3x3 Percentage-Closer Filtering (PCF).

use glam::{Mat4, Vec3, Vec4};

use crate::render::pipeline::{vertex_buffer_layout, DEPTH_FORMAT};

pub const SHADOW_MAP_SIZE: u32 = 2048;
pub const SHADOW_ORTHO_RADIUS: f32 = 80.0;
const SHADOW_DISTANCE: f32 = 140.0;
const SHADOW_Z_NEAR: f32 = 1.0;
const SHADOW_Z_FAR: f32 = 300.0;

const SHADOW_SHADER_SRC: &str = r#"
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

struct VertexInput {
    @location(0) position:      vec3<f32>,
    @location(1) normal:        vec3<f32>,
    @location(2) color:         vec3<f32>,
    @location(3) light:         vec4<f32>,
    @location(4) ao_and_flags:  vec2<f32>,
};

@vertex
fn vs_shadow(in: VertexInput) -> @builtin(position) vec4<f32> {
    return camera.light_view_proj * vec4<f32>(in.position, 1.0);
}
"#;

/// Computes the orthographic `light_projection * light_view` matrix centered on `focus_pos`
/// with a ~80-block radius, texel-snapped to prevent shadow swimming while moving.
pub fn compute_light_view_proj(focus_pos: Vec3, sun_dir: Vec3) -> Mat4 {
    // Use sun direction when above horizon; clamp minimum elevation so grazing shadows remain stable.
    let raw_dir = if sun_dir.y >= -0.08 {
        sun_dir
    } else {
        -sun_dir
    };
    let dir = Vec3::new(raw_dir.x, raw_dir.y.max(0.18), raw_dir.z).normalize();

    let center = Vec3::new(focus_pos.x.round(), 20.0, focus_pos.z.round());
    let eye = center + dir * SHADOW_DISTANCE;
    let light_view = Mat4::look_at_rh(eye, center, Vec3::Y);

    // Texel snapping in light view space to eliminate shimmering when the player walks
    let world_units_per_texel = (SHADOW_ORTHO_RADIUS * 2.0) / SHADOW_MAP_SIZE as f32;
    let mut origin_ls = light_view * Vec4::new(0.0, 0.0, 0.0, 1.0);
    origin_ls.x /= origin_ls.w;
    origin_ls.y /= origin_ls.w;
    let snap_x = (origin_ls.x / world_units_per_texel).round() * world_units_per_texel - origin_ls.x;
    let snap_y = (origin_ls.y / world_units_per_texel).round() * world_units_per_texel - origin_ls.y;

    let r = SHADOW_ORTHO_RADIUS;
    let light_proj = Mat4::orthographic_rh(
        -r - snap_x,
        r - snap_x,
        -r - snap_y,
        r - snap_y,
        SHADOW_Z_NEAR,
        SHADOW_Z_FAR,
    );
    light_proj * light_view
}

pub struct ShadowRenderer {
    #[allow(dead_code)]
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub pipeline: wgpu::RenderPipeline,
}

impl ShadowRenderer {
    pub fn new(device: &wgpu::Device, camera_bgl: &wgpu::BindGroupLayout) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sun_shadow_map"),
            size: wgpu::Extent3d {
                width: SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sun_shadow_comparison_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADOW_SHADER_SRC.into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow_pipeline_layout"),
            bind_group_layouts: &[camera_bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow_depth_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_shadow",
                buffers: &[vertex_buffer_layout()],
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.2,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            texture,
            view,
            sampler,
            pipeline,
        }
    }
}
