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

const SHADER_SRC: &str = include_str!("shaders/voxel.wgsl");

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
/// Position (3×f32) + Normal (3×f32) + Color (3×f32) + Light (4×f32) + AO/Flags (2×f32)
/// + UV (2×f32) + TexLayer (1×u32) = 72 bytes.
pub fn vertex_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRIBS: [wgpu::VertexAttribute; 7] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x3,
        3 => Float32x4,
        4 => Float32x2,
        5 => Float32x2,
        6 => Uint32
    ];
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<crate::voxel::mesher::Vertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTRIBS,
    }
}

/// Bind group layout `@group(1)` holding the 2048x2048 Shadow Map, comparison sampler,
/// opaque scene depth texture copy, and the 16x16xN `D2Array` block texture array + sampler.
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
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
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
    block_tex_view: &wgpu::TextureView,
    block_tex_sampler: &wgpu::Sampler,
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
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(block_tex_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(block_tex_sampler),
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
