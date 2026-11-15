// Render pipeline module.
//
// Creates:
//   • Sky background + 3D Sun/Moon Billboard Quads pipeline (`create_sky_pipeline`).
//   • Opaque voxel pipeline (`create_render_pipeline`) with depth write + backface culling.
//   • Translucent voxel pipeline (`create_translucent_pipeline`) with alpha blending,
//     `depth_write_enabled: false`, `depth_compare: LessEqual`, and `cull_mode: None`.
//   • Depth texture & camera bind group helpers.

use wgpu::util::DeviceExt;

// ─── Inline WGSL Shaders ─────────────────────────────────────────────────────

const SKY_SHADER_SRC: &str = r#"
struct CameraUniform {
    view_proj:     mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    sun_dir:       vec4<f32>,
    sun_color:     vec4<f32>,
    sky_color:     vec4<f32>,
    cam_pos:       vec4<f32>,
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
        // Enlarge visible solar disc on sunrise/sunset (`horizon_factor` -> 1.0)
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

    // 1. 3D Sun Billboard Quad (`quad_id == 1`)
    if in.quad_id == 1u {
        let r = length(in.uv);
        let box_d = max(abs(in.uv.x), abs(in.uv.y));
        if r > 1.0 {
            discard;
        }
        // Zenith: bright white core + golden corona; Sunrise/Sunset: blood-red / deep orange
        let core_zenith  = vec3<f32>(1.00, 0.99, 0.92);
        let core_horizon = vec3<f32>(1.00, 0.26, 0.06);
        let glow_zenith  = vec3<f32>(1.00, 0.82, 0.35);
        let glow_horizon = vec3<f32>(0.92, 0.14, 0.02);

        let core_col = mix(core_zenith, core_horizon, horizon_factor);
        let glow_col = mix(glow_zenith, glow_horizon, horizon_factor);

        // Voxel-styled inner solar quad (`box_d < 0.42`) surrounded by a smooth radial corona
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
        let core_col = vec3<f32>(0.92, 0.96, 1.00);
        let halo_col = vec3<f32>(0.48, 0.64, 0.95);
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
    var col = camera.sky_color.rgb * mix(1.14, 0.86, horizon_mix);

    // Subtle atmospheric horizon glow towards the sun on sunrise/sunset
    let sun_direction = normalize(camera.sun_dir.xyz);
    let sun_dot = max(dot(ray_dir, sun_direction), 0.0);
    let sunset_haze = pow(sun_dot, 6.0) * horizon_factor * 0.35;
    col = col + vec3<f32>(0.95, 0.25, 0.05) * sunset_haze;

    var final_color = vec4<f32>(col, 1.0);
    // Underwater fog tint when camera eye is inside water (`sun_color.w > 0.5`)
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
    view_proj:     mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    sun_dir:       vec4<f32>,
    sun_color:     vec4<f32>,
    sky_color:     vec4<f32>,
    cam_pos:       vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

// ── Vertex stage ──────────────────────────────────────────────────────────────
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) color:    vec3<f32>,
    @location(3) light:    vec4<f32>, // [R, G, B, Sun] in 0.0..=15.0
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0)       normal:        vec3<f32>,
    @location(1)       color:         vec3<f32>,
    @location(2)       light:         vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.normal        = in.normal;
    out.color         = in.color;
    out.light         = in.light;
    return out;
}

// ── Shared RGB + Sunlight voxel lighting calculation ──────────────────────────
fn shade_voxel(in: VertexOutput) -> vec3<f32> {
    let n = normalize(in.normal);
    let sun_l = normalize(camera.sun_dir.xyz);
    let moon_l = -sun_l;

    // Directional face contrast so top faces, X faces, and Z faces are always distinct even at night.
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

    // Sunlight N·L & opposite Moonlight N·L (`moon_direction = -sun_direction`)
    let sun_diff = max(dot(n, sun_l), 0.0) * clamp(sun_l.y + 0.15, 0.0, 1.0);
    let moon_diff = max(dot(n, moon_l), 0.0) * clamp(moon_l.y + 0.10, 0.0, 1.0);
    let moon_term = (0.22 * moon_diff) * vec3<f32>(0.45, 0.62, 0.95);

    // Voxel RGB artificial/filtered light (0..15 -> 0.0..1.0) & Sunlight channel (0..15 -> 0.0..1.0)
    let rgb_light = in.light.xyz * 0.06667;
    let sun_intensity = camera.sun_color.rgb * (0.38 + 0.62 * sun_diff)
        + moon_term
        + vec3<f32>(camera.sun_dir.w) * 0.35;
    let sun_light = (in.light.w * 0.06667) * sun_intensity;

    // Unfading night Ambient Floor (`vec3(0.04, 0.06, 0.12)`)
    let ambient_floor = vec3<f32>(0.04, 0.06, 0.12);
    let final_light = (max(rgb_light, sun_light) + ambient_floor) * face_factor;

    // Emissive glow for active machines, Torches, molten Lava, Fire, and flashing TNT (`color > 1.0`)
    let peak = max(in.color.r, max(in.color.g, in.color.b));
    let emissive = clamp(peak - 1.0, 0.0, 1.0);
    let light = mix(final_light, vec3<f32>(1.28), emissive);

    let clamped_srgb = min(in.color, vec3<f32>(1.0));
    let albedo = pow(clamped_srgb, vec3<f32>(2.2));
    return albedo * light;
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
    let lit = shade_voxel(in);
    let peak = max(in.color.r, max(in.color.g, in.color.b));
    // Fire (`peak > 1.0`) has alpha = 0.84; Water & Purple Stained Glass have alpha = 0.60.
    let alpha = select(0.60, 0.84, peak > 1.05);
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
    #[allow(dead_code)]
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
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
/// Position (3×f32) + Normal (3×f32) + Color (3×f32) + Light (4×f32) = 52 bytes per vertex.
pub fn vertex_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
    const ATTRIBS: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x3,
        3 => Float32x4
    ];
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<[f32; 13]>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTRIBS,
    }
}

/// Full-screen sky dome + 3D World-Space Sun & Moon Billboard Quads (`draw(0..15, 0..1)`).
pub fn create_sky_pipeline(
    device: &wgpu::Device,
    surface_format: wgpu::TextureFormat,
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
                format: surface_format,
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

/// Opaque voxel geometry pipeline (`depth_write_enabled: true`, backface culling).
pub fn create_render_pipeline(
    device: &wgpu::Device,
    surface_format: wgpu::TextureFormat,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("voxel_shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("render_pipeline_layout"),
        bind_group_layouts: &[camera_bind_group_layout],
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
                format: surface_format,
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

/// Translucent water/fire/stained-glass pipeline (`ALPHA_BLENDING`, `depth_write_enabled: false`,
/// `depth_compare: LessEqual`, `cull_mode: None` so water/glass is visible from both sides).
pub fn create_translucent_pipeline(
    device: &wgpu::Device,
    surface_format: wgpu::TextureFormat,
    camera_bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("voxel_translucent_shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("translucent_pipeline_layout"),
        bind_group_layouts: &[camera_bind_group_layout],
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
                format: surface_format,
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
