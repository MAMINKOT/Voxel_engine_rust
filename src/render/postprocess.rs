// HDR Framebuffer (`Rgba16Float`), Dual-Filter Kawase Bloom (1/4 resolution),
// 32-step Screen-Space Volumetric God Rays, Chromatic Aberration, Vignette,
// and ACES Filmic Tonemapping.

use crate::render::pipeline::DepthTexture;

pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

const POSTPROCESS_SHADER_SRC: &str = r#"
struct CameraUniform {
    view_proj:       mat4x4<f32>,
    inv_view_proj:   mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    sun_dir:         vec4<f32>,
    sun_color:       vec4<f32>,
    sky_color:       vec4<f32>,
    cam_pos:         vec4<f32>,
    sun_screen:      vec4<f32>, // xy = sun screen UV (0..1), z = godray intensity, w = aspect
    render_params:   vec4<f32>, // x = shadow_quality, y = bloom_strength, z = clouds_enabled, w = water_waves
};

struct FsVarying {
    @builtin(position) clip_position: vec4<f32>,
    @location(0)       uv:            vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) idx: u32) -> FsVarying {
    let x = f32(i32(idx & 1u) * 4 - 1);
    let y = f32(i32(idx & 2u) * 2 - 1);
    var out: FsVarying;
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    // NDC (-1..1) -> UV (0..1), flipping Y for texture coordinates
    out.uv = vec2<f32>(x * 0.5 + 0.5, 0.5 - y * 0.5);
    return out;
}

// ── Bindings for Bloom passes ─────────────────────────────────────────────────
@group(0) @binding(0) var src_tex:     texture_2d<f32>;
@group(0) @binding(1) var lin_sampler: sampler;

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn bright_extract(c: vec3<f32>) -> vec3<f32> {
    let luma = luminance(c);
    if luma <= 1.0 {
        return vec3<f32>(0.0);
    }
    let excess = (luma - 1.0) / max(luma, 1e-4);
    return c * excess;
}

// Pass 1: Bright-pass extraction (`luma > 1.0`) + 5-tap Kawase downsample (1/4 resolution)
@fragment
fn fs_bloom_bright(in: FsVarying) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(src_tex));
    let texel = 1.0 / max(dims, vec2<f32>(1.0));

    let c0 = bright_extract(textureSample(src_tex, lin_sampler, in.uv).rgb);
    let c1 = bright_extract(textureSample(src_tex, lin_sampler, in.uv + vec2<f32>(-1.5, -1.5) * texel).rgb);
    let c2 = bright_extract(textureSample(src_tex, lin_sampler, in.uv + vec2<f32>( 1.5, -1.5) * texel).rgb);
    let c3 = bright_extract(textureSample(src_tex, lin_sampler, in.uv + vec2<f32>(-1.5,  1.5) * texel).rgb);
    let c4 = bright_extract(textureSample(src_tex, lin_sampler, in.uv + vec2<f32>( 1.5,  1.5) * texel).rgb);

    let avg = c0 * 0.5 + (c1 + c2 + c3 + c4) * 0.125;
    return vec4<f32>(avg, 1.0);
}

// Pass 2: First Kawase Dual-Filter Blur pass
@fragment
fn fs_bloom_blur1(in: FsVarying) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(src_tex));
    let texel = 1.5 / max(dims, vec2<f32>(1.0));

    var sum = textureSample(src_tex, lin_sampler, in.uv).rgb * 0.20;
    sum += textureSample(src_tex, lin_sampler, in.uv + vec2<f32>(-1.0, -1.0) * texel).rgb * 0.12;
    sum += textureSample(src_tex, lin_sampler, in.uv + vec2<f32>( 1.0, -1.0) * texel).rgb * 0.12;
    sum += textureSample(src_tex, lin_sampler, in.uv + vec2<f32>(-1.0,  1.0) * texel).rgb * 0.12;
    sum += textureSample(src_tex, lin_sampler, in.uv + vec2<f32>( 1.0,  1.0) * texel).rgb * 0.12;
    sum += textureSample(src_tex, lin_sampler, in.uv + vec2<f32>(-2.0,  0.0) * texel).rgb * 0.08;
    sum += textureSample(src_tex, lin_sampler, in.uv + vec2<f32>( 2.0,  0.0) * texel).rgb * 0.08;
    sum += textureSample(src_tex, lin_sampler, in.uv + vec2<f32>( 0.0, -2.0) * texel).rgb * 0.08;
    sum += textureSample(src_tex, lin_sampler, in.uv + vec2<f32>( 0.0,  2.0) * texel).rgb * 0.08;
    return vec4<f32>(sum, 1.0);
}

// Pass 3: Second wider Kawase Dual-Filter Blur pass
@fragment
fn fs_bloom_blur2(in: FsVarying) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(src_tex));
    let texel = 3.0 / max(dims, vec2<f32>(1.0));

    var sum = textureSample(src_tex, lin_sampler, in.uv).rgb * 0.20;
    sum += textureSample(src_tex, lin_sampler, in.uv + vec2<f32>(-1.0, -1.0) * texel).rgb * 0.12;
    sum += textureSample(src_tex, lin_sampler, in.uv + vec2<f32>( 1.0, -1.0) * texel).rgb * 0.12;
    sum += textureSample(src_tex, lin_sampler, in.uv + vec2<f32>(-1.0,  1.0) * texel).rgb * 0.12;
    sum += textureSample(src_tex, lin_sampler, in.uv + vec2<f32>( 1.0,  1.0) * texel).rgb * 0.12;
    sum += textureSample(src_tex, lin_sampler, in.uv + vec2<f32>(-2.0,  0.0) * texel).rgb * 0.08;
    sum += textureSample(src_tex, lin_sampler, in.uv + vec2<f32>( 2.0,  0.0) * texel).rgb * 0.08;
    sum += textureSample(src_tex, lin_sampler, in.uv + vec2<f32>( 0.0, -2.0) * texel).rgb * 0.08;
    sum += textureSample(src_tex, lin_sampler, in.uv + vec2<f32>( 0.0,  2.0) * texel).rgb * 0.08;
    return vec4<f32>(sum, 1.0);
}

// ── Composite Pass Bindings ───────────────────────────────────────────────────
@group(0) @binding(2) var bloom_tex:   texture_2d<f32>;
@group(0) @binding(3) var depth_tex:   texture_depth_2d;
@group(0) @binding(4) var<uniform> camera: CameraUniform;

fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// 32-step Screen-Space Volumetric God Rays (Radial Blur against Depth Buffer)
fn compute_god_rays(uv: vec2<f32>) -> vec3<f32> {
    let sun_vis = camera.sun_screen.z;
    if sun_vis <= 0.001 {
        return vec3<f32>(0.0);
    }

    let sun_uv = camera.sun_screen.xy;
    let aspect = max(camera.sun_screen.w, 0.1);
    let diff_aspect = vec2<f32>((uv.x - sun_uv.x) * aspect, uv.y - sun_uv.y);
    let radial_falloff = exp(-dot(diff_aspect, diff_aspect) * 2.2);
    if radial_falloff <= 0.005 {
        return vec3<f32>(0.0);
    }

    let depth_dims = vec2<f32>(textureDimensions(depth_tex));
    let max_coord = vec2<i32>(depth_dims) - vec2<i32>(1, 1);

    let step_uv = (sun_uv - uv) * (1.0 / 32.0);
    var cur_uv = uv;
    var accum = 0.0;
    var weight = 1.0;

    for (var i = 0; i < 32; i = i + 1) {
        cur_uv += step_uv;
        if cur_uv.x >= 0.0 && cur_uv.x <= 1.0 && cur_uv.y >= 0.0 && cur_uv.y <= 1.0 {
            let px = clamp(vec2<i32>(cur_uv * depth_dims), vec2<i32>(0, 0), max_coord);
            let d = textureLoad(depth_tex, px, 0);
            // Sky background has depth >= 0.9994; terrain/trees/leaves have depth < 0.9994
            if d >= 0.9994 {
                accum += weight;
            }
        }
        weight *= 0.955;
    }

    let ray_strength = (accum / 32.0) * radial_falloff * sun_vis;
    // Shift God Rays from warm golden at noon to fiery orange-gold at sunrise/sunset
    let horizon_factor = camera.sky_color.w;
    let ray_tint = mix(vec3<f32>(1.00, 0.86, 0.52), vec3<f32>(1.00, 0.45, 0.16), horizon_factor);
    return ray_tint * ray_strength * 0.75;
}

@fragment
fn fs_composite(in: FsVarying) -> @location(0) vec4<f32> {
    // Exact 1:1 integer pixel fetch from the full-resolution HDR framebuffer (zero subpixel blur!)
    let dims = vec2<i32>(textureDimensions(src_tex));
    let px = clamp(vec2<i32>(in.clip_position.xy), vec2<i32>(0, 0), dims - vec2<i32>(1, 1));
    let scene_color = textureLoad(src_tex, px, 0).rgb + compute_god_rays(in.uv);

    let bloom_color = textureSample(bloom_tex, lin_sampler, in.uv).rgb;
    let bloom_strength = camera.render_params.y;
    let final_color = aces_tonemap(scene_color + bloom_color * bloom_strength);

    return vec4<f32>(final_color, 1.0);
}
"#;

pub struct PostProcessor {
    #[allow(dead_code)]
    pub hdr_texture: wgpu::Texture,
    pub hdr_view: wgpu::TextureView,
    #[allow(dead_code)]
    pub bloom_a_tex: wgpu::Texture,
    pub bloom_a_view: wgpu::TextureView,
    #[allow(dead_code)]
    pub bloom_b_tex: wgpu::Texture,
    pub bloom_b_view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,

    bloom_bgl: wgpu::BindGroupLayout,
    composite_bgl: wgpu::BindGroupLayout,

    pub bloom_bright_bg: wgpu::BindGroup,
    pub bloom_blur1_bg: wgpu::BindGroup,
    pub bloom_blur2_bg: wgpu::BindGroup,
    pub composite_bg: wgpu::BindGroup,

    pub bloom_bright_pipeline: wgpu::RenderPipeline,
    pub bloom_blur1_pipeline: wgpu::RenderPipeline,
    pub bloom_blur2_pipeline: wgpu::RenderPipeline,
    pub composite_pipeline: wgpu::RenderPipeline,
}

impl PostProcessor {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        scene_depth_copy: &DepthTexture,
        camera_buffer: &wgpu::Buffer,
    ) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("postprocess_linear_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bloom_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Depth,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("postprocess_shader"),
            source: wgpu::ShaderSource::Wgsl(POSTPROCESS_SHADER_SRC.into()),
        });

        let bloom_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bloom_pipeline_layout"),
            bind_group_layouts: &[&bloom_bgl],
            push_constant_ranges: &[],
        });

        let make_bloom_pipe = |entry: &str, label: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&bloom_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_fullscreen",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: entry,
                    targets: &[Some(wgpu::ColorTargetState {
                        format: HDR_FORMAT,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            })
        };

        let bloom_bright_pipeline = make_bloom_pipe("fs_bloom_bright", "bloom_bright_pipeline");
        let bloom_blur1_pipeline = make_bloom_pipe("fs_bloom_blur1", "bloom_blur1_pipeline");
        let bloom_blur2_pipeline = make_bloom_pipe("fs_bloom_blur2", "bloom_blur2_pipeline");

        let comp_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("composite_pipeline_layout"),
            bind_group_layouts: &[&composite_bgl],
            push_constant_ranges: &[],
        });

        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("composite_tonemap_pipeline"),
            layout: Some(&comp_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_fullscreen",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_composite",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let (
            hdr_texture,
            hdr_view,
            bloom_a_tex,
            bloom_a_view,
            bloom_b_tex,
            bloom_b_view,
            bloom_bright_bg,
            bloom_blur1_bg,
            bloom_blur2_bg,
            composite_bg,
        ) = Self::create_textures_and_bind_groups(
            device,
            width,
            height,
            &sampler,
            &bloom_bgl,
            &composite_bgl,
            scene_depth_copy,
            camera_buffer,
        );

        Self {
            hdr_texture,
            hdr_view,
            bloom_a_tex,
            bloom_a_view,
            bloom_b_tex,
            bloom_b_view,
            sampler,
            bloom_bgl,
            composite_bgl,
            bloom_bright_bg,
            bloom_blur1_bg,
            bloom_blur2_bg,
            composite_bg,
            bloom_bright_pipeline,
            bloom_blur1_pipeline,
            bloom_blur2_pipeline,
            composite_pipeline,
        }
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn create_textures_and_bind_groups(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        sampler: &wgpu::Sampler,
        bloom_bgl: &wgpu::BindGroupLayout,
        composite_bgl: &wgpu::BindGroupLayout,
        scene_depth_copy: &DepthTexture,
        camera_buffer: &wgpu::Buffer,
    ) -> (
        wgpu::Texture,
        wgpu::TextureView,
        wgpu::Texture,
        wgpu::TextureView,
        wgpu::Texture,
        wgpu::TextureView,
        wgpu::BindGroup,
        wgpu::BindGroup,
        wgpu::BindGroup,
        wgpu::BindGroup,
    ) {
        let make_hdr_tex = |w: u32, h: u32, label: &str| {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w.max(1),
                    height: h.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: HDR_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            (tex, view)
        };

        let (hdr_texture, hdr_view) = make_hdr_tex(width, height, "hdr_framebuffer");
        let bw = (width / 4).max(1);
        let bh = (height / 4).max(1);
        let (bloom_a_tex, bloom_a_view) = make_hdr_tex(bw, bh, "bloom_quarter_a");
        let (bloom_b_tex, bloom_b_view) = make_hdr_tex(bw, bh, "bloom_quarter_b");

        let make_bloom_bg = |view: &wgpu::TextureView, label: &str| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: bloom_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            })
        };

        let bloom_bright_bg = make_bloom_bg(&hdr_view, "bloom_bright_bg");
        let bloom_blur1_bg = make_bloom_bg(&bloom_a_view, "bloom_blur1_bg");
        let bloom_blur2_bg = make_bloom_bg(&bloom_b_view, "bloom_blur2_bg");

        let composite_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite_bg"),
            layout: composite_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&bloom_a_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&scene_depth_copy.view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: camera_buffer.as_entire_binding(),
                },
            ],
        });

        (
            hdr_texture,
            hdr_view,
            bloom_a_tex,
            bloom_a_view,
            bloom_b_tex,
            bloom_b_view,
            bloom_bright_bg,
            bloom_blur1_bg,
            bloom_blur2_bg,
            composite_bg,
        )
    }

    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        scene_depth_copy: &DepthTexture,
        camera_buffer: &wgpu::Buffer,
    ) {
        let (
            hdr_texture,
            hdr_view,
            bloom_a_tex,
            bloom_a_view,
            bloom_b_tex,
            bloom_b_view,
            bloom_bright_bg,
            bloom_blur1_bg,
            bloom_blur2_bg,
            composite_bg,
        ) = Self::create_textures_and_bind_groups(
            device,
            width,
            height,
            &self.sampler,
            &self.bloom_bgl,
            &self.composite_bgl,
            scene_depth_copy,
            camera_buffer,
        );
        self.hdr_texture = hdr_texture;
        self.hdr_view = hdr_view;
        self.bloom_a_tex = bloom_a_tex;
        self.bloom_a_view = bloom_a_view;
        self.bloom_b_tex = bloom_b_tex;
        self.bloom_b_view = bloom_b_view;
        self.bloom_bright_bg = bloom_bright_bg;
        self.bloom_blur1_bg = bloom_blur1_bg;
        self.bloom_blur2_bg = bloom_blur2_bg;
        self.composite_bg = composite_bg;
    }

    /// Executes the 3 Bloom passes (`Bright Downsample` -> `Blur 1` -> `Blur 2`) and the
    /// final `Composite + God Rays + ACES Tonemap` pass onto `surface_view`.
    pub fn execute(&self, encoder: &mut wgpu::CommandEncoder, surface_view: &wgpu::TextureView) {
        // 1. Bright-pass downsample into `bloom_a_view`
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom_bright_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_a_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bloom_bright_pipeline);
            pass.set_bind_group(0, &self.bloom_bright_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // 2. Kawase Blur 1 (`bloom_a` -> `bloom_b`)
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom_blur1_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_b_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bloom_blur1_pipeline);
            pass.set_bind_group(0, &self.bloom_blur1_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // 3. Kawase Blur 2 (`bloom_b` -> `bloom_a`)
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom_blur2_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_a_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.bloom_blur2_pipeline);
            pass.set_bind_group(0, &self.bloom_blur2_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        // 4. Composite + Volumetric God Rays + ACES Filmic Tonemap onto swapchain `surface_view`
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("composite_tonemap_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: surface_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &self.composite_bg, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}
