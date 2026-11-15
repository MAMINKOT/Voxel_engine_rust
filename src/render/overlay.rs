// Overlay rendering: wireframe box around the targeted block + screen-space crosshair.
//
// * Wireframe: LineList pipeline sharing the camera bind group. The 24 line vertices are
//   rewritten every frame (cheap: 288 bytes).
// * Crosshair: three small quads in NDC (rebuilt on resize so arms stay 10 px). The blend
//   state inverts the background colour, so it is visible on any terrain.

use glam::Vec3;
use wgpu::util::DeviceExt;

use crate::render::pipeline::DEPTH_FORMAT;

const WIRE_SHADER: &str = r#"
struct CameraUniform {
    view_proj:     mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    sun_dir:       vec4<f32>,
    sun_color:     vec4<f32>,
    sky_color:     vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return camera.view_proj * vec4<f32>(position, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
"#;

const CROSSHAIR_SHADER: &str = r#"
@vertex
fn vs_main(@location(0) position: vec2<f32>) -> @builtin(position) vec4<f32> {
    return vec4<f32>(position, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
"#;

const WIRE_VERTICES: usize = 24;
const CROSSHAIR_VERTICES: usize = 18;
const CROSSHAIR_ARM_PX: f32 = 10.0;
const CROSSHAIR_HALF_THICKNESS_PX: f32 = 1.0;
/// The box is inflated slightly so its lines don't z-fight with the block faces.
const WIRE_INFLATE: f32 = 0.002;

pub struct OverlayRenderer {
    wire_pipeline: wgpu::RenderPipeline,
    wire_buffer: wgpu::Buffer,
    wire_visible: bool,
    crosshair_pipeline: wgpu::RenderPipeline,
    crosshair_buffer: wgpu::Buffer,
}

fn wire_vertices(bmin: Vec3, bmax: Vec3) -> Vec<[f32; 3]> {
    let min = bmin - Vec3::splat(WIRE_INFLATE);
    let max = bmax + Vec3::splat(WIRE_INFLATE);
    // Corner i: bit 0 -> x, bit 1 -> y, bit 2 -> z (set = max).
    let corner = |i: usize| {
        [
            if i & 1 != 0 { max.x } else { min.x },
            if i & 2 != 0 { max.y } else { min.y },
            if i & 4 != 0 { max.z } else { min.z },
        ]
    };
    let mut v = Vec::with_capacity(WIRE_VERTICES);
    for i in 0..8usize {
        for bit in [1usize, 2, 4] {
            if i & bit == 0 {
                v.push(corner(i));
                v.push(corner(i | bit));
            }
        }
    }
    v
}

fn crosshair_vertices(width: u32, height: u32) -> Vec<[f32; 2]> {
    let sx = 2.0 / width.max(1) as f32; // NDC units per pixel
    let sy = 2.0 / height.max(1) as f32;
    let (arm_x, arm_y) = (CROSSHAIR_ARM_PX * sx, CROSSHAIR_ARM_PX * sy);
    let (th_x, th_y) = (CROSSHAIR_HALF_THICKNESS_PX * sx, CROSSHAIR_HALF_THICKNESS_PX * sy);

    let mut v = Vec::with_capacity(CROSSHAIR_VERTICES);
    let mut rect = |x0: f32, y0: f32, x1: f32, y1: f32| {
        v.extend_from_slice(&[[x0, y0], [x1, y0], [x1, y1], [x0, y0], [x1, y1], [x0, y1]]);
    };
    rect(-arm_x, -th_y, arm_x, th_y); // horizontal bar
    rect(-th_x, th_y, th_x, arm_y); // vertical, upper arm
    rect(-th_x, -arm_y, th_x, -th_y); // vertical, lower arm (no overlap => no double invert)
    v
}

impl OverlayRenderer {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera_bgl: &wgpu::BindGroupLayout,
        width: u32,
        height: u32,
    ) -> Self {
        // ── Wireframe pipeline ────────────────────────────────────────────
        let wire_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wire_shader"),
            source: wgpu::ShaderSource::Wgsl(WIRE_SHADER.into()),
        });
        let wire_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("wire_pipeline_layout"),
            bind_group_layouts: &[camera_bgl],
            push_constant_ranges: &[],
        });
        let wire_attribs = wgpu::vertex_attr_array![0 => Float32x3];
        let wire_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("wire_pipeline"),
            layout: Some(&wire_layout),
            vertex: wgpu::VertexState {
                module: &wire_shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 12,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wire_attribs,
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &wire_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
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
        let wire_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wire_vertex_buffer"),
            size: (WIRE_VERTICES * std::mem::size_of::<[f32; 3]>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Crosshair pipeline ────────────────────────────────────────────
        let cross_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("crosshair_shader"),
            source: wgpu::ShaderSource::Wgsl(CROSSHAIR_SHADER.into()),
        });
        let cross_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("crosshair_pipeline_layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });
        let cross_attribs = wgpu::vertex_attr_array![0 => Float32x2];
        let invert = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::OneMinusDst,
                dst_factor: wgpu::BlendFactor::Zero,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        };
        let crosshair_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("crosshair_pipeline"),
            layout: Some(&cross_layout),
            vertex: wgpu::VertexState {
                module: &cross_shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &cross_attribs,
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &cross_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(invert),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });
        let crosshair_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("crosshair_vertex_buffer"),
            contents: bytemuck::cast_slice(&crosshair_vertices(width, height)),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        Self { wire_pipeline, wire_buffer, wire_visible: false, crosshair_pipeline, crosshair_buffer }
    }

    /// Sets (or hides) the highlighted block bounding box.
    pub fn update_wire(&mut self, queue: &wgpu::Queue, target: Option<(Vec3, Vec3)>) {
        self.wire_visible = target.is_some();
        if let Some((bmin, bmax)) = target {
            queue.write_buffer(&self.wire_buffer, 0, bytemuck::cast_slice(&wire_vertices(bmin, bmax)));
        }
    }

    pub fn resize(&self, queue: &wgpu::Queue, width: u32, height: u32) {
        queue.write_buffer(
            &self.crosshair_buffer,
            0,
            bytemuck::cast_slice(&crosshair_vertices(width, height)),
        );
    }

    /// Expects the camera bind group to be set at slot 0 already.
    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if self.wire_visible {
            pass.set_pipeline(&self.wire_pipeline);
            pass.set_vertex_buffer(0, self.wire_buffer.slice(..));
            pass.draw(0..WIRE_VERTICES as u32, 0..1);
        }
        pass.set_pipeline(&self.crosshair_pipeline);
        pass.set_vertex_buffer(0, self.crosshair_buffer.slice(..));
        pass.draw(0..CROSSHAIR_VERTICES as u32, 0..1);
    }
}
