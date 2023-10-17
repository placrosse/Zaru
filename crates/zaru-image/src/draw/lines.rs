//! Instanced line rendering.

use wgpu::*;

use crate::{Gpu, Image};

pub fn create_pipeline(device: &Device) -> RenderPipeline {
    let line_shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("line_shader"),
        source: ShaderSource::Wgsl(include_str!("lines.wgsl").into()),
    });
    device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("line_pipe"),
        layout: None,
        vertex: VertexState {
            module: &line_shader,
            entry_point: "vertex",
            buffers: &[VertexBufferLayout {
                array_stride: 4 * 6,
                step_mode: VertexStepMode::Vertex,
                attributes: &[
                    VertexAttribute {
                        format: VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    },
                    VertexAttribute {
                        format: VertexFormat::Float32x4,
                        offset: 4 * 2,
                        shader_location: 1,
                    },
                ],
            }],
        },
        primitive: PrimitiveState {
            topology: PrimitiveTopology::LineList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            module: &line_shader,
            entry_point: "fragment",
            targets: &[Some(TextureFormat::Rgba8UnormSrgb.into())],
        }),
        multiview: None,
    })
}

#[derive(Clone, Copy, bytemuck::NoUninit)]
#[repr(C)]
pub struct Point {
    pub position: [f32; 2],
    pub color: [f32; 4],
}

pub fn draw(gpu: &Gpu, dest: &mut Image, points: &[Point]) {
    let buffer = gpu.vertex_buffer(&points);

    let mut enc = gpu.device().create_command_encoder(&Default::default());
    let mut pass = enc.begin_render_pass(&RenderPassDescriptor {
        color_attachments: &[Some(RenderPassColorAttachment {
            view: &dest.view,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Load,
                store: true,
            },
        })],
        ..Default::default()
    });
    pass.set_pipeline(&gpu.line_pipe);
    pass.set_vertex_buffer(0, buffer.slice(..));
    pass.draw(0..points.len() as u32, 0..1);
    drop(pass);
    gpu.queue().submit([enc.finish()]);
}
