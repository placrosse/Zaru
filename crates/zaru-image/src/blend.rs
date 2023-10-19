use std::thread;

use wgpu::{
    Device, FragmentState, LoadOp, MultisampleState, Operations, PrimitiveState, PrimitiveTopology,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor,
    ShaderModuleDescriptor, ShaderSource, TextureFormat, VertexAttribute, VertexBufferLayout,
    VertexFormat, VertexState, VertexStepMode,
};
use zaru_linalg::Vec2f;

use crate::{AsImageView, AsImageViewMut, Gpu, ImageView, ImageViewMut};

/// Blends `src` onto `dest`.
///
/// If `src` is larger than `dest`, it will be scaled down. If it is larger than `dest` it will be
/// scaled up. Both operations use linear filtering, but no true downscaling is attempted.
///
/// In order to choose the source or destination rectangle to blend from or onto, the `view` family
/// of methods can be used prior to calling [`blend`].
///
/// Returns a [`BlendOp`] that will perform the operation when dropped.
#[doc(alias = "blit")]
pub fn blend<'a, S, D>(dest: &'a mut D, src: &'a S) -> BlendOp<'a>
where
    D: AsImageViewMut,
    S: AsImageView,
{
    BlendOp {
        dest: dest.as_view_mut(),
        src: src.as_view(),
    }
}

// TODO: allow customizing the blend mode and fragment shader to use (make it an actual blend)

/// A blend operation from a source to a destination image.
///
/// Returned by [`blend`].
pub struct BlendOp<'a> {
    dest: ImageViewMut<'a>,
    src: ImageView<'a>,
}

impl<'a> Drop for BlendOp<'a> {
    fn drop(&mut self) {
        if thread::panicking() {
            return;
        }

        let gpu = Gpu::get();

        // Compute vertex positions and UVs
        let positions = self.dest.data.clip_corners(self.dest.image);
        let uvs = self.src.data.uvs(self.src.image);
        let corners = [
            // triangle strip
            Vertex {
                position: positions[0],
                uv: uvs[0],
            },
            Vertex {
                position: positions[2],
                uv: uvs[2],
            },
            Vertex {
                position: positions[1],
                uv: uvs[1],
            },
            Vertex {
                position: positions[3],
                uv: uvs[3],
            },
        ];

        let buffer = gpu.vertex_buffer(&corners);

        let mut enc = gpu.device().create_command_encoder(&Default::default());
        let mut pass = enc.begin_render_pass(&RenderPassDescriptor {
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &self.dest.image.view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Load,
                    store: true,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&gpu.blend_pipeline);
        pass.set_bind_group(0, &gpu.bg_linear_sampler, &[]);
        pass.set_bind_group(1, &self.src.image.texture_bind_group, &[]);
        pass.set_vertex_buffer(0, buffer.slice(..));
        pass.draw(0..4, 0..1);
        drop(pass);
        gpu.queue().submit([enc.finish()]);
    }
}

pub(crate) fn create_pipeline(device: &Device) -> RenderPipeline {
    let blend_shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("default_fragment_shader"),
        source: ShaderSource::Wgsl(include_str!("blend.wgsl").into()),
    });

    device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("default_render_pipeline"),
        layout: None,
        vertex: VertexState {
            module: &blend_shader,
            entry_point: "vertex",
            buffers: &[VertexBufferLayout {
                // FIXME use a helper to make this simpler to construct
                array_stride: 16,
                step_mode: VertexStepMode::Vertex,
                attributes: &[
                    VertexAttribute {
                        format: VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    },
                    VertexAttribute {
                        format: VertexFormat::Float32x2,
                        offset: 8,
                        shader_location: 1,
                    },
                ],
            }],
        },
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            module: &blend_shader,
            entry_point: "fragment",
            targets: &[Some(TextureFormat::Rgba8UnormSrgb.into())],
        }),
        multiview: None,
    })
}

#[derive(Clone, Copy, bytemuck::NoUninit)]
#[repr(C)]
pub(crate) struct Vertex {
    pub(crate) position: Vec2f,
    pub(crate) uv: Vec2f,
}

#[cfg(test)]
mod tests {
    use crate::{rect::Rect, Color, Image};

    use super::*;

    #[test]
    fn blend_to_partial_target() {
        let mut source = Image::new((3, 3));
        source.clear(Color::from_rgba8(0xAA, 0xBB, 0xCC, 0xDD));

        let mut target = Image::new((2, 1));
        target.clear(Color::NONE);
        // Destination view only targets the 2nd pixel.
        let mut dest = target.view_mut(Rect::from_top_left(1.0, 0.0, 1.0, 1.0));

        blend(
            &mut dest,
            &source.view(Rect::from_top_left(1.0, 1.0, 1.0, 1.0)),
        );

        target.with_data(|data| {
            assert_eq!(
                data.to_vec(),
                [0x00, 0x00, 0x00, 0x00, 0xAA, 0xBB, 0xCC, 0xDD]
            );
        });
    }
}
