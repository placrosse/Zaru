use egui_wgpu::renderer::ScreenDescriptor;
use epaint::{tessellate_shapes, ClippedShape, Rect, Shape, TessellationOptions};
use wgpu::{
    Operations, RenderPassColorAttachment, RenderPassDescriptor, TextureFormat,
    TextureViewDescriptor,
};

use crate::{Gpu, Image};

pub fn draw(gpu: &Gpu, dest: &mut Image, shapes: impl IntoIterator<Item = Shape>) {
    let clipped_shapes = shapes
        .into_iter()
        .map(|shape| ClippedShape {
            clip_rect: Rect::EVERYTHING,
            shape,
        })
        .collect::<Vec<_>>();

    draw_impl(gpu, dest, clipped_shapes);
}

fn draw_impl(gpu: &Gpu, dest: &mut Image, shapes: Vec<ClippedShape>) {
    let atlas = gpu.fonts.texture_atlas();
    let mut atlas = atlas.lock();
    let prim = tessellate_shapes(
        1.0,
        TessellationOptions::default(),
        atlas.size(),
        atlas.prepared_discs(),
        shapes,
    );

    let delta = atlas.take_delta();
    drop(atlas);

    let screen = ScreenDescriptor {
        size_in_pixels: [dest.width(), dest.height()],
        pixels_per_point: 1.0,
    };

    let renderer = &mut gpu.epaint.lock().unwrap();
    if let Some(delta) = delta {
        renderer.update_texture(
            gpu.device(),
            gpu.queue(),
            epaint::TextureId::Managed(0),
            &delta,
        );
    }
    let mut enc = gpu.device().create_command_encoder(&Default::default());
    let command_buffers =
        renderer.update_buffers(&gpu.device(), &gpu.queue(), &mut enc, &prim, &screen);
    let view = dest.texture.create_view(&TextureViewDescriptor {
        // egui/epaint expect to operate in linear color, so use a non-srgb view.
        format: Some(TextureFormat::Rgba8Unorm),
        ..Default::default()
    });
    let mut render_pass = enc.begin_render_pass(&RenderPassDescriptor {
        color_attachments: &[Some(RenderPassColorAttachment {
            view: &view,
            resolve_target: None,
            ops: Operations {
                load: wgpu::LoadOp::Load,
                store: true,
            },
        })],
        depth_stencil_attachment: None,
        label: None,
    });
    renderer.render(&mut render_pass, &prim, &screen);

    drop(render_pass);
    let buf = enc.finish();
    gpu.queue().submit(command_buffers.into_iter().chain([buf]));
}
