use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource, BufferDescriptor,
    BufferUsages, BufferView, Extent3d, ImageCopyBuffer, ImageDataLayout, LoadOp, Operations,
    RenderPassColorAttachment, RenderPassDescriptor, Texture, TextureDescriptor, TextureDimension,
    TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
};

use crate::{rect::Rect, Color, Gpu, Resolution};

/// An 8-bit sRGB image with alpha channel.
pub struct Image {
    pub(crate) texture: Texture,
    /// Default texture view that interprets the pixel data as non-linear sRGB (auto-converted to linear sRGB in the shader).
    pub(crate) view: TextureView,
    /// A [`BindGroup`] conforming to `bgl_single_texture` that binds the image's [`Texture`].
    pub(crate) texture_bind_group: BindGroup,
    /// A [`BindGroup`] conforming to `bgl_single_storage_texture_rgba8unorm_srgb` that binds the image's [`Texture`].
    #[allow(unused)] // TODO: compute shader interface
    storage_texture_bind_group: BindGroup,
}

impl Image {
    /// Creates a blank [`Image`] of the given dimensions.
    ///
    /// The contents of the [`Image`] are unspecified and should not be relied on.
    pub fn new(res: impl Into<Resolution>) -> Self {
        // FIXME: empty images (where width or height is 0) fail wgpu validation, should we support them?
        let res = res.into();
        Self::new_impl(res)
    }

    fn new_impl(res: Resolution) -> Self {
        let gpu = Gpu::get();
        let texture = gpu.device().create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d {
                width: res.width(),
                height: res.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::COPY_DST
                | TextureUsages::COPY_SRC,
            view_formats: &[TextureFormat::Rgba8UnormSrgb],
        });
        let view = texture.create_view(&TextureViewDescriptor {
            format: Some(TextureFormat::Rgba8UnormSrgb),
            ..Default::default()
        });
        Self {
            texture_bind_group: gpu.device().create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &gpu.bgl_single_texture,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&view),
                }],
            }),
            storage_texture_bind_group: gpu.device().create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &gpu.bgl_single_storage_texture_rgba8unorm_srgb,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&view),
                }],
            }),
            texture,
            view,
        }
    }

    /// Returns a new [`Image`] of the given size, with every pixel initialized to the given [`Color`].
    pub fn filled(res: impl Into<Resolution>, color: Color) -> Self {
        let mut this = Self::new(res);
        this.clear(color);
        this
    }

    /// Creates an [`Image`] from raw, preexisting RGBA pixel data.
    ///
    /// `buf` needs to contain data in the following interleaved pixel format:
    /// `rrrrrrrr gggggggg bbbbbbbb aaaaaaaa`. Its length needs to be exactly `width * height * 4`,
    /// or this function will panic.
    pub fn from_rgba8(res: impl Into<Resolution>, buf: &[u8]) -> Self {
        let res = res.into();
        Self::from_rgba8_impl(res, buf)
    }

    fn from_rgba8_impl(res: Resolution, buf: &[u8]) -> Self {
        let expected_size = res.width() as usize * res.height() as usize * 4;
        assert_eq!(
            expected_size,
            buf.len(),
            "incorrect buffer size {} for {} image (expected {} bytes)",
            buf.len(),
            res,
            expected_size,
        );

        let image = Image::new(res);
        Gpu::get().queue().write_texture(
            image.texture.as_image_copy(),
            buf,
            ImageDataLayout {
                bytes_per_row: Some(res.width() * 4),
                ..Default::default()
            },
            Extent3d {
                width: res.width(),
                height: res.height(),
                depth_or_array_layers: 1,
            },
        );
        image
    }

    /// Returns the width of this image, in pixels.
    #[inline]
    pub fn width(&self) -> u32 {
        self.texture.width()
    }

    /// Returns the height of this image, in pixels.
    #[inline]
    pub fn height(&self) -> u32 {
        self.texture.height()
    }

    /// Returns the size of this image.
    #[inline]
    pub fn resolution(&self) -> Resolution {
        Resolution::new(self.width(), self.height())
    }

    /// Returns a [`Rect`] covering this image.
    ///
    /// The rectangle will be positioned at `(0, 0)` and have the width and height of the image.
    #[inline]
    pub fn rect(&self) -> Rect {
        Rect::from_top_left(0.0, 0.0, self.width() as f32, self.height() as f32)
    }

    #[inline]
    fn bytes_per_pixel(&self) -> u32 {
        4
    }

    /// Sets every pixel in the [`Image`] to the given [`Color`].
    pub fn clear(&mut self, color: Color) {
        // The wgpu clear color is in linear space.
        let color = color.to_linear().map(f64::from);

        let gpu = &Gpu::get();
        let mut enc = gpu.device().create_command_encoder(&Default::default());
        let mut pass = enc.begin_render_pass(&RenderPassDescriptor {
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &self.view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(wgpu::Color {
                        r: color[0],
                        g: color[1],
                        b: color[2],
                        a: color[3],
                    }),
                    store: true, // must be `true` otherwise clearing won't work
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&gpu.clear_pipeline);
        pass.draw(0..0, 0..0);
        drop(pass);
        gpu.queue().submit([enc.finish()]);
    }

    /// Maps the [`Image`]'s pixel data into the process address space and invokes a closure with
    /// the resulting [`ImageData`].
    pub fn with_data<R>(&self, cb: impl FnOnce(ImageData<'_>) -> R) -> R {
        let gpu = &Gpu::get();
        let stride = (self.width() * self.bytes_per_pixel())
            .next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = gpu.device().create_buffer(&BufferDescriptor {
            label: Some("texture_data"),
            size: u64::from(stride) * u64::from(self.height()),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut enc = gpu.device().create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            self.texture.as_image_copy(),
            ImageCopyBuffer {
                buffer: &buffer,
                layout: ImageDataLayout {
                    bytes_per_row: Some(stride),
                    ..Default::default()
                },
            },
            Extent3d {
                width: self.width(),
                height: self.height(),
                depth_or_array_layers: 1,
            },
        );
        let index = gpu.queue().submit([enc.finish()]);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, Result::unwrap);
        gpu.device()
            .poll(wgpu::MaintainBase::WaitForSubmissionIndex(index));

        let view = buffer.slice(..).get_mapped_range();

        let width_bytes = self.width() * self.bytes_per_pixel();
        let ret = cb(ImageData {
            view,
            stride,
            width_bytes,
        });

        buffer.unmap();
        ret
    }
}

/// Mapped [`Image`] pixel data.
///
/// This is the argument of the callback passed to [`Image::with_data`].
///
/// The pixel data is exposed as a list of rows with potential gaps inbetween them.
pub struct ImageData<'a> {
    view: BufferView<'a>,
    stride: u32,
    width_bytes: u32,
}

impl<'a> ImageData<'a> {
    /// Returns an iterator over the contiguous rows of pixel data in the [`Image`].
    #[inline]
    pub fn rows(&self) -> impl Iterator<Item = &[u8]> + '_ {
        self.view
            .chunks(self.stride as usize)
            .map(|row| &row[..self.width_bytes as usize])
    }

    /// Collects all pixels into a contiguous [`Vec`] without gaps.
    pub fn to_vec(&self) -> Vec<u8> {
        self.rows().flat_map(|row| row.iter().copied()).collect()
    }

    /// Returns the number of bytes between the start of each row of pixel data.
    #[inline]
    pub fn stride(&self) -> u32 {
        self.stride
    }

    /// Returns the raw underlying bytes.
    ///
    /// Every row of image data is separated by [`self.stride()`][Self::stride()] bytes. Pixels of
    /// each row are stored in order, with all channels interleaved.
    #[inline]
    pub fn raw_bytes(&self) -> &[u8] {
        &self.view
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_access() {
        // Make an image containing 2 rows with 1 pixel each.
        let pixels = &[
            0xab, 0xcd, 0xef, 0x12, // 0
            0x32, 0x43, 0x54, 0x76, // 1
        ];
        let image = Image::from_rgba8((1, 2), pixels);
        image.with_data(|data| {
            // Rows should have been padded to a multiple of the copy alignment requirement.
            assert_eq!(data.stride, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);

            let rows = data.rows().collect::<Vec<_>>();
            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0], &pixels[0..4]);
            assert_eq!(rows[1], &pixels[4..8]);

            assert_eq!(data.to_vec(), pixels);
        });
    }

    #[test]
    fn clear() {
        let mut image = Image::new((1, 1));

        image.clear(Color::WHITE);
        image.with_data(|data| {
            assert_eq!(data.to_vec(), &[0xFF, 0xFF, 0xFF, 0xFF]);
        });

        image.clear(Color::BLACK);
        image.with_data(|data| {
            assert_eq!(data.to_vec(), &[0x00, 0x00, 0x00, 0xFF]);
        });

        image.clear(Color::NONE);
        image.with_data(|data| {
            assert_eq!(data.to_vec(), &[0x00, 0x00, 0x00, 0x00]);
        });

        image.clear(Color::BLUE);
        image.with_data(|data| {
            assert_eq!(data.to_vec(), &[0x00, 0x00, 0xFF, 0xFF]);
        });

        // Test non-linear to linear sRGB conversion.
        image.clear(Color::from_rgb8(15, 100, 170));
        image.with_data(|data| {
            assert_eq!(data.to_vec(), &[15, 100, 170, 0xFF]);
        });
    }
}
