//! Image manipulation.

mod draw;
mod rect;

use std::{fmt, ops::Index, path::Path};

use embedded_graphics::{pixelcolor::raw::RawU32, prelude::PixelColor};
use image::{GenericImage, GenericImageView, ImageBuffer, ImageFormat, Rgba};

use crate::resolution::Resolution;

pub use draw::*;
pub use rect::Rect;

/// Whether to use "mozjpeg"/libjpeg-turbo for JPEG decoding. It's faster than the `image` crate
/// (6-7 ms instead of ~10ms for 1080p frames).
const USE_MOZJPEG: bool = true;

type Img = ImageBuffer<Rgba<u8>, Vec<u8>>;

/// An 8-bit RGB image.
#[derive(Clone)]
pub struct Image {
    // Internal representation is meant to be compatible with wgpu's texture formats. The alpha
    // channel is otherwise unused and not exposed to the user.
    pub(crate) buf: Img,
}

impl Image {
    /// Loads an image from the filesystem.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, crate::Error> {
        let image = image::open(path)?;
        let buf = image.into_rgba8();

        Ok(Self { buf })
    }

    /// Decodes a JFIF JPEG or Motion JPEG from a byte slice.
    pub fn decode_jpeg(data: &[u8]) -> Result<Self, crate::Error> {
        let buf = if USE_MOZJPEG {
            let decompressor = mozjpeg::Decompress::new_mem(data)?;
            let mut decomp = decompressor.rgba()?;
            let buf = decomp.read_scanlines_flat().unwrap();
            let buf = ImageBuffer::from_raw(decomp.width() as u32, decomp.height() as u32, buf)
                .expect("failed to create ImageBuffer");
            buf
        } else {
            image::load_from_memory_with_format(data, ImageFormat::Jpeg)?.to_rgba8()
        };

        Ok(Self { buf })
    }

    /// Creates an empty image of a specified size.
    ///
    /// The image will start out black.
    pub fn new(width: u32, height: u32) -> Self {
        // FIXME alpha channel?
        Self {
            buf: ImageBuffer::new(width, height),
        }
    }

    /// Returns the width of this image, in pixels.
    #[inline]
    pub fn width(&self) -> u32 {
        self.buf.width()
    }

    /// Returns the height of this image, in pixels.
    #[inline]
    pub fn height(&self) -> u32 {
        self.buf.height()
    }

    /// Returns the size of this image.
    #[inline]
    pub fn resolution(&self) -> Resolution {
        Resolution::new(self.width(), self.height())
    }

    /// Resizes this image to a new size, adding black bars to keep the original aspect ratio.
    ///
    /// For performance (as this runs on the CPU), this uses nearest neighbor interpolation, so the
    /// result won't look very good, but it should suffice for most use cases.
    pub fn aspect_aware_resize(&self, new_res: Resolution) -> Image {
        self.as_view().aspect_aware_resize(new_res)
    }

    /// Gets the image color at the given pixel coordinates.
    ///
    /// # Panics
    ///
    /// This will panic if `(x, y)` is outside the bounds of this image.
    #[inline]
    pub fn get(&self, x: u32, y: u32) -> Color {
        let rgb = &self.buf[(x, y)];
        Color(rgb.0)
    }

    /// Sets the image color at the given pixel coordinates.
    ///
    /// # Panics
    ///
    /// This will panic if `(x, y)` is outside the bounds of this image.
    #[inline]
    pub fn set(&mut self, x: u32, y: u32, color: Color) {
        self.buf[(x, y)] = Rgba(color.0);
    }

    /// Creates an immutable view into an area of this image, specified by `rect`.
    ///
    /// # Panics
    ///
    /// This will panic if `rect` is not fully contained in the bounds of `self`.
    pub fn view(&self, rect: &Rect) -> ImageView<'_> {
        let my_rect = Rect::from_top_left(0, 0, self.width(), self.height());
        assert!(
            my_rect.contains_rect(rect),
            "attempted to create out-of-bounds view of {}x{} image at {:?}",
            self.width(),
            self.height(),
            rect
        );

        ImageView {
            sub_image: self
                .buf
                .view(rect.x() as _, rect.y() as _, rect.width(), rect.height()),
        }
    }

    /// Creates a mutable view into an area of this image, specified by `rect`.
    ///
    /// # Panics
    ///
    /// This will panic if `rect` is not fully contained in the bounds of `self`.
    pub fn view_mut(&mut self, rect: &Rect) -> ImageViewMut<'_> {
        let my_rect = Rect::from_top_left(0, 0, self.width(), self.height());
        assert!(
            my_rect.contains_rect(rect),
            "attempted to create out-of-bounds view of {}x{} image at {:?}",
            self.width(),
            self.height(),
            rect
        );

        ImageViewMut {
            sub_image: self.buf.sub_image(
                rect.x() as _,
                rect.y() as _,
                rect.width(),
                rect.height(),
            ),
        }
    }

    #[inline]
    pub(crate) fn data(&self) -> &[u8] {
        self.buf.as_raw()
    }
}

impl fmt::Debug for Image {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{} Image", self.width(), self.height())
    }
}

/// An immutable view of a rectangular section of an [`Image`].
pub struct ImageView<'a> {
    sub_image: image::SubImage<&'a Img>,
}

impl<'a> ImageView<'a> {
    /// Returns the width of this view, in pixels.
    pub fn width(&self) -> u32 {
        self.sub_image.width()
    }

    /// Returns the height of this view, in pixels.
    pub fn height(&self) -> u32 {
        self.sub_image.height()
    }

    /// Returns the size of this view.
    #[inline]
    pub fn resolution(&self) -> Resolution {
        Resolution::new(self.width(), self.height())
    }

    /// Gets the image color at the given pixel coordinates.
    ///
    /// # Panics
    ///
    /// This will panic if `(x, y)` is outside the bounds of this view.
    #[inline]
    pub fn get(&self, x: u32, y: u32) -> Color {
        let rgb = self.sub_image.get_pixel(x, y);
        Color(rgb.0)
    }

    /// Borrows an identical [`ImageView`] from `self` that may have a shorter lifetime.
    ///
    /// This is equivalent to the implicit "reborrowing" that happens on Rust references. It needs
    /// to be a method call here because user-defined types cannot opt into making this happen
    /// automatically.
    pub fn reborrow(&self) -> ImageView<'_> {
        ImageView {
            sub_image: self.sub_image.view(0, 0, self.width(), self.height()),
        }
    }

    /// Creates an immutable subview into an area of this view, specified by `rect`.
    ///
    /// # Panics
    ///
    /// This will panic if `rect` is not fully contained in the bounds of `self`.
    pub fn view(&self, rect: &Rect) -> ImageView<'_> {
        let my_rect = Rect::from_top_left(0, 0, self.width(), self.height());
        assert!(
            my_rect.contains_rect(rect),
            "attempted to create out-of-bounds view of {}x{} view at {:?}",
            self.width(),
            self.height(),
            rect
        );

        ImageView {
            sub_image: self.sub_image.view(
                rect.x() as _,
                rect.y() as _,
                rect.width(),
                rect.height(),
            ),
        }
    }

    /// Resizes this image to a new size, adding black bars to keep the original aspect ratio.
    ///
    /// For performance (as this runs on the CPU), this uses nearest neighbor interpolation, so the
    /// result won't look very good, but it should suffice for most use cases.
    pub fn aspect_aware_resize(&self, new_res: Resolution) -> Image {
        let cur_ratio = self.resolution().aspect_ratio();
        let new_ratio = new_res.aspect_ratio();

        log::trace!(
            "aspect-aware resize from {} -> {} ({} -> {})",
            self.resolution(),
            new_res,
            cur_ratio,
            new_ratio,
        );

        let mut out = Image {
            buf: ImageBuffer::new(new_res.width(), new_res.height()),
        };

        let target_rect = new_res.fit_aspect_ratio(self.resolution().aspect_ratio());
        let mut target_view = out.view_mut(&target_rect);

        for dest_y in 0..target_rect.height() {
            for dest_x in 0..target_rect.width() {
                let src_x = ((dest_x as f32 + 0.5) / target_rect.width() as f32
                    * self.width() as f32) as u32;
                let src_y = ((dest_y as f32 + 0.5) / target_rect.height() as f32
                    * self.height() as f32) as u32;

                let pixel = self.get(src_x, src_y);
                target_view
                    .sub_image
                    .put_pixel(dest_x, dest_y, Rgba(pixel.0));
            }
        }

        out
    }
}

impl fmt::Debug for ImageView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{} ImageView", self.width(), self.height())
    }
}

/// A mutable view of a rectangular section of an [`Image`].
pub struct ImageViewMut<'a> {
    sub_image: image::SubImage<&'a mut Img>,
}

impl<'a> ImageViewMut<'a> {
    /// Returns the width of this view, in pixels.
    pub fn width(&self) -> u32 {
        self.sub_image.width()
    }

    /// Returns the height of this view, in pixels.
    pub fn height(&self) -> u32 {
        self.sub_image.height()
    }

    /// Returns the size of this view.
    #[inline]
    pub fn resolution(&self) -> Resolution {
        Resolution::new(self.width(), self.height())
    }

    /// Gets the image color at the given pixel coordinates.
    ///
    /// # Panics
    ///
    /// This will panic if `(x, y)` is outside the bounds of this view.
    #[inline]
    pub fn get(&self, x: u32, y: u32) -> Color {
        let rgb = self.sub_image.get_pixel(x, y);
        Color(rgb.0)
    }

    /// Sets the image color at the given pixel coordinates.
    ///
    /// # Panics
    ///
    /// This will panic if `(x, y)` is outside the bounds of this view.
    #[inline]
    pub fn set(&mut self, x: u32, y: u32, color: Color) {
        self.sub_image.put_pixel(x, y, Rgba(color.0));
    }

    /// Borrows an identical [`ImageViewMut`] from `self` that may have a shorter lifetime.
    ///
    /// This is equivalent to the implicit "reborrowing" that happens on Rust references. It needs
    /// to be a method call here because user-defined types cannot opt into making this happen
    /// automatically.
    pub fn reborrow(&mut self) -> ImageViewMut<'_> {
        ImageViewMut {
            sub_image: self.sub_image.sub_image(0, 0, self.width(), self.height()),
        }
    }

    /// Creates an immutable subview into an area of this view, specified by `rect`.
    ///
    /// # Panics
    ///
    /// This will panic if `rect` is not fully contained in the bounds of `self`.
    pub fn view(&self, rect: &Rect) -> ImageView<'_> {
        let my_rect = Rect::from_top_left(0, 0, self.width(), self.height());
        assert!(
            my_rect.contains_rect(rect),
            "attempted to create out-of-bounds view of {}x{} view at {:?}",
            self.width(),
            self.height(),
            rect
        );

        ImageView {
            sub_image: self.sub_image.view(
                rect.x() as _,
                rect.y() as _,
                rect.width(),
                rect.height(),
            ),
        }
    }

    /// Creates a mutable view into an area of this view, specified by `rect`.
    ///
    /// # Panics
    ///
    /// This will panic if `rect` is not fully contained in the bounds of `self`.
    pub fn view_mut(&mut self, rect: &Rect) -> ImageViewMut<'_> {
        let my_rect = Rect::from_top_left(0, 0, self.width(), self.height());
        assert!(
            my_rect.contains_rect(rect),
            "attempted to create out-of-bounds view of {}x{} view at {:?}",
            self.width(),
            self.height(),
            rect
        );

        ImageViewMut {
            sub_image: self.sub_image.sub_image(
                rect.x() as _,
                rect.y() as _,
                rect.width(),
                rect.height(),
            ),
        }
    }
}

impl fmt::Debug for ImageViewMut<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{} ImageViewMut", self.width(), self.height())
    }
}

/// An RGB color.
#[derive(PartialEq, Eq, Clone, Copy)]
pub struct Color(pub(crate) [u8; 4]);

impl Color {
    pub const BLACK: Self = Self([0, 0, 0, 255]);
    pub const WHITE: Self = Self([255, 255, 255, 255]);
    pub const RED: Self = Self([255, 0, 0, 255]);
    pub const GREEN: Self = Self([0, 255, 0, 255]);
    pub const BLUE: Self = Self([0, 0, 255, 255]);

    #[inline]
    pub const fn from_rgb8(r: u8, g: u8, b: u8) -> Self {
        Self([r, g, b, 255])
    }

    #[inline]
    pub fn r(&self) -> u8 {
        self.0[0]
    }

    #[inline]
    pub fn g(&self) -> u8 {
        self.0[1]
    }

    #[inline]
    pub fn b(&self) -> u8 {
        self.0[2]
    }
}

// FIXME allows accessing alpha
impl Index<usize> for Color {
    type Output = u8;

    #[inline]
    fn index(&self, index: usize) -> &u8 {
        &self.0[index]
    }
}

// FIXME leaks `embedded-graphics` dependency
impl PixelColor for Color {
    type Raw = RawU32;
}

/// Trait for types that can be treated as read-only views of image data.
///
/// This allows abstracting over [`Image`] and [`ImageView`] and should be used by any code that
/// takes immutable image data as input.
pub trait AsImageView {
    /// Returns an [`ImageView`] covering `self`.
    fn as_view(&self) -> ImageView<'_>;
}

/// Trait for types that can be treated as mutable views of image data.
///
/// This allows abstracting over [`Image`] and [`ImageViewMut`] and should be used by any code that
/// writes to image data.
pub trait AsImageViewMut: AsImageView {
    /// Returns an [`ImageViewMut`] covering `self`.
    fn as_view_mut(&mut self) -> ImageViewMut<'_>;
}

impl AsImageView for Image {
    fn as_view(&self) -> ImageView<'_> {
        self.view(&Rect::from_top_left(0, 0, self.width(), self.height()))
    }
}

impl<'a> AsImageView for ImageView<'a> {
    fn as_view(&self) -> ImageView<'_> {
        self.reborrow()
    }
}

impl AsImageViewMut for Image {
    fn as_view_mut(&mut self) -> ImageViewMut<'_> {
        self.view_mut(&Rect::from_top_left(0, 0, self.width(), self.height()))
    }
}

impl<'a> AsImageView for ImageViewMut<'a> {
    fn as_view(&self) -> ImageView<'_> {
        ImageView {
            sub_image: self.sub_image.view(0, 0, self.width(), self.height()),
        }
    }
}

impl<'a> AsImageViewMut for ImageViewMut<'a> {
    fn as_view_mut(&mut self) -> ImageViewMut<'_> {
        self.reborrow()
    }
}
