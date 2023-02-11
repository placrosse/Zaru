//! Image manipulation.
//!
//! This module provides:
//!
//! - The [`Image`] type, an owned RGBA image.
//! - [`ImageView`] and [`ImageViewMut`], borrowed rectangular views into an underlying [`Image`].
//! - The [`AsImageView`] and [`AsImageViewMut`] traits to abstract over images and views.
//! - A variety of [`draw`] functions to quickly visualize objects.
//! - [`Rect`] and [`RotatedRect`], integer-valued rectangles representing parts of an image.

mod blend;
pub mod draw;
mod jpeg;
mod resolution;

#[cfg(test)]
mod tests;

use std::{fmt, ops::Index, path::Path};

use embedded_graphics::{pixelcolor::raw::RawU32, prelude::PixelColor};
use image::{GenericImage, GenericImageView, ImageBuffer, Rgba, RgbaImage};

pub use blend::*;
pub use resolution::*;

use crate::rect::{Rect, RotatedRect};

#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
enum ImageFormat {
    Jpeg,
    Png,
}

impl ImageFormat {
    fn from_path(path: &Path) -> anyhow::Result<Self> {
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("jpg" | "jpeg") => Ok(Self::Jpeg),
            Some("png") => Ok(Self::Png),
            _ => anyhow::bail!(
                "invalid image path '{}' (must have one of the supported extensions)",
                path.display()
            ),
        }
    }
}

/// An 8-bit sRGB image with alpha channel.
#[derive(Clone)]
pub struct Image {
    // Internal representation is meant to be compatible with wgpu's texture formats for easy GPU
    // up/downloading.
    pub(crate) buf: RgbaImage,
}

impl Image {
    /// Loads an image from the filesystem.
    ///
    /// The path must have a supported file extension (`jpeg`, `jpg` or `png`).
    pub fn load<A: AsRef<Path>>(path: A) -> anyhow::Result<Self> {
        Self::load_impl(path.as_ref())
    }

    fn load_impl(path: &Path) -> anyhow::Result<Self> {
        match ImageFormat::from_path(path)? {
            ImageFormat::Jpeg => {
                let data = std::fs::read(path)?;
                Self::decode_jpeg(&data)
            }
            ImageFormat::Png => {
                let data = std::fs::read(path)?;
                let buf =
                    image::load_from_memory_with_format(&data, image::ImageFormat::Png)?.to_rgba8();
                Ok(Self { buf })
            }
        }
    }

    /// Decodes a JFIF JPEG or Motion JPEG from a byte slice.
    pub fn decode_jpeg(data: &[u8]) -> anyhow::Result<Self> {
        jpeg::decode_jpeg(data)
    }

    pub fn from_rgba8(res: Resolution, buf: &[u8]) -> Self {
        let expected_size = res.width() as usize * res.height() as usize * 4;
        assert_eq!(
            expected_size,
            buf.len(),
            "incorrect buffer size {} for {} image (expected {} bytes)",
            buf.len(),
            res,
            expected_size,
        );

        Self {
            buf: ImageBuffer::from_vec(res.width(), res.height(), buf.to_vec())
                .expect("buffer size does not match image resolution"),
        }
    }

    /// Saves an image to the file system.
    ///
    /// The path must have a supported file extension (`jpeg`, `jpg` or `png`).
    pub fn save<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> {
        self.save_impl(path.as_ref())
    }

    fn save_impl(&self, path: &Path) -> anyhow::Result<()> {
        match ImageFormat::from_path(path)? {
            _ => Ok(self.buf.save(path)?),
        }
    }

    /// Creates an empty image of a specified size.
    ///
    /// The image will start out black and fully transparent.
    pub fn new(width: u32, height: u32) -> Self {
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

    /// Returns a [`Rect`] covering this image.
    ///
    /// The rectangle will be positioned at `(0, 0)` and have the width and height of the image.
    #[inline]
    pub fn rect(&self) -> Rect {
        Rect::from_top_left(0.0, 0.0, self.width() as f32, self.height() as f32)
    }

    /// Gets the image color at the given pixel coordinates.
    ///
    /// # Panics
    ///
    /// This will panic if `(x, y)` is outside the bounds of this image.
    #[cfg(test)]
    fn get(&self, x: u32, y: u32) -> Color {
        let rgb = &self.buf[(x, y)];
        Color(rgb.0)
    }

    /// Sets the image color at the given pixel coordinates.
    ///
    /// # Panics
    ///
    /// This will panic if `(x, y)` is outside the bounds of this image.
    #[cfg(test)]
    fn set(&mut self, x: u32, y: u32, color: Color) {
        self.buf[(x, y)] = Rgba(color.0);
    }

    /// Creates an immutable view into an area of this image, specified by `rect`.
    ///
    /// If `rect` lies partially outside of `self`, the pixels that are outside of `self` will have
    /// the value [`Color::NULL`] and ignore writes. The returned view always has the size of
    /// `rect`.
    pub fn view(&self, rect: impl Into<RotatedRect>) -> ImageView<'_> {
        ImageView {
            image: self,
            data: ViewData::full(self).view(rect),
        }
    }

    /// Creates a mutable view into an area of this image, specified by `rect`.
    ///
    /// If `rect` lies partially outside of `self`, the pixels that are outside of `self` will have
    /// the value [`Color::NULL`] and ignore writes. The returned view always has the size of
    /// `rect`.
    pub fn view_mut(&mut self, rect: impl Into<RotatedRect>) -> ImageViewMut<'_> {
        ImageViewMut {
            data: ViewData::full(self).view(rect),
            image: self,
        }
    }

    pub fn flip_horizontal(&self) -> Image {
        Image {
            buf: image::imageops::flip_horizontal(&self.buf),
        }
    }

    pub fn flip_vertical(&self) -> Image {
        Image {
            buf: image::imageops::flip_vertical(&self.buf),
        }
    }

    pub fn flip_horizontal_in_place(&mut self) {
        image::imageops::flip_horizontal_in_place(&mut self.buf);
    }

    pub fn flip_vertical_in_place(&mut self) {
        image::imageops::flip_vertical_in_place(&mut self.buf);
    }

    /// Overwrites the data in `self` with a `src` image, stretching or shrinking `src` as
    /// necessary.
    ///
    /// Note that this always blends the *entire* `src` with the *entire* destination. A smaller
    /// source/destination area can be selected by creating a sub-view first.
    ///
    /// By default, this performs alpha blending.
    pub fn blend_from<'b, V: AsImageView>(&'b mut self, src: &'b V) -> Blend<'b> {
        Blend::new(self.as_view_mut(), src.as_view())
    }

    /// Clears the image, setting every pixel value to `color`.
    pub fn clear(&mut self, color: Color) {
        self.buf.pixels_mut().for_each(|pix| pix.0 = color.0);
    }

    #[inline]
    pub fn data(&self) -> &[u8] {
        // FIXME: useful method, but maybe should take a callback
        self.buf.as_raw()
    }
}

impl fmt::Debug for Image {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{} Image", self.width(), self.height())
    }
}

#[derive(Debug, Clone, Copy)]
struct ViewData {
    /// Rectangle in the root image's coordinates.
    rect: RotatedRect,
}

impl ViewData {
    fn full(image: &Image) -> Self {
        Self {
            rect: image.rect().into(),
        }
    }

    fn view(&self, rect: impl Into<RotatedRect>) -> Self {
        let rect: RotatedRect = rect.into();
        let radians = self.rect.rotation_radians() + rect.rotation_radians();

        let (cx, cy) = rect.rect().center();
        let [cx, cy] = self.rect.transform_out(cx, cy);
        let [x, y] = [
            cx - rect.rect().width() / 2.0,
            cy - rect.rect().height() / 2.0,
        ];

        Self {
            rect: RotatedRect::new(rect.rect().move_to(x, y), radians),
        }
    }

    fn rect(&self) -> Rect {
        Rect::from_top_left(0.0, 0.0, self.width(), self.height())
    }

    fn width(&self) -> f32 {
        self.rect.rect().width()
    }

    fn height(&self) -> f32 {
        self.rect.rect().height()
    }

    fn image_coord(&self, x: u32, y: u32, image: &Image) -> Option<(u32, u32)> {
        let [x, y] = self.rect.transform_out(x as f32 + 0.5, y as f32 + 0.5);
        let [x, y] = [(x - 0.5).round(), (y - 0.5).round()];

        if x < 0.0 || y < 0.0 || x.ceil() >= u32::MAX as f32 || y.ceil() >= u32::MAX as f32 {
            return None;
        }

        let [x, y] = [x.round() as u32, y.round() as u32];
        if x >= image.width() || y >= image.height() {
            return None;
        }
        Some((x, y))
    }

    fn get(&self, x: u32, y: u32, image: &Image) -> Color {
        match self.image_coord(x, y, image) {
            Some((x, y)) => Color(image.buf[(x, y)].0),
            _ => Color::NULL,
        }
    }
}

/// An immutable view of a rectangular section of an [`Image`].
#[derive(Clone, Copy)]
pub struct ImageView<'a> {
    image: &'a Image,
    data: ViewData,
}

impl<'a> ImageView<'a> {
    fn as_generic_image_view(&self) -> impl GenericImageView<Pixel = Rgba<u8>> + '_ {
        struct Wrapper<'a>(ImageView<'a>);

        impl GenericImageView for Wrapper<'_> {
            type Pixel = Rgba<u8>;

            fn dimensions(&self) -> (u32, u32) {
                (self.0.width(), self.0.height())
            }

            fn bounds(&self) -> (u32, u32, u32, u32) {
                (0, 0, self.0.width(), self.0.height())
            }

            fn get_pixel(&self, x: u32, y: u32) -> Self::Pixel {
                Rgba(self.0.data.get(x, y, self.0.image).0)
            }
        }

        Wrapper(*self)
    }

    /// Returns the width of this view, in pixels.
    pub fn width(&self) -> u32 {
        self.data.width() as u32
    }

    /// Returns the height of this view, in pixels.
    pub fn height(&self) -> u32 {
        self.data.height() as u32
    }

    /// Returns the size of this view.
    #[inline]
    pub fn resolution(&self) -> Resolution {
        Resolution::new(self.width(), self.height())
    }

    /// Returns a [`Rect`] of the size of this view.
    ///
    /// The rectangle will be positioned at `(0, 0)` and have the width and height of the view.
    #[inline]
    pub fn rect(&self) -> Rect {
        self.data.rect()
    }

    /// Gets the image color at the given pixel coordinates.
    ///
    /// # Panics
    ///
    /// This will panic if `(x, y)` is outside the bounds of this view.
    #[inline]
    pub fn get(&self, x: u32, y: u32) -> Color {
        Color(self.as_generic_image_view().get_pixel(x, y).0)
    }

    /// Creates an immutable subview into an area of this view, specified by `rect`.
    ///
    /// If `rect` lies partially outside of `self`, the pixels that are outside of `self` will
    /// access the underlying [`Image`] outside of this [`ImageView`]. If part of `rect` are outside
    /// of the underlying [`Image`], they will be read as [`Color::NULL`].
    ///
    /// The returned view always has the size of `rect`.
    pub fn view(&self, rect: impl Into<RotatedRect>) -> ImageView<'_> {
        ImageView {
            image: self.image,
            data: self.data.view(rect),
        }
    }

    pub fn flip_horizontal(&self) -> Image {
        Image {
            buf: image::imageops::flip_horizontal(&self.as_generic_image_view()),
        }
    }

    pub fn flip_vertical(&self) -> Image {
        Image {
            buf: image::imageops::flip_vertical(&self.as_generic_image_view()),
        }
    }

    /// Copies the contents of this view into a new [`Image`].
    pub fn to_image(&self) -> Image {
        let mut image = Image::new(self.width(), self.height());
        image
            .buf
            .copy_from(&self.as_generic_image_view(), 0, 0)
            .unwrap();
        image
    }
}

impl fmt::Debug for ImageView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{} ImageView", self.width(), self.height())
    }
}

/// A mutable view of a rectangular section of an [`Image`].
pub struct ImageViewMut<'a> {
    image: &'a mut Image,
    data: ViewData,
}

impl<'a> ImageViewMut<'a> {
    fn as_generic_image(&mut self) -> impl GenericImage<Pixel = Rgba<u8>> + '_ {
        struct Wrapper<'b>(ImageViewMut<'b>, Rgba<u8>);

        impl GenericImageView for Wrapper<'_> {
            type Pixel = Rgba<u8>;

            fn dimensions(&self) -> (u32, u32) {
                (self.0.width(), self.0.height())
            }

            fn bounds(&self) -> (u32, u32, u32, u32) {
                (0, 0, self.0.width(), self.0.height())
            }

            fn get_pixel(&self, x: u32, y: u32) -> Self::Pixel {
                Rgba(self.0.data.get(x, y, self.0.image).0)
            }
        }

        impl GenericImage for Wrapper<'_> {
            fn get_pixel_mut(&mut self, x: u32, y: u32) -> &mut Self::Pixel {
                self.0
                    .data
                    .image_coord(x, y, self.0.image)
                    .map_or(&mut self.1, |(x, y)| &mut self.0.image.buf[(x, y)])
            }

            fn put_pixel(&mut self, x: u32, y: u32, pixel: Self::Pixel) {
                if let Some((x, y)) = self.0.data.image_coord(x, y, self.0.image) {
                    self.0.image.buf[(x, y)] = pixel;
                }
            }

            fn blend_pixel(&mut self, x: u32, y: u32, pixel: Self::Pixel) {
                if let Some((x, y)) = self.0.data.image_coord(x, y, self.0.image) {
                    #[allow(deprecated)]
                    self.0.image.buf.blend_pixel(x, y, pixel);
                }
            }
        }

        Wrapper(self.reborrow(), Rgba([0, 0, 0, 0]))
    }

    /// Returns the width of this view, in pixels.
    pub fn width(&self) -> u32 {
        self.data.width() as u32
    }

    /// Returns the height of this view, in pixels.
    pub fn height(&self) -> u32 {
        self.data.height() as u32
    }

    /// Returns the size of this view.
    #[inline]
    pub fn resolution(&self) -> Resolution {
        Resolution::new(self.width(), self.height())
    }

    /// Returns a [`Rect`] of the size of this view.
    ///
    /// The rectangle will be positioned at `(0, 0)` and have the width and height of the view.
    #[inline]
    pub fn rect(&self) -> Rect {
        self.data.rect()
    }

    /// Gets the image color at the given pixel coordinates.
    ///
    /// # Panics
    ///
    /// This will panic if `(x, y)` is outside the bounds of this view.
    #[inline]
    fn get(&self, x: u32, y: u32) -> Color {
        let rgb = self.as_view().as_generic_image_view().get_pixel(x, y);
        Color(rgb.0)
    }

    /// Sets the image color at the given pixel coordinates.
    ///
    /// # Panics
    ///
    /// This will panic if `(x, y)` is outside the bounds of this view.
    #[inline]
    fn set(&mut self, x: u32, y: u32, color: Color) {
        self.as_generic_image().put_pixel(x, y, Rgba(color.0));
    }

    /// Borrows an identical [`ImageViewMut`] from `self` that may have a shorter lifetime.
    ///
    /// This is equivalent to the implicit "reborrowing" that happens on Rust references. It needs
    /// to be a method call here because user-defined types cannot opt into making this happen
    /// automatically.
    pub fn reborrow(&mut self) -> ImageViewMut<'_> {
        ImageViewMut {
            image: self.image,
            data: self.data,
        }
    }

    /// Creates an immutable subview into an area of this view, specified by `rect`.
    ///
    /// If `rect` lies partially outside of `self`, the pixels that are outside of `self` will have
    /// the value [`Color::NULL`] and ignore writes. The returned view always has the size of
    /// `rect`.
    pub fn view(&self, rect: impl Into<RotatedRect>) -> ImageView<'_> {
        ImageView {
            image: self.image,
            data: self.data.view(rect),
        }
    }

    /// Creates a mutable view into an area of this view, specified by `rect`.
    ///
    /// If `rect` lies partially outside of `self`, the pixels that are outside of `self` will have
    /// the value [`Color::NULL`] and ignore writes. The returned view always has the size of
    /// `rect`.
    pub fn view_mut(&mut self, rect: impl Into<RotatedRect>) -> ImageViewMut<'_> {
        ImageViewMut {
            image: self.image,
            data: self.data.view(rect),
        }
    }

    pub fn flip_horizontal(&self) -> Image {
        self.as_view().flip_horizontal()
    }

    pub fn flip_vertical(&self) -> Image {
        self.as_view().flip_vertical()
    }

    pub fn flip_horizontal_in_place(&mut self) {
        image::imageops::flip_horizontal_in_place(&mut self.as_generic_image());
    }

    pub fn flip_vertical_in_place(&mut self) {
        image::imageops::flip_vertical_in_place(&mut self.as_generic_image());
    }

    /// Copies the contents of this view into a new [`Image`].
    pub fn to_image(&self) -> Image {
        self.as_view().to_image()
    }

    /// Overwrites the data in `self` with a `src` image, stretching or shrinking `src` as
    /// necessary.
    ///
    /// Note that this always blends the *entire* `src` with the *entire* destination. A smaller
    /// source/destination area can be selected by creating a sub-view first.
    ///
    /// By default, this performs alpha blending.
    pub fn blend_from<'b, V: AsImageView>(&'b mut self, src: &'b V) -> Blend<'b> {
        Blend::new(self.as_view_mut(), src.as_view())
    }
}

impl fmt::Debug for ImageViewMut<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{} ImageViewMut", self.width(), self.height())
    }
}

/// An 8-bit RGBA color.
///
/// Colors are always in the sRGB color space and use non-premultiplied alpha.
#[derive(PartialEq, Eq, Clone, Copy)]
pub struct Color(pub(crate) [u8; 4]);

impl Color {
    /// Fully transparent black (all components are 0).
    pub const NULL: Self = Self([0, 0, 0, 0]);
    pub const BLACK: Self = Self([0, 0, 0, 255]);
    pub const WHITE: Self = Self([255, 255, 255, 255]);
    pub const RED: Self = Self([255, 0, 0, 255]);
    pub const GREEN: Self = Self([0, 255, 0, 255]);
    pub const BLUE: Self = Self([0, 0, 255, 255]);
    pub const YELLOW: Self = Self([255, 255, 0, 255]);
    pub const MAGENTA: Self = Self([255, 0, 255, 255]);
    pub const CYAN: Self = Self([0, 255, 255, 255]);

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

    #[inline]
    pub fn a(&self) -> u8 {
        self.0[3]
    }

    pub fn with_alpha(mut self, a: u8) -> Color {
        self.0[3] = a;
        self
    }
}

impl fmt::Debug for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "#{:02x}{:02x}{:02x}{:02x}",
            self.r(),
            self.g(),
            self.b(),
            self.a(),
        )
    }
}

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
        self.view(Rect::from_top_left(
            0.0,
            0.0,
            self.width() as f32,
            self.height() as f32,
        ))
    }
}

impl<'a> AsImageView for ImageView<'a> {
    fn as_view(&self) -> ImageView<'_> {
        *self
    }
}

impl AsImageViewMut for Image {
    fn as_view_mut(&mut self) -> ImageViewMut<'_> {
        self.view_mut(Rect::from_top_left(
            0.0,
            0.0,
            self.width() as f32,
            self.height() as f32,
        ))
    }
}

impl<'a> AsImageView for ImageViewMut<'a> {
    fn as_view(&self) -> ImageView<'_> {
        ImageView {
            data: self.data,
            image: self.image,
        }
    }
}

impl<'a> AsImageViewMut for ImageViewMut<'a> {
    fn as_view_mut(&mut self) -> ImageViewMut<'_> {
        self.reborrow()
    }
}

impl<'a, V: AsImageView> AsImageView for &'a V {
    fn as_view(&self) -> ImageView<'_> {
        (*self).as_view()
    }
}

impl<'a, V: AsImageView> AsImageView for &'a mut V {
    fn as_view(&self) -> ImageView<'_> {
        (**self).as_view()
    }
}

impl<'a, V: AsImageViewMut> AsImageViewMut for &'a mut V {
    fn as_view_mut(&mut self) -> ImageViewMut<'_> {
        (*self).as_view_mut()
    }
}
