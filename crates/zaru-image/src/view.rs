//! Image sub-views.

use std::fmt;

use zaru_linalg::{vec2, Vec2, Vec2f};

use crate::{
    blend,
    rect::{Rect, RotatedRect},
    Image,
};

#[allow(unused)] // doc links only
use crate::Color;

/// Sub-view construction.
impl Image {
    /// Creates an immutable view into an area of this image, specified by `rect`.
    ///
    /// If `rect` lies partially outside of `self`, the pixels that are outside of `self` will have
    /// the value [`Color::NONE`] and ignore writes. The returned view always has the size of
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
    /// the value [`Color::NONE`] and ignore writes. The returned view always has the size of
    /// `rect`.
    pub fn view_mut(&mut self, rect: impl Into<RotatedRect>) -> ImageViewMut<'_> {
        ImageViewMut {
            data: ViewData::full(self).view(rect),
            image: self,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ViewData {
    /// Rectangle in the root image's coordinates.
    pub(crate) rect: RotatedRect,
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

        let pt = self.rect.transform_out(rect.rect().center()) - rect.rect().size() * 0.5;

        Self {
            rect: RotatedRect::new(rect.rect().move_to(pt.x, pt.y), radians),
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

    /// Returns the UV coordinates of the top left and bottom right corners of this view in the
    /// original image.
    pub(crate) fn uvs(&self, image: &Image) -> [Vec2f; 4] {
        let top_left = self.uv(0.0, 0.0, image);
        let bottom_right = self.uv(self.width(), self.height(), image);
        // top left, top right, bottom left, bottom right
        [
            top_left,
            vec2(bottom_right.x, top_left.y),
            vec2(top_left.x, bottom_right.y),
            bottom_right,
        ]
    }

    /// Returns the positions of the top left and bottom right corners of this view, in clip space.
    pub(crate) fn clip_corners(&self, image: &Image) -> [Vec2f; 4] {
        let top_left = self.position(0.0, 0.0, image);
        let bottom_right = self.position(self.width(), self.height(), image);
        // top left, top right, bottom left, bottom right
        [
            top_left,
            vec2(bottom_right.x, top_left.y),
            vec2(top_left.x, bottom_right.y),
            bottom_right,
        ]
    }

    fn uv(&self, x: f32, y: f32, image: &Image) -> Vec2f {
        let size = vec2(image.width() as f32, image.height() as f32);
        let pt = self.rect.transform_out([x, y]);
        pt / size
    }

    /// Computes the clip-space position in the underlying [`Image`] that corresponds to the given
    /// coordinates in this view.
    pub(crate) fn position(&self, x: f32, y: f32, image: &Image) -> Vec2f {
        let size = vec2(image.width() as f32, image.height() as f32);
        let pt = self.rect.transform_out([x, y]) / size - Vec2::splat(0.5);

        pt * vec2(2.0, -2.0)
    }
}

/// An immutable view of a rectangular section of an [`Image`].
#[derive(Clone, Copy)]
pub struct ImageView<'a> {
    pub(crate) image: &'a Image,
    pub(crate) data: ViewData,
}

impl<'a> ImageView<'a> {
    /// Returns a [`Rect`] of the size of this view.
    ///
    /// The rectangle will be positioned at `(0, 0)` and have the width and height of the view. Note
    /// that view sizes are allowed to be fractional.
    #[inline]
    pub fn rect(&self) -> Rect {
        self.data.rect()
    }

    /// Creates an immutable subview into an area of this view, specified by `rect`.
    ///
    /// If `rect` lies partially outside of `self`, the pixels that are outside of `self` will
    /// access the underlying [`Image`] outside of this [`ImageView`]. If part of `rect` are outside
    /// of the underlying [`Image`], they will be read as [`Color::NONE`].
    ///
    /// The returned view always has the size of `rect`.
    pub fn view(&self, rect: impl Into<RotatedRect>) -> ImageView<'_> {
        ImageView {
            image: self.image,
            data: self.data.view(rect),
        }
    }

    /// Copies the contents of this view into a new [`Image`].
    ///
    /// The returned [`Image`] will have the size of `self`. If the width or height of `self` is not
    /// an integer, it is rounded up to the next integer.
    pub fn to_image(&self) -> Image {
        let mut dest = Image::new((
            self.data.width().ceil() as u32,
            self.data.height().ceil() as u32,
        ));
        blend(&mut dest, self);
        dest
    }
}

impl fmt::Debug for ImageView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ImageView @ {:?}", self.data.rect)
    }
}

/// A mutable view of a rectangular section of an [`Image`].
pub struct ImageViewMut<'a> {
    pub(crate) image: &'a mut Image,
    pub(crate) data: ViewData,
}

impl<'a> ImageViewMut<'a> {
    /// Computes the clip-space position in the underlying [`Image`] that corresponds to the given
    /// coordinates in this view.
    pub(crate) fn position(&self, x: f32, y: f32) -> Vec2f {
        self.data.position(x, y, self.image)
    }

    /// Returns a [`Rect`] of the size of this view.
    ///
    /// The rectangle will be positioned at `(0, 0)` and have the width and height of the view. Note
    /// that view sizes are allowed to be fractional.
    #[inline]
    pub fn rect(&self) -> Rect {
        self.data.rect()
    }

    /// Borrows an identical [`ImageViewMut`] from `self` that may have a shorter lifetime.
    ///
    /// This is equivalent to the implicit "reborrowing" that happens on Rust references. It needs
    /// to be a method call here because user-defined types cannot opt into making this happen
    /// automatically.
    #[inline]
    pub fn reborrow(&mut self) -> ImageViewMut<'_> {
        ImageViewMut {
            image: self.image,
            data: self.data,
        }
    }

    /// Creates an immutable subview into an area of this view, specified by `rect`.
    ///
    /// If `rect` lies partially outside of `self`, the pixels that are outside of `self` will have
    /// the value [`Color::NONE`] and ignore writes. The returned view always has the size of
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
    /// the value [`Color::NONE`] and ignore writes. The returned view always has the size of
    /// `rect`.
    pub fn view_mut(&mut self, rect: impl Into<RotatedRect>) -> ImageViewMut<'_> {
        ImageViewMut {
            image: self.image,
            data: self.data.view(rect),
        }
    }

    /// Copies the contents of this view into a new [`Image`].
    ///
    /// The returned [`Image`] will have the size of `self`. If the width or height of `self` is not
    /// an integer, it is rounded up to the next integer.
    pub fn to_image(&self) -> Image {
        self.as_view().to_image()
    }
}

impl fmt::Debug for ImageViewMut<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ImageViewMut @ {:?}", self.data.rect)
    }
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

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use super::*;

    #[test]
    fn view_to_image() {
        let pixels = &[
            0x00, 0x11, 0x22, 0x33, // 0
            0x44, 0x55, 0x66, 0x77, // 1
        ];
        let image = Image::from_rgba8((2, 1), pixels);

        // Full view.
        let out = image.as_view().to_image();
        assert_eq!(out.width(), 2);
        assert_eq!(out.height(), 1);
        out.with_data(|data| {
            assert_eq!(data.to_vec(), pixels);
        });

        // First pixel only.
        let out = image
            .view(Rect::from_top_left(0.0, 0.0, 1.0, 1.0))
            .to_image();
        assert_eq!(out.width(), 1);
        assert_eq!(out.height(), 1);
        out.with_data(|data| {
            assert_eq!(data.to_vec(), &pixels[0..4]);
        });

        // Second pixel only.
        let out = image
            .view(Rect::from_top_left(1.0, 0.0, 1.0, 1.0))
            .to_image();
        assert_eq!(out.width(), 1);
        assert_eq!(out.height(), 1);
        out.with_data(|data| {
            assert_eq!(data.to_vec(), &pixels[4..8]);
        });

        // Second and third pixel (which doesn't exist, and so ends up black).
        let out = image
            .view(Rect::from_top_left(1.0, 0.0, 2.0, 1.0))
            .to_image();
        assert_eq!(out.width(), 2);
        assert_eq!(out.height(), 1);
        out.with_data(|data| {
            assert_eq!(
                data.to_vec(),
                &[0x44, 0x55, 0x66, 0x77, 0x00, 0x00, 0x00, 0x00]
            );
        });

        // Full view, but the rectangle is rotated 180°, swapping the pixel positions.
        let out = image.view(RotatedRect::new(image.rect(), PI)).to_image();
        assert_eq!(out.width(), 2);
        assert_eq!(out.height(), 1);
        out.with_data(|data| {
            assert_eq!(
                data.to_vec(),
                &[
                    0x44, 0x55, 0x66, 0x77, // 1
                    0x00, 0x11, 0x22, 0x33, // 0
                ]
            );
        });
    }
}
