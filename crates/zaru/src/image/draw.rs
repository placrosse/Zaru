//! Functions for drawing onto images.
//!
//! The functions in this module all return a guard type that will perform the operation when
//! dropped. This type may provide methods that can be used to customize the default style of the
//! object being drawn. This results in a pretty ergonomic fluent builder-like API, without
//! requiring any unnecessary method calls.

use std::convert::Infallible;

use embedded_graphics::{
    draw_target::DrawTarget,
    mono_font::{ascii, MonoTextStyle},
    prelude::*,
    primitives::{Line, PrimitiveStyle, Rectangle},
    text::{Alignment, Baseline, Text, TextStyleBuilder},
};
use itertools::Itertools;
use nalgebra::{UnitQuaternion, Vector3};
use zaru_linalg::{vec2, Vec2f};

use crate::image::{AsImageViewMut, Color, ImageViewMut, Rect};

use super::RotatedRect;

/// Guard returned by [`rect`]; draws the rectangle when dropped and allows customization.
pub struct DrawRect<'a> {
    image: ImageViewMut<'a>,
    rect: Rect,
    color: Color,
}

impl DrawRect<'_> {
    /// Sets the rectangle's color.
    pub fn color(&mut self, color: Color) -> &mut Self {
        self.color = color;
        self
    }
}

impl Drop for DrawRect<'_> {
    fn drop(&mut self) {
        let corners = self.rect.corners();
        for (start, end) in corners.into_iter().circular_tuple_windows().take(4) {
            line(&mut self.image, start, end).color(self.color);
        }
    }
}

/// Guard returned by [`rotated_rect`]; draws the rotated rectangle when dropped and allows
/// customization.
pub struct DrawRotatedRect<'a> {
    image: ImageViewMut<'a>,
    rect: RotatedRect,
    color: Color,
}

impl<'a> DrawRotatedRect<'a> {
    /// Sets the color.
    pub fn color(&mut self, color: Color) -> &mut Self {
        self.color = color;
        self
    }
}

impl<'a> Drop for DrawRotatedRect<'a> {
    fn drop(&mut self) {
        let corners = self.rect.rotated_corners();
        for (start, end) in corners.into_iter().circular_tuple_windows().take(4) {
            line(&mut self.image, start, end).color(self.color);
        }
    }
}

/// Guard returned by [`marker`]; draws the marker when dropped and allows customization.
pub struct DrawMarker<'a> {
    image: ImageViewMut<'a>,
    pos: Vec2f,
    color: Color,
    size: u32,
}

impl<'a> DrawMarker<'a> {
    /// Sets the marker's color.
    pub fn color(&mut self, color: Color) -> &mut Self {
        self.color = color;
        self
    }

    /// Sets the width and height of the marker.
    ///
    /// The default size is 5. The size must be *uneven* and *non-zero*. A size of 1 will result in
    /// a single pixel getting drawn.
    pub fn size(&mut self, size: u32) -> &mut Self {
        assert!(size != 0, "marker size must be greater than zero");
        assert!(size % 2 == 1, "marker size must be an uneven number");
        self.size = size;
        self
    }
}

impl Drop for DrawMarker<'_> {
    fn drop(&mut self) {
        let offset = ((self.size - 1) / 2) as i32;
        for (xoff, yoff) in (-offset..=offset)
            .zip(-offset..=offset)
            .chain((-offset..=offset).rev().zip(-offset..=offset))
        {
            match Pixel(
                Point {
                    x: self.pos.x.round() as i32 + xoff,
                    y: self.pos.y.round() as i32 + yoff,
                },
                self.color,
            )
            .draw(&mut Target(self.image.reborrow()))
            {
                Ok(_) => {}
                Err(infallible) => match infallible {},
            }
        }
    }
}

/// Guard returned by [`line`][line()]; draws the line when dropped and allows customization.
pub struct DrawLine<'a> {
    image: ImageViewMut<'a>,
    start: Vec2f,
    end: Vec2f,
    color: Color,
}

impl<'a> DrawLine<'a> {
    /// Sets the line's color.
    pub fn color(&mut self, color: Color) -> &mut Self {
        self.color = color;
        self
    }
}

impl<'a> Drop for DrawLine<'a> {
    fn drop(&mut self) {
        match Line::new(
            Point::new(self.start.x.round() as i32, self.start.y.round() as i32),
            Point::new(self.end.x.round() as i32, self.end.y.round() as i32),
        )
        .into_styled(PrimitiveStyle::with_stroke(self.color, 1))
        .draw(&mut Target(self.image.reborrow()))
        {
            Ok(_) => {}
            Err(infallible) => match infallible {},
        }
    }
}

/// Guard returned by [`text`]; draws the text when dropped and allows customization.
pub struct DrawText<'a> {
    image: ImageViewMut<'a>,
    pos: Vec2f,
    text: &'a str,
    color: Color,
    alignment: Alignment,
    baseline: Baseline,
}

impl<'a> DrawText<'a> {
    /// Sets the text color.
    pub fn color(&mut self, color: Color) -> &mut Self {
        self.color = color;
        self
    }

    /// Aligns the top of the text with the `y` coordinate.
    pub fn align_top(&mut self) -> &mut Self {
        self.baseline = Baseline::Top;
        self
    }

    /// Aligns the bottom of the text with the `y` coordinate.
    pub fn align_bottom(&mut self) -> &mut Self {
        self.baseline = Baseline::Bottom;
        self
    }

    /// Aligns the left side of the text with the `x` coordinate.
    pub fn align_left(&mut self) -> &mut Self {
        self.alignment = Alignment::Left;
        self
    }

    /// Aligns the right side of the text with the `x` coordinate.
    pub fn align_right(&mut self) -> &mut Self {
        self.alignment = Alignment::Right;
        self
    }
}

impl<'a> Drop for DrawText<'a> {
    fn drop(&mut self) {
        // FIXME: do this in a better way, e-g's fonts lack some common glyphs
        let character_style = MonoTextStyle::new(&ascii::FONT_6X10, self.color);
        let text_style = TextStyleBuilder::new()
            .alignment(self.alignment)
            .baseline(self.baseline)
            .build();
        match Text::with_text_style(
            self.text,
            Point::new(self.pos.x.round() as i32, self.pos.y.round() as i32),
            character_style,
            text_style,
        )
        .draw(&mut Target(self.image.reborrow()))
        {
            Ok(_) => {}
            Err(infallible) => match infallible {},
        }
    }
}

/// Guard returned by [`quaternion`]; draws the rotated coordinate system when dropped.
pub struct DrawQuaternion<'a> {
    image: ImageViewMut<'a>,
    pos: Vec2f,
    quaternion: UnitQuaternion<f32>,
    axis_length: f32,
}

impl<'a> DrawQuaternion<'a> {
    /// Sets the length of each coordinate axis, in pixels.
    pub fn axis_length(&mut self, length: f32) -> &mut Self {
        self.axis_length = length;
        self
    }
}

impl<'a> Drop for DrawQuaternion<'a> {
    fn drop(&mut self) {
        let axis_length = self.axis_length;

        let x = (self.quaternion * Vector3::x() * axis_length).xy();
        let y = (self.quaternion * Vector3::y() * axis_length).xy();
        let z = (self.quaternion * Vector3::z() * axis_length).xy();
        // Flip Y axis, since it points up in 3D space but down in image coordinates.
        let x_end = self.pos + vec2(x.x, -x.y);
        let y_end = self.pos + vec2(y.x, -y.y);
        let z_end = self.pos + vec2(z.x, -z.y);

        line(&mut self.image, self.pos, x_end).color(Color::RED);
        line(&mut self.image, self.pos, y_end).color(Color::GREEN);
        line(&mut self.image, self.pos, z_end).color(Color::BLUE);
    }
}

/// Draws a rectangle onto an image.
pub fn rect<I: AsImageViewMut>(image: &mut I, rect: Rect) -> DrawRect<'_> {
    DrawRect {
        image: image.as_view_mut(),
        rect,
        color: Color::RED,
    }
}

/// Draws a rotated rectangle onto an image.
pub fn rotated_rect<I: AsImageViewMut>(image: &mut I, rect: RotatedRect) -> DrawRotatedRect<'_> {
    DrawRotatedRect {
        image: image.as_view_mut(),
        rect,
        color: Color::RED,
    }
}

/// Draws a marker onto an image.
///
/// This can be used to visualize shape landmarks or points of interest.
pub fn marker<I: AsImageViewMut>(image: &mut I, pos: impl Into<Vec2f>) -> DrawMarker<'_> {
    DrawMarker {
        image: image.as_view_mut(),
        pos: pos.into(),
        color: Color::RED,
        size: 5,
    }
}

/// Draws a line onto an image.
pub fn line<I: AsImageViewMut>(
    image: &mut I,
    start: impl Into<Vec2f>,
    end: impl Into<Vec2f>,
) -> DrawLine<'_> {
    DrawLine {
        image: image.as_view_mut(),
        start: start.into(),
        end: end.into(),
        color: Color::BLUE,
    }
}

/// Draws a text string onto an image.
///
/// By default, the text is drawn centered horizontally and vertically around `x` and `y`.
pub fn text<'a, I: AsImageViewMut>(
    image: &'a mut I,
    pos: impl Into<Vec2f>,
    text: &'a str,
) -> DrawText<'a> {
    DrawText {
        image: image.as_view_mut(),
        pos: pos.into(),
        text,
        color: Color::RED,
        alignment: Alignment::Center,
        baseline: Baseline::Middle,
    }
}

/// Visualizes a rotation in 3D space by drawing XYZ coordinate axes rotated accordingly.
///
/// This assumes that the quaternion describes a rotation in Zaru's 3D coordinate reference frame
/// (X points right, Y points up, Z points into the screen).
///
/// The `x` and `y` parameters describe where to put the origin of the coordinate system. Typically
/// this is the center of an image or of the object of interest.
pub fn quaternion<'a, I: AsImageViewMut>(
    image: &'a mut I,
    pos: impl Into<Vec2f>,
    quaternion: UnitQuaternion<f32>,
) -> DrawQuaternion<'a> {
    DrawQuaternion {
        image: image.as_view_mut(),
        pos: pos.into(),
        quaternion,
        axis_length: 10.0,
    }
}

struct Target<'a>(ImageViewMut<'a>);

impl Dimensions for Target<'_> {
    fn bounding_box(&self) -> Rectangle {
        let (width, height) = (
            self.0.rect().width().ceil() as u32,
            self.0.rect().height().ceil() as u32,
        );

        Rectangle {
            top_left: Point { x: 0, y: 0 },
            size: Size { width, height },
        }
    }
}

impl DrawTarget for Target<'_> {
    type Color = Color;

    type Error = Infallible;

    fn draw_iter<I>(&mut self, pixels: I) -> Result<(), Self::Error>
    where
        I: IntoIterator<Item = embedded_graphics::Pixel<Self::Color>>,
    {
        for pixel in pixels {
            let rgb = pixel.1 .0;
            if pixel.0.x >= 0
                && (pixel.0.x as u32) < self.0.rect().width().ceil() as u32
                && pixel.0.y >= 0
                && (pixel.0.y as u32) < self.0.rect().height().ceil() as u32
            {
                self.0.set(pixel.0.x as _, pixel.0.y as _, Color(rgb));
            }
        }

        Ok(())
    }
}
