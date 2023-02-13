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
    primitives::{self, Line, PrimitiveStyle, Rectangle},
    text::{Alignment, Baseline, Text, TextStyleBuilder},
};
use itertools::Itertools;
use nalgebra::{UnitQuaternion, Vector2, Vector3};

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
        for ([x1, y1], [x2, y2]) in corners.into_iter().circular_tuple_windows().take(4) {
            line(&mut self.image, x1, y1, x2, y2).color(self.color);
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
        for ([x1, y1], [x2, y2]) in corners.into_iter().circular_tuple_windows().take(4) {
            line(&mut self.image, x1, y1, x2, y2).color(self.color);
        }
    }
}

/// Guard returned by [`marker`]; draws the marker when dropped and allows customization.
pub struct DrawMarker<'a> {
    image: ImageViewMut<'a>,
    x: f32,
    y: f32,
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
                    x: self.x.round() as i32 + xoff,
                    y: self.y.round() as i32 + yoff,
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
    start_x: f32,
    start_y: f32,
    end_x: f32,
    end_y: f32,
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
            Point::new(self.start_x.round() as i32, self.start_y.round() as i32),
            Point::new(self.end_x.round() as i32, self.end_y.round() as i32),
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
    x: f32,
    y: f32,
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
            Point::new(self.x.round() as i32, self.y.round() as i32),
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

/// Guard returned by [`circle`]; draws the circle when dropped and allows customization.
pub struct DrawCircle<'a> {
    image: ImageViewMut<'a>,
    x: f32,
    y: f32,
    diameter: f32,
    color: Color,
}

impl<'a> DrawCircle<'a> {
    /// Sets the circle's color.
    pub fn color(&mut self, color: Color) -> &mut Self {
        self.color = color;
        self
    }
}

impl<'a> Drop for DrawCircle<'a> {
    fn drop(&mut self) {
        let top_left = Point {
            x: (self.x - (self.diameter / 2.0)).round() as i32,
            y: (self.y - (self.diameter / 2.0)).round() as i32,
        };
        let circle = primitives::Circle {
            top_left,
            diameter: self.diameter as _,
        };
        match circle
            .into_styled(PrimitiveStyle::with_stroke(self.color, 1))
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
    x: f32,
    y: f32,
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
        let origin = Vector2::new(self.x, self.y);

        let x = (self.quaternion * Vector3::x() * axis_length).xy();
        let y = (self.quaternion * Vector3::y() * axis_length).xy();
        let z = (self.quaternion * Vector3::z() * axis_length).xy();
        // Flip Y axis, since it points up in 3D space but down in image coordinates.
        let x_end = origin + Vector2::new(x.x, -x.y);
        let y_end = origin + Vector2::new(y.x, -y.y);
        let z_end = origin + Vector2::new(z.x, -z.y);

        line(&mut self.image, self.x, self.y, x_end.x, x_end.y).color(Color::RED);
        line(&mut self.image, self.x, self.y, y_end.x, y_end.y).color(Color::GREEN);
        line(&mut self.image, self.x, self.y, z_end.x, z_end.y).color(Color::BLUE);
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
pub fn marker<I: AsImageViewMut>(image: &mut I, x: f32, y: f32) -> DrawMarker<'_> {
    DrawMarker {
        image: image.as_view_mut(),
        x,
        y,
        color: Color::from_rgb8(255, 0, 0),
        size: 5,
    }
}

/// Draws a line onto an image.
pub fn line<I: AsImageViewMut>(
    image: &mut I,
    start_x: f32,
    start_y: f32,
    end_x: f32,
    end_y: f32,
) -> DrawLine<'_> {
    DrawLine {
        image: image.as_view_mut(),
        start_x,
        start_y,
        end_x,
        end_y,
        color: Color::from_rgb8(0, 0, 255),
    }
}

/// Draws a text string onto an image.
///
/// By default, the text is drawn centered horizontally and vertically around `x` and `y`.
pub fn text<'a, I: AsImageViewMut>(
    image: &'a mut I,
    x: f32,
    y: f32,
    text: &'a str,
) -> DrawText<'a> {
    DrawText {
        image: image.as_view_mut(),
        x,
        y,
        text,
        color: Color::from_rgb8(255, 0, 0),
        alignment: Alignment::Center,
        baseline: Baseline::Middle,
    }
}

/// Draws a circle onto an image.
pub fn circle<'a, I: AsImageViewMut>(
    image: &'a mut I,
    x: f32,
    y: f32,
    diameter: f32,
) -> DrawCircle<'a> {
    DrawCircle {
        image: image.as_view_mut(),
        x,
        y,
        diameter,
        color: Color::GREEN,
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
    x: f32,
    y: f32,
    quaternion: UnitQuaternion<f32>,
) -> DrawQuaternion<'a> {
    DrawQuaternion {
        image: image.as_view_mut(),
        x,
        y,
        quaternion,
        axis_length: 10.0,
    }
}

struct Target<'a>(ImageViewMut<'a>);

impl Dimensions for Target<'_> {
    fn bounding_box(&self) -> Rectangle {
        let (width, height) = (self.0.width(), self.0.height());

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
                && (pixel.0.x as u32) < self.0.width()
                && pixel.0.y >= 0
                && (pixel.0.y as u32) < self.0.height()
            {
                self.0.set(pixel.0.x as _, pixel.0.y as _, Color(rgb));
            }
        }

        Ok(())
    }
}
