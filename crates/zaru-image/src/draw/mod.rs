//! Drawing API for [`Image`][crate::Image]s.
//!
//! This module contains a collection of freestanding functions that can draw shapes onto anything
//! that implements [`AsImageViewMut`]. All functions return a *guard object* that allows optional
//! customization of the shape and performs the draw operation when dropped.
//!
//! All drawing operations *overwrite* the target pixel with the shape color. They do not perform
//! blending.
//!
//! # A Note on Pixel Coordinates
//!
//! Pixel coordinates in this module are represented as [`f32`]. Even if they were integers, using
//! them with rotated views necessarily involves fractional pixel coordinates.
//!
//! This has some consequences that may be surprising: pixels are `1.0 x 1.0` in size, and a pixel
//! is only written to if the drawn geometry covers its center. For example, drawing a line from
//! `(0,0)` to `(1,0)` only touches the pixel's center and may not actually write to them.
//!
//! This is typically not a problem with "natural" data, but may result in some confusion when using
//! synthetic inputs.

/*

Impl notes:
- two primitives needed: lines and text.
- text is more difficult, but `epaint` can draw it and more

*/

use std::thread;

use ::epaint::{
    text::{LayoutJob, LayoutSection, TextFormat},
    FontFamily, FontId, Shape, Stroke, TextShape,
};

use crate::{draw::lines::Point, AsImageViewMut, Color, Gpu, ImageViewMut};

mod epaint;
pub(crate) mod lines;

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
        if thread::panicking() {
            return;
        }

        let start = self.image.position(self.start_x, self.start_y);
        let end = self.image.position(self.end_x, self.end_y);

        let color = self.color.to_linear();
        lines::draw(
            Gpu::get(),
            self.image.image,
            &[
                Point {
                    position: start,
                    color,
                },
                Point {
                    position: end,
                    color,
                },
            ],
        );
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
        color: Color::RED,
    }
}

/// Determines the vertical placement of a [`text`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VAlign {
    /// The top of the text is at the provided Y coordinate, the text is placed below it.
    Top,
    /// The vertical center of the text is at the provided Y coordinate, the text extends upwards and downwards from there.
    Center,
    /// The bottom of the text is at the provided Y coordinate, the text is drawn
    Bottom,
}

/// Determines the horizontal placement of a [`text`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HAlign {
    /// The left edge of the text is at the provided X coordinate, the text is placed to the right.
    Left,
    /// The center of the text is at the provided X coordinate.
    Center,
    /// The right edge of the text is at the provided X coordinate, the text is placed to the left.
    Right,
}

pub struct DrawText<'a> {
    image: ImageViewMut<'a>,
    x: f32,
    y: f32,
    text: &'a str,
    size: f32,
    color: Color,
    valign: VAlign,
    halign: HAlign,
}

impl<'a> DrawText<'a> {
    /// Sets the font height in points.
    pub fn size(&mut self, size: f32) -> &mut Self {
        self.size = size;
        self
    }

    /// Sets the text color.
    pub fn color(&mut self, color: Color) -> &mut Self {
        self.color = color;
        self
    }

    /// Sets the vertical alignment of the text.
    pub fn valign(&mut self, valign: VAlign) -> &mut Self {
        self.valign = valign;
        self
    }

    /// Sets the horizontal alignment of the text.
    pub fn halign(&mut self, halign: HAlign) -> &mut Self {
        self.halign = halign;
        self
    }
}

impl<'a> Drop for DrawText<'a> {
    fn drop(&mut self) {
        if thread::panicking() {
            return;
        }

        // `epaint` operates in pixel coordinates of the destination `Image`, not in clip space.
        // FIXME: perhaps all primitives should work like that? they all have access to the dest `Image` already.
        let pos = self.image.data.rect.transform_out(self.x, self.y);

        let font_id = FontId {
            size: self.size,
            family: FontFamily::Monospace,
        };
        let color = ::epaint::Rgba::from_srgba_unmultiplied(
            self.color.r(),
            self.color.g(),
            self.color.b(),
            self.color.a(),
        );
        let gpu = Gpu::get();
        let galley = gpu.fonts.layout_job(LayoutJob {
            text: self.text.to_string(),
            sections: vec![LayoutSection {
                leading_space: 0.0,
                byte_range: 0..self.text.len(),
                format: TextFormat {
                    font_id,
                    color: color.into(),
                    ..Default::default()
                },
            }],
            ..Default::default()
        });

        // Galleys have their top-left corner at 0,0. Adjust the position if different positioning
        // was requested.
        let x = match self.halign {
            HAlign::Left => pos[0],
            HAlign::Center => pos[0] - galley.size().x * 0.5,
            HAlign::Right => pos[0] - galley.size().x,
        };
        let y = match self.valign {
            VAlign::Top => pos[1],
            VAlign::Center => pos[1] - galley.size().y * 0.5,
            VAlign::Bottom => pos[1] - galley.size().y,
        };

        epaint::draw(
            gpu,
            self.image.image,
            [Shape::Text(TextShape {
                pos: [x, y].into(),
                galley,
                underline: Stroke::NONE,
                override_text_color: None,
                angle: 0.0,
            })],
        );
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
        size: 16.0,
        color: Color::WHITE,
        valign: VAlign::Center,
        halign: HAlign::Center,
    }
}

#[cfg(test)]
mod tests {
    use crate::Image;

    use super::*;

    #[test]
    fn test_line() {
        let mut image = Image::from_rgba8((2, 1), &[0; 8]);

        // Draw a line of the same color as the destination.
        line(&mut image, 0.0, 0.5, 2.0, 0.5).color(Color::NONE);
        image.with_data(|data| assert_eq!(data.to_vec(), &[0; 8]));

        // Draw a solid white line.
        // NB: a 1-pixel-wide line may not cover the pixel's sample location if drawn at Y=0.0, so we draw at Y=0.5
        line(&mut image, 0.0, 0.5, 2.0, 0.5).color(Color::WHITE);
        image.with_data(|data| assert_eq!(data.to_vec(), &[0xFF; 8]));

        // Draw a transparent black line over the second pixel.
        line(&mut image, 1.0, 0.5, 2.0, 0.5).color(Color::NONE);
        image.with_data(|data| {
            assert_eq!(
                data.to_vec(),
                &[0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00]
            )
        });
    }

    #[test]
    fn test_text() {
        // Hard to test without using reference images. Just draw something that fills a pixel.
        let mut image = Image::filled((1, 1), Color::NONE);
        text(&mut image, 0.5, 0.5, "■").size(4.0).color(Color::RED);
        image.with_data(|data| assert_eq!(data.to_vec(), &[0xFF, 0x00, 0x00, 0xFF]));
    }
}
