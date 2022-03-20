use std::convert::Infallible;

use embedded_graphics::{
    draw_target::DrawTarget,
    prelude::*,
    primitives::{Line, PrimitiveStyle, Rectangle},
};

use crate::image::{AsImageViewMut, Color, ImageViewMut, Rect};

/// Guard returned by [`draw_rect`]; draws the rectangle when dropped and allows customization.
pub struct DrawRect<'a> {
    image: ImageViewMut<'a>,
    rect: Rect,
    color: Color,
    stroke_width: u32,
}

impl DrawRect<'_> {
    /// Sets the rectangle's color.
    pub fn color(&mut self, color: Color) -> &mut Self {
        self.color = color;
        self
    }

    /// Sets the rectangle's stroke width.
    ///
    /// By default, a stroke width of 1 is used.
    pub fn stroke_width(&mut self, width: u32) -> &mut Self {
        self.stroke_width = width;
        self
    }
}

impl Drop for DrawRect<'_> {
    fn drop(&mut self) {
        match self
            .rect
            .rect
            .into_styled(PrimitiveStyle::with_stroke(self.color, self.stroke_width))
            .draw(&mut Target(self.image.as_view_mut()))
        {
            Ok(_) => {}
            Err(infallible) => match infallible {},
        }
    }
}

/// Guard returned by [`draw_marker`]; draws the marker when dropped and allows customization.
pub struct DrawMarker<'a> {
    image: ImageViewMut<'a>,
    x: i32,
    y: i32,
    color: Color,
    size: u32,
}

impl<'a> DrawMarker<'a> {
    /// Sets the marker's color.
    pub fn color(&mut self, color: Color) -> &mut Self {
        self.color = color;
        self
    }

    pub fn size(&mut self, size: u32) -> &mut Self {
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
                    x: self.x + xoff,
                    y: self.y + yoff,
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

/// Guard returned by [`draw_line`]; draws the line when dropped and allows customization.
pub struct DrawLine<'a> {
    image: ImageViewMut<'a>,
    start_x: i32,
    start_y: i32,
    end_x: i32,
    end_y: i32,
    color: Color,
    stroke_width: u32,
}

impl<'a> DrawLine<'a> {
    /// Sets the line's color.
    pub fn color(&mut self, color: Color) -> &mut Self {
        self.color = color;
        self
    }

    /// Sets the line's stroke width.
    ///
    /// By default, a stroke width of 1 is used.
    pub fn stroke_width(&mut self, width: u32) -> &mut Self {
        self.stroke_width = width;
        self
    }
}

impl<'a> Drop for DrawLine<'a> {
    fn drop(&mut self) {
        match Line::new(
            Point::new(self.start_x, self.start_y),
            Point::new(self.end_x, self.end_y),
        )
        .into_styled(PrimitiveStyle::with_stroke(self.color, self.stroke_width))
        .draw(&mut Target(self.image.reborrow()))
        {
            Ok(_) => {}
            Err(infallible) => match infallible {},
        }
    }
}

/// Draws a rectangle onto an image.
pub fn draw_rect<I: AsImageViewMut>(image: &mut I, rect: Rect) -> DrawRect<'_> {
    DrawRect {
        image: image.as_view_mut(),
        rect,
        color: Color::RED,
        stroke_width: 1,
    }
}

/// Draws a marker onto an image.
///
/// This can be used to visualize shape landmarks or points of interest.
pub fn draw_marker<I: AsImageViewMut>(image: &mut I, x: i32, y: i32) -> DrawMarker<'_> {
    DrawMarker {
        image: image.as_view_mut(),
        x,
        y,
        color: Color::from_rgb8(255, 0, 0),
        size: 5,
    }
}

/// Draws a line onto an image.
pub fn draw_line<I: AsImageViewMut>(
    image: &mut I,
    start_x: i32,
    start_y: i32,
    end_x: i32,
    end_y: i32,
) -> DrawLine<'_> {
    DrawLine {
        image: image.as_view_mut(),
        start_x,
        start_y,
        end_x,
        end_y,
        color: Color::from_rgb8(0, 0, 255),
        stroke_width: 1,
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
