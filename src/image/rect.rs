use std::{
    fmt,
    ops::{Bound, RangeBounds},
};

use embedded_graphics::prelude::*;
use itertools::Itertools;

/// An axis-aligned rectangle.
///
/// This rectangle type uses (signed) integer coordinates and is meant to be used with the
/// [`crate::image`] module.
///
/// Rectangles are allowed to be degenerate, ie. to have zero height and/or width.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub(crate) rect: embedded_graphics::primitives::Rectangle,
}

impl Rect {
    /// Creates a rectangle extending outwards from a center point.
    pub fn from_center(x_center: i32, y_center: i32, width: u32, height: u32) -> Self {
        let top_left = Point {
            x: x_center - (width / 2) as i32,
            y: y_center - (height / 2) as i32,
        };

        Self {
            rect: embedded_graphics::primitives::Rectangle {
                top_left,
                size: Size { width, height },
            },
        }
    }

    /// Creates a rectangle extending downwards and right from a point.
    #[inline]
    pub fn from_top_left(top_left_x: i32, top_left_y: i32, width: u32, height: u32) -> Self {
        Self {
            rect: embedded_graphics::primitives::Rectangle {
                top_left: Point {
                    x: top_left_x,
                    y: top_left_y,
                },
                size: Size { width, height },
            },
        }
    }

    /// Constructs a [`Rect`] that spans a range of X and Y coordinates.
    pub fn from_ranges<X, Y>(x: X, y: Y) -> Self
    where
        X: RangeBounds<i32>,
        Y: RangeBounds<i32>,
    {
        fn cvt_lower_bound(b: Bound<&i32>) -> i32 {
            match b {
                Bound::Included(&min) => min,
                Bound::Excluded(&i32::MAX) | Bound::Unbounded => {
                    panic!("invalid rectangle range bound: {:?}", b)
                }
                Bound::Excluded(&v) => v + 1,
            }
        }

        fn cvt_upper_bound(b: Bound<&i32>) -> i32 {
            match b {
                Bound::Included(&max) => max,
                Bound::Excluded(&i32::MIN) | Bound::Unbounded => {
                    panic!("invalid rectangle range bound: {:?}", b)
                }
                Bound::Excluded(&v) => v - 1,
            }
        }

        let x_min = cvt_lower_bound(x.start_bound());
        let x_max = cvt_upper_bound(x.end_bound());
        let y_min = cvt_lower_bound(y.start_bound());
        let y_max = cvt_upper_bound(y.end_bound());

        Self::span_inner(x_min, y_min, x_max, y_max)
    }

    /// Creates a rectangle from top left and bottom right corner coordinates.
    pub fn from_corners(top_left: (i32, i32), bottom_right: (i32, i32)) -> Self {
        Self::span_inner(top_left.0, top_left.1, bottom_right.0, bottom_right.1)
    }

    fn span_inner(x_min: i32, y_min: i32, x_max: i32, y_max: i32) -> Self {
        assert!(x_min <= x_max);
        assert!(y_min <= y_max);

        Self {
            rect: embedded_graphics::primitives::Rectangle {
                top_left: Point { x: x_min, y: y_min },
                size: Size {
                    width: (x_max - x_min + 1) as _,
                    height: (y_max - y_min + 1) as _,
                },
            },
        }
    }

    /// Grows this rectangle by adding a margin on every side.
    ///
    /// # Panics
    ///
    /// This method will panic if the adding margin makes the rectangle's width or height overflow a
    /// `u32`, or if the resulting width or height would be less than 0.
    #[must_use]
    pub fn grow(&self, left: i32, right: i32, top: i32, bottom: i32) -> Self {
        Self {
            rect: embedded_graphics::primitives::Rectangle {
                top_left: Point {
                    x: self.rect.top_left.x - left,
                    y: self.rect.top_left.y - top,
                },
                size: Size {
                    width: (i64::from(self.rect.size.width) + i64::from(left) + i64::from(right))
                        .try_into()
                        .unwrap(),
                    height: (i64::from(self.rect.size.height) + i64::from(top) + i64::from(bottom))
                        .try_into()
                        .unwrap(),
                },
            },
        }
    }

    /// Grows this rectangle by relative margins, returning the new rectangle.
    ///
    /// `left` and `right` are fractions of the rectangle's width, `top` and
    /// `bottom` are fractions of the rectangle's height.
    #[must_use]
    pub fn grow_rel(&self, left: f32, right: f32, top: f32, bottom: f32) -> Self {
        let left = self.rect.size.width as f32 * left;
        let right = self.rect.size.width as f32 * right;
        let top = self.rect.size.height as f32 * top;
        let bottom = self.rect.size.height as f32 * bottom;
        self.grow(left as i32, right as i32, top as i32, bottom as i32)
    }

    /// Returns the X coordinate of the left side of the rectangle.
    #[inline]
    pub fn x(&self) -> i32 {
        self.rect.top_left.x
    }

    /// Returns the Y coordinate of the top side of the rectangle.
    #[inline]
    pub fn y(&self) -> i32 {
        self.rect.top_left.y
    }

    #[inline]
    pub fn width(&self) -> u32 {
        self.rect.size.width
    }

    #[inline]
    pub fn height(&self) -> u32 {
        self.rect.size.height
    }

    /// Computes the intersection of `self` and `other`.
    pub fn intersection(&self, other: &Rect) -> Rect {
        let rect = Rect::from_corners(
            (self.x().max(other.x()), self.y().max(other.y())),
            (
                (i64::from(self.x()) + i64::from(self.width()))
                    .min(i64::from(other.x()) + i64::from(other.width())) as i32
                    - 1,
                (i64::from(self.y()) + i64::from(self.height()))
                    .min(i64::from(other.y()) + i64::from(other.height())) as i32
                    - 1,
            ),
        );
        assert!(
            self.contains_rect(&rect),
            "intersect self={:?} other={:?} res={:?}",
            self,
            other,
            rect,
        );
        assert!(
            other.contains_rect(&rect),
            "intersect self={:?} other={:?} res={:?}",
            self,
            other,
            rect,
        );
        rect
    }

    /// Returns whether `self` contains `other`.
    pub fn contains_rect(&self, other: &Rect) -> bool {
        // TODO: specify behavior with 0-area rects
        self.x() <= other.x()
            && self.y() <= other.y()
            && i64::from(self.x()) + i64::from(self.width())
                >= i64::from(other.x()) + i64::from(other.width())
            && i64::from(self.y()) + i64::from(self.height())
                >= i64::from(other.y()) + i64::from(other.height())
    }

    /// Returns an iterator over all X,Y coordinates contained in this `Rect`.
    pub fn iter_coords(&self) -> impl Iterator<Item = (i64, i64)> {
        let x = i64::from(self.x());
        let y = i64::from(self.y());
        let w = i64::from(self.width());
        let h = i64::from(self.height());

        (x..x + w).cartesian_product(y..y + h)
    }
}

impl fmt::Debug for Rect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let x = self.rect.top_left.x;
        let y = self.rect.top_left.y;
        let w = self.rect.size.width;
        let h = self.rect.size.height;
        let bx = i64::from(x) + i64::from(w);
        let by = i64::from(y) + i64::from(h);
        write!(f, "Rect @ ({x},{y})-({bx},{by})/{w}x{h}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contains_rect() {
        let outer = Rect::from_top_left(-8, -8, 16, 16);
        assert!(outer.contains_rect(&outer));
        assert!(outer.contains_rect(&Rect::from_top_left(-8, -8, 15, 15)));
        assert!(outer.contains_rect(&Rect::from_top_left(-7, -7, 15, 15)));
        assert!(!outer.contains_rect(&Rect::from_top_left(-7, -8, 16, 16)));
        assert!(!outer.contains_rect(&Rect::from_top_left(-8, -7, 16, 16)));
        assert!(!outer.contains_rect(&Rect::from_top_left(-8, -8, 17, 16)));
        assert!(!outer.contains_rect(&Rect::from_top_left(-8, -8, 16, 17)));
        assert!(outer.contains_rect(&Rect::from_top_left(-8, -8, 10, 10)));
        assert!(!outer.contains_rect(&Rect::from_top_left(-9, -8, 10, 10)));
        assert!(!outer.contains_rect(&Rect::from_top_left(-8, -9, 10, 10)));
    }

    #[test]
    fn test_intersection() {
        assert_eq!(
            Rect::from_ranges(0..=10, 0..=10).intersection(&Rect::from_ranges(5..=5, 5..=5)),
            Rect::from_ranges(5..=5, 5..=5)
        );
        assert_eq!(
            Rect::from_ranges(5..=5, 5..=5).intersection(&Rect::from_ranges(0..=10, 0..=10)),
            Rect::from_ranges(5..=5, 5..=5)
        );
    }
}
