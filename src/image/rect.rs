//! TODO: just make them use floats already

use std::{
    cmp, fmt,
    ops::{Bound, RangeBounds},
};

use embedded_graphics::prelude::*;
use itertools::Itertools;
use nalgebra::{Point2, Rotation2};

use crate::{num::TotalF32, resolution::AspectRatio};

/// An axis-aligned rectangle.
///
/// This rectangle type uses (signed) integer coordinates and is meant to be used with the
/// [`crate::image`] module.
///
/// Rectangles are allowed to have zero height and/or width.
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

    /// Creates a rectangle from two opposing corner points.
    pub fn from_corners(top_left: (i32, i32), bottom_right: (i32, i32)) -> Self {
        Self::span_inner(top_left.0, top_left.1, bottom_right.0, bottom_right.1)
    }

    /// Computes the (axis-aligned) bounding rectangle that encompasses `points`.
    ///
    /// Returns `None` if `points` is an empty iterator.
    pub fn bounding<I: IntoIterator<Item = (i32, i32)>>(points: I) -> Option<Self> {
        let mut iter = points.into_iter();

        let (x, y) = iter.next()?;
        let (mut x_min, mut x_max, mut y_min, mut y_max) = (x, x, y, y);

        for (x, y) in iter {
            x_min = cmp::min(x_min, x);
            x_max = cmp::max(x_max, x);
            y_min = cmp::min(y_min, y);
            y_max = cmp::max(y_max, y);
        }

        Some(Self::span_inner(x_min, y_min, x_max, y_max))
    }

    pub fn including(&self, x: i32, y: i32) -> Self {
        let mut x_min = self.x();
        let mut y_min = self.y();
        let mut x_max = self.x() + self.width() as i32 - 1;
        let mut y_max = self.y() + self.height() as i32 - 1;

        x_min = cmp::min(x_min, x);
        x_max = cmp::max(x_max, x);
        y_min = cmp::min(y_min, y);
        y_max = cmp::max(y_max, y);

        Self::span_inner(x_min, y_min, x_max, y_max)
    }

    fn span_inner(x_min: i32, y_min: i32, x_max: i32, y_max: i32) -> Self {
        assert!(x_min <= x_max, "x_min={}, x_max={}", x_min, x_max);
        assert!(y_min <= y_max, "y_min={}, y_max={}", y_min, y_max);
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

    /// Grows each side of this rectangle by adding a margin.
    ///
    /// # Panics
    ///
    /// This method will panic if the adding margin makes the rectangle's width or height overflow a
    /// `u32`, or if the resulting width or height would be less than 0.
    #[must_use]
    pub fn grow_sides(&self, left: i32, right: i32, top: i32, bottom: i32) -> Self {
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

    /// Grows this rectangle by adding a margin relative to width and height.
    ///
    /// `amount` is the relative amount of the rectangles width and height to add to each side.
    #[must_use]
    pub fn grow_rel(&self, amount: f32) -> Self {
        let left = self.rect.size.width as f32 * amount;
        let right = self.rect.size.width as f32 * amount;
        let top = self.rect.size.height as f32 * amount;
        let bottom = self.rect.size.height as f32 * amount;
        self.grow_sides(left as i32, right as i32, top as i32, bottom as i32)
    }

    /// Grows each side of this rectangle by a margin relative to the rectangles width or height.
    ///
    /// `left` and `right` are fractions of the rectangle's width, `top` and `bottom` are fractions
    /// of the rectangle's height.
    #[must_use]
    pub fn grow_sides_rel(&self, left: f32, right: f32, top: f32, bottom: f32) -> Self {
        let left = self.rect.size.width as f32 * left;
        let right = self.rect.size.width as f32 * right;
        let top = self.rect.size.height as f32 * top;
        let bottom = self.rect.size.height as f32 * bottom;
        self.grow_sides(left as i32, right as i32, top as i32, bottom as i32)
    }

    // FIXME: grow width and height by different amounts, grow width and height by the same amt

    /// Symmetrically extends one dimension of `self` so that the resulting rectangle has the given
    /// aspect ratio.
    ///
    /// # Panics
    ///
    /// This method will panic if `self` has a width or height of 0.
    #[must_use]
    pub fn grow_to_fit_aspect(&self, target_aspect: AspectRatio) -> Self {
        let mut res = *self;
        let target_width = (self.height() as f32 * target_aspect.as_f32() + 0.5) as u32;
        if target_width >= self.width() {
            let inc_w = target_width - self.width();
            res.rect.top_left.x -= (inc_w / 2) as i32;
            res.rect.size.width += inc_w;
        } else {
            let target_height = (self.width() as f32 / target_aspect.as_f32() + 0.5) as u32;
            let inc_h = target_height - self.height();
            res.rect.top_left.y -= (inc_h / 2) as i32;
            res.rect.size.height += inc_h;
        }

        res
    }

    pub fn grow_move_center(&self, x_center: i32, y_center: i32) -> Self {
        let w = cmp::max(
            (i64::from(x_center) - i64::from(self.x())).abs(),
            (i64::from(x_center) - (i64::from(self.x()) + i64::from(self.width()))).abs(),
        ) * 2;
        let h = cmp::max(
            (i64::from(y_center) - i64::from(self.y())).abs(),
            (i64::from(y_center) - (i64::from(self.y()) + i64::from(self.height()))).abs(),
        ) * 2;

        Self::from_center(
            x_center,
            y_center,
            w.try_into().unwrap(),
            h.try_into().unwrap(),
        )
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

    /// Returns the number of pixels contained in `self`.
    pub fn area(&self) -> u64 {
        u64::from(self.rect.size.width) * u64::from(self.rect.size.height)
    }

    pub fn center(&self) -> (f32, f32) {
        (
            self.x() as f32 + (self.width() as f32 / 2.0),
            self.y() as f32 + (self.height() as f32 / 2.0),
        )
    }

    #[must_use]
    pub fn move_by(&self, x: i32, y: i32) -> Rect {
        Rect::from_top_left(self.x() + x, self.y() + y, self.width(), self.height())
    }

    #[must_use]
    pub fn move_to(&self, x: i32, y: i32) -> Rect {
        Rect::from_top_left(x, y, self.width(), self.height())
    }

    /// Computes the intersection of `self` and `other`.
    ///
    /// Returns `None` when the intersection is empty (ie. the rectangles do not overlap).
    pub fn intersection(&self, other: &Rect) -> Option<Rect> {
        let x_min = self.x().max(other.x());
        let y_min = self.y().max(other.y());
        let x_max = (i64::from(self.x()) + i64::from(self.width()))
            .min(i64::from(other.x()) + i64::from(other.width())) as i32
            - 1;
        let y_max = (i64::from(self.y()) + i64::from(self.height()))
            .min(i64::from(other.y()) + i64::from(other.height())) as i32
            - 1;
        if x_min > x_max || y_min > y_max {
            return None;
        }
        let rect = Rect::from_corners((x_min, y_min), (x_max, y_max));
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
        Some(rect)
    }

    fn intersection_area(&self, other: &Self) -> u64 {
        self.intersection(other).map_or(0, |rect| rect.area())
    }

    fn union_area(&self, other: &Self) -> u64 {
        self.area() + other.area() - self.intersection_area(other)
    }

    /// Computes the Intersection over Union (IOU) of `self` and `other`.
    pub fn iou(&self, other: &Self) -> f32 {
        self.intersection_area(other) as f32 / self.union_area(other) as f32
    }

    pub fn contains_point(&self, x: i64, y: i64) -> bool {
        i64::from(self.x()) <= x
            && i64::from(self.y()) <= y
            && i64::from(self.x()) + i64::from(self.width()) > x
            && i64::from(self.y()) + i64::from(self.height()) > y
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

/// A [`Rect`], rotated around its center.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotatedRect {
    rect: Rect,
    radians: f32,
    sin: f32,
    cos: f32,
    inv_sin: f32,
    inv_cos: f32,
}

impl RotatedRect {
    /// Creates a new rotated rectangle.
    ///
    /// `radians` is the clockwise rotation to apply to the [`Rect`].
    #[inline]
    pub fn new(rect: Rect, radians: f32) -> Self {
        Self {
            rect,
            radians,
            sin: radians.sin(),
            cos: radians.cos(),
            inv_sin: (-radians).sin(),
            inv_cos: (-radians).cos(),
        }
    }

    /// Approximates the rotated bounding rectangle that encompasses `points`.
    ///
    /// Returns `None` if `points` is an empty iterator.
    pub fn bounding<I: IntoIterator<Item = (i32, i32)>>(radians: f32, points: I) -> Option<Self> {
        let mut points = points.into_iter().peekable();

        // Make sure we have at least 1 point.
        points.peek()?;

        // Approach: we rotate all points so that we can compute width, height, and center point of
        // the rectangle, then we rotate the center back into the right coordinate system.
        // Note that, since we rotate the center back into the original coordinate system before
        // using it, it doesn't matter what point we rotate everything around. We pick the origin
        // for convenience. Picking a point closer to the centroid of the points could potentially
        // reduce rounding errors, but for now, this works fine.

        let mut x_min = f32::MAX;
        let mut x_max = f32::MIN;
        let mut y_min = f32::MAX;
        let mut y_max = f32::MIN;
        for (x, y) in points {
            let [x, y] = [x as f32, y as f32];

            let [x, y] = [
                x * (-radians).cos() - y * (-radians).sin(),
                x * (-radians).sin() + y * (-radians).cos(),
            ];

            x_min = cmp::min(TotalF32(x_min), TotalF32(x)).0;
            x_max = cmp::max(TotalF32(x_max), TotalF32(x)).0;
            y_min = cmp::min(TotalF32(y_min), TotalF32(y)).0;
            y_max = cmp::max(TotalF32(y_max), TotalF32(y)).0;
        }

        // Center in rotated frame.
        let [cx, cy] = [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0];

        // Center in non-rotated frame (original coordinates).
        let [cx, cy] = [
            cx * radians.cos() - cy * radians.sin(),
            cx * radians.sin() + cy * radians.cos(),
        ];

        // NOTE: Adds 1.0 because rectangles are integer-valued and would "round out" points otherwise.
        let [w, h] = [x_max - x_min + 1.0, y_max - y_min + 1.0];

        let [x, y] = [cx - w / 2.0, cy - h / 2.0];

        Some(Self::new(
            Rect::from_top_left(x as i32, y as i32, w.ceil() as u32, h.ceil() as u32),
            radians,
        ))
    }

    /// Returns the rectangle's clockwise rotation in radians.
    #[inline]
    pub fn rotation_radians(&self) -> f32 {
        self.radians
    }

    /// Returns the rectangle's clockwise rotation in degrees.
    pub fn rotation_degrees(&self) -> f32 {
        self.radians.to_degrees()
    }

    /// Returns a reference to the underlying non-rotated rectangle.
    #[inline]
    pub fn rect(&self) -> &Rect {
        &self.rect
    }

    /// Sets the underlying non-rotated rectangle.
    #[inline]
    pub fn set_rect(&mut self, rect: Rect) {
        self.rect = rect;
    }

    /// Applies a closure to the underlying non-rotated [`Rect`].
    pub fn map(mut self, f: impl FnOnce(Rect) -> Rect) -> Self {
        self.rect = f(self.rect);
        self
    }

    /// Returns the rotated rectangle's corners.
    ///
    /// The order is: top-left, top-right, bottom-right, bottom-left, as seen from the non-rotated
    /// rect: after the rotation is applied, the corners can be rotated anywhere else, but the order
    /// is retained.
    pub fn rotated_corners(&self) -> [(f32, f32); 4] {
        let (x, y) = (self.rect.x() as f32, self.rect.y() as f32);
        let (w, h) = (self.rect.width() as f32, self.rect.height() as f32);
        let corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)];

        let rotation = Rotation2::new(self.radians);
        let (cx, cy) = self.rect.center();
        let center = Point2::new(cx, cy);
        corners.map(|(x, y)| {
            let point = Point2::new(x, y);
            let rel = point - center;
            let rot = rotation * rel;
            let abs = center + rot;
            (abs.x, abs.y)
        })
    }

    pub fn contains_point(&self, x: i32, y: i32) -> bool {
        let [x, y] = self.transform_in(x, y).map(i64::from);

        // The rect offset was already compensated for by the transform.
        self.rect.move_to(0, 0).contains_point(x, y)
    }

    /// Transforms a point from the parent coordinate system into the [`RotatedRect`]'s system.
    pub fn transform_in_f32(&self, x: f32, y: f32) -> [f32; 2] {
        let [x, y] = [
            x as f32 - self.rect.x() as f32,
            y as f32 - self.rect.y() as f32,
        ];
        let [cx, cy] = [
            self.rect.width() as f32 / 2.0,
            self.rect.height() as f32 / 2.0,
        ];
        // Offset by [0.5,0.5] to place us in pixel center.
        let [x, y] = [x - cx + 0.5, y - cy + 0.5];
        let [x, y] = [
            x * self.inv_cos - y * self.inv_sin + cx - 0.5,
            y * self.inv_cos + x * self.inv_sin + cy - 0.5,
        ];
        [x, y]
    }

    pub fn transform_in(&self, x: i32, y: i32) -> [i32; 2] {
        let [x, y] = self.transform_in_f32(x as f32, y as f32);
        [x.round() as i32, y.round() as i32]
    }

    /// Transforms a point from the [`RotatedRect`]'s coordinate system to the parent system.
    pub fn transform_out_f32(&self, x: f32, y: f32) -> [f32; 2] {
        let [cx, cy] = [
            self.rect.width() as f32 / 2.0,
            self.rect.height() as f32 / 2.0,
        ];
        // Offset by [0.5,0.5] to place us in pixel center.
        // FIXME: this should be done in the integer functions only, not here
        let [x, y] = [x - cx + 0.5, y - cy + 0.5];
        let [x, y] = [
            x * self.cos - y * self.sin + cx - 0.5,
            y * self.cos + x * self.sin + cy - 0.5,
        ];
        [x + self.rect.x() as f32, y + self.rect.y() as f32]
    }

    pub fn transform_out(&self, x: i32, y: i32) -> [i32; 2] {
        let [x, y] = self.transform_out_f32(x as f32, y as f32);
        [x.round() as i32, y.round() as i32]
    }
}

impl From<Rect> for RotatedRect {
    fn from(rect: Rect) -> Self {
        Self::new(rect, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::TAU;

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
    fn test_contains_point() {
        let rect = Rect::from_top_left(-5, 5, 10, 5);
        assert!(rect.contains_point(-5, 5));
        assert!(rect.contains_point(-5 + 9, 5 + 4));
        assert!(!rect.contains_point(-5 + 9 + 1, 5 + 4));
        assert!(!rect.contains_point(-5 + 9, 5 + 4 + 1));

        let empty = Rect::from_center(0, 0, 0, 0);
        assert!(!empty.contains_point(0, 0));
        assert!(!empty.contains_point(0, 1));
        assert!(!empty.contains_point(0, -1));
    }

    #[test]
    fn test_intersection() {
        assert_eq!(
            Rect::from_ranges(0..=10, 0..=10).intersection(&Rect::from_ranges(5..=5, 5..=5)),
            Some(Rect::from_ranges(5..=5, 5..=5))
        );
        assert_eq!(
            Rect::from_ranges(5..=5, 5..=5).intersection(&Rect::from_ranges(0..=10, 0..=10)),
            Some(Rect::from_ranges(5..=5, 5..=5))
        );
        assert_eq!(
            Rect::from_ranges(5..=5, 5..=5).intersection(&Rect::from_ranges(6..=10, 0..=10)),
            None,
        );
    }

    #[test]
    fn test_bounding() {
        assert_eq!(
            Rect::bounding([(0, 0), (1, 1), (-1, -1)]).unwrap(),
            Rect::from_corners((-1, -1), (1, 1)),
        );
        assert_eq!(
            Rect::bounding([(1, 1), (-1, -1)]).unwrap(),
            Rect::from_corners((-1, -1), (1, 1)),
        );
        assert_eq!(
            Rect::bounding([(-1, -1), (1, 1)]).unwrap(),
            Rect::from_corners((-1, -1), (1, 1)),
        );
        assert_eq!(
            Rect::bounding([(1, 1), (2, 2)]).unwrap(),
            Rect::from_corners((1, 1), (2, 2)),
        );
        assert_eq!(
            Rect::bounding([(0, 0), (10, 0)]).unwrap(),
            Rect::from_top_left(0, 0, 11, 1),
        );
    }

    #[test]
    fn test_including() {
        let zero = Rect::from_corners((0, 0), (0, 0));
        assert_eq!(zero.including(0, 0), Rect::from_corners((0, 0), (0, 0)));
        assert_eq!(zero.including(1, 1), Rect::from_corners((0, 0), (1, 1)));
        assert_eq!(zero.including(-1, -1), Rect::from_corners((-1, -1), (0, 0)));
    }

    #[test]
    fn test_fit_aspect() {
        assert_eq!(
            Rect::from_center(10, 10, 50, 100).grow_to_fit_aspect(AspectRatio::SQUARE),
            Rect::from_center(10, 10, 100, 100),
        );
        assert_eq!(
            Rect::from_center(10, 10, 100, 50).grow_to_fit_aspect(AspectRatio::SQUARE),
            Rect::from_center(10, 10, 100, 100),
        );
        assert_eq!(
            Rect::from_center(10, 10, 100, 98).grow_to_fit_aspect(AspectRatio::SQUARE),
            Rect::from_center(10, 10, 100, 100),
        );
    }

    #[test]
    fn test_grow_move_center() {
        let orig = Rect::from_top_left(0, 0, 0, 0);
        assert_eq!(orig.grow_move_center(0, 0), orig);
        assert_eq!(orig.grow_move_center(1, 0), Rect::from_top_left(0, 0, 2, 0));
    }

    #[test]
    fn test_rotated_rect_transform() {
        // Not actually rotated
        let null = RotatedRect::new(Rect::from_top_left(0, 0, 1, 1), 0.0);
        assert_eq!(null.transform_in(0, 0), [0, 0]);
        assert_eq!(null.transform_out(0, 0), [0, 0]);

        assert_eq!(null.transform_in(1, -1), [1, -1]);
        assert_eq!(null.transform_out(1, -1), [1, -1]);

        let offset = RotatedRect::new(Rect::from_top_left(10, 20, 1, 1), 0.0);
        assert_eq!(offset.transform_in(0, 0), [-10, -20]);
        assert_eq!(offset.transform_in(10, 20), [0, 0]);

        // Rotated clockwise by 90°
        let right = RotatedRect::new(Rect::from_top_left(0, 0, 1, 1), TAU / 4.0);
        assert_eq!(right.transform_in(0, 0), [0, 0]);
        assert_eq!(right.transform_out(0, 0), [0, 0]);

        assert_eq!(right.transform_in(1, 0), [0, -1],);
        assert_eq!(right.transform_out(0, -1), [1, 0]);

        // Offset, rotated by 180°
        let rect = RotatedRect::new(Rect::from_top_left(10, 20, 1, 1), TAU / 2.0);
        assert_eq!(rect.transform_in(10, 20), [0, 0]);
        assert_eq!(rect.transform_out(0, 0), [10, 20]);
    }

    #[test]
    fn test_rotated_rect_contains_point() {
        // 1x1 rect at origin
        let rect = RotatedRect::new(Rect::from_top_left(0, 0, 1, 1), 1.0);
        assert!(rect.contains_point(0, 0));
        assert!(!rect.contains_point(0, 1));
        assert!(!rect.contains_point(1, 1));
        assert!(!rect.contains_point(1, 0));
        assert!(!rect.contains_point(0, -1));

        // 1x1 rect offset
        let rect = RotatedRect::new(Rect::from_top_left(10, 20, 1, 1), 1.0);
        assert!(rect.contains_point(10, 20));
        assert!(!rect.contains_point(9, 20));
        assert!(!rect.contains_point(10, 21));

        // Wide rect, flipped
        let rect = RotatedRect::new(Rect::from_top_left(10, 20, 100, 1), TAU / 2.0);
        assert!(!rect.contains_point(-20, 20));
        assert!(!rect.contains_point(9, 20));
        assert!(rect.contains_point(10, 20));
        assert!(rect.contains_point(100, 20));
        assert!(rect.contains_point(55, 20));
        assert!(!rect.contains_point(55, 21));
        assert!(!rect.contains_point(55, 19));

        // Wide rect, rotated 90°
        let rect = RotatedRect::new(Rect::from_center(0, 0, 51, 1), TAU / 4.0);
        assert!(rect.contains_point(0, 0));
        assert!(rect.contains_point(0, 1));
        assert!(rect.contains_point(0, 25));
        assert!(!rect.contains_point(0, 26));
        assert!(rect.contains_point(0, -1));
        assert!(rect.contains_point(0, -25));
        assert!(!rect.contains_point(0, -26));
        assert!(!rect.contains_point(1, 0));
        assert!(!rect.contains_point(-1, 0));

        // Wide rect, offset, rotated 90°
        let rect = RotatedRect::new(Rect::from_center(10, 10, 51, 1), TAU / 4.0);
        assert!(rect.contains_point(10, 0));
        assert!(rect.contains_point(10, 1));
        assert!(rect.contains_point(10, 35));
        assert!(!rect.contains_point(10, 36));
        assert!(rect.contains_point(10, -1));
        assert!(rect.contains_point(10, -15));
        assert!(!rect.contains_point(10, -16));
        assert!(!rect.contains_point(11, 0));
        assert!(!rect.contains_point(9, 0));
    }

    #[test]
    fn test_rotated_rect_bounding() {
        #[track_caller]
        fn bounding<I: IntoIterator<Item = (i32, i32)>>(radians: f32, points: I) -> RotatedRect
        where
            I::IntoIter: Clone,
        {
            let points = points.into_iter();
            let rect = RotatedRect::bounding(radians, points.clone()).unwrap();

            for (x, y) in points {
                assert!(
                    rect.contains_point(x, y),
                    "{rect:?} does not contain {x},{y}"
                );
            }

            rect
        }

        assert!(RotatedRect::bounding(0.0, []).is_none());

        assert_eq!(
            bounding(0.0, [(0, 0), (1, 1)]),
            Rect::from_top_left(0, 0, 2, 2).into(),
        );
        assert_eq!(
            bounding(0.0, [(0, 0), (10, 0)]),
            Rect::from_top_left(0, 0, 11, 1).into(),
        );
        assert_eq!(
            bounding(TAU / 2.0, [(0, 0), (1, 1)]),
            RotatedRect::new(Rect::from_top_left(0, 0, 2, 2), TAU / 2.0),
        );
        assert_eq!(
            bounding(TAU / 4.0, [(0, 0), (1, 1)]),
            RotatedRect::new(Rect::from_top_left(0, 0, 2, 2), TAU / 4.0),
        );
        assert_eq!(
            bounding(TAU / 4.0, [(0, 0), (9, 9)]),
            RotatedRect::new(Rect::from_top_left(0, 0, 10, 10), TAU / 4.0),
        );
        assert_eq!(
            bounding(TAU / 3.0, [(0, 0), (9, 9)]),
            RotatedRect::new(Rect::from_top_left(2, -2, 5, 14), TAU / 3.0),
        );
    }
}
