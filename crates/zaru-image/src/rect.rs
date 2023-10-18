//! Rectangle types.
//!
//! These are used throughout the library for image manipulation, object detection, regions of
//! interest, etc.

use std::{cmp, fmt, ops::RangeInclusive};

use crate::num::TotalF32;
use crate::AspectRatio;
use nalgebra::{Point2, Rotation2};
use zaru_linalg::{vec2, Vec2f};

/// An axis-aligned rectangle.
///
/// Rectangles are allowed to have zero height and/or width. Negative dimensions are not allowed.
#[derive(Clone, Copy, PartialEq)]
pub struct Rect {
    xc: f32,
    yc: f32,
    w: f32,
    h: f32,
}

impl Rect {
    /// Creates a rectangle extending outwards from a center point.
    #[inline]
    pub fn from_center(x_center: f32, y_center: f32, width: f32, height: f32) -> Self {
        Self {
            xc: x_center,
            yc: y_center,
            w: width,
            h: height,
        }
    }

    /// Creates a rectangle extending downwards and right from a point.
    #[inline]
    pub fn from_top_left(top_left_x: f32, top_left_y: f32, width: f32, height: f32) -> Self {
        Self::from_center(
            top_left_x + width * 0.5,
            top_left_y + height * 0.5,
            width,
            height,
        )
    }

    /// Constructs a [`Rect`] that spans a range of X and Y coordinates.
    pub fn from_ranges(x: RangeInclusive<f32>, y: RangeInclusive<f32>) -> Self {
        Self::span_inner(*x.start(), *y.start(), *x.end(), *y.end())
    }

    /// Computes the (axis-aligned) bounding rectangle that encompasses `points`.
    ///
    /// Returns [`None`] if `points` is an empty iterator.
    pub fn bounding<I: IntoIterator<Item = T>, T: Into<Vec2f>>(points: I) -> Option<Self> {
        let mut iter = points.into_iter();

        let first: Vec2f = iter.next()?.into();
        let (mut min, mut max) = (first, first);

        for pt in iter {
            let pt = pt.into();
            min = min.min(pt);
            max = max.max(pt);
        }

        Some(Self::span_inner(min.x, min.y, max.x, max.y))
    }

    fn span_inner(x_min: f32, y_min: f32, x_max: f32, y_max: f32) -> Self {
        assert!(x_min <= x_max, "x_min={}, x_max={}", x_min, x_max);
        assert!(y_min <= y_max, "y_min={}, y_max={}", y_min, y_max);
        Self::from_top_left(x_min, y_min, x_max - x_min, y_max - y_min)
    }

    /// Scales the width and height of this [`Rect`] by the given amount.
    ///
    /// The center position of the [`Rect`] remains the same.
    pub fn scale(&self, scale: f32) -> Self {
        Self {
            xc: self.xc,
            yc: self.yc,
            w: self.w * scale,
            h: self.h * scale,
        }
    }

    /// Grows this rectangle by adding a margin relative to width and height.
    ///
    /// `amount` is the relative amount of the rectangles width and height to add to each side.
    #[must_use]
    pub fn grow_rel(&self, amount: f32) -> Self {
        let left = self.width() * amount;
        let right = self.width() * amount;
        let top = self.height() * amount;
        let bottom = self.height() * amount;
        Rect {
            w: self.w + left + right,
            h: self.h + top + bottom,
            ..*self
        }
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
        let target_width = self.height() * target_aspect.as_f32();
        if target_width >= self.width() {
            let inc_w = target_width - self.width();
            res.w += inc_w;
        } else {
            let target_height = self.width() / target_aspect.as_f32();
            let inc_h = target_height - self.height();
            res.h += inc_h;
        }

        res
    }

    /// Moves this rectangle's center to the given coordinates, ensuring that the resulting [`Rect`]
    /// still contains all points in the original area.
    pub fn grow_move_center(&self, x_center: f32, y_center: f32) -> Self {
        let w = f32::max(
            (x_center - self.x()).abs(),
            (x_center - (self.x() + self.width())).abs(),
        ) * 2.0;
        let h = f32::max(
            (y_center - self.y()).abs(),
            (y_center - (self.y() + self.height())).abs(),
        ) * 2.0;

        Self::from_center(x_center, y_center, w, h)
    }

    #[inline]
    pub fn x_center(&self) -> f32 {
        self.xc
    }

    #[inline]
    pub fn y_center(&self) -> f32 {
        self.yc
    }

    #[inline]
    pub fn top_left(&self) -> Vec2f {
        vec2(self.xc - self.w * 0.5, self.yc - self.h * 0.5)
    }

    /// Returns the X coordinate of the left side of the rectangle.
    #[inline]
    pub fn x(&self) -> f32 {
        self.xc - self.w * 0.5
    }

    /// Returns the Y coordinate of the top side of the rectangle.
    #[inline]
    pub fn y(&self) -> f32 {
        self.yc - self.h * 0.5
    }

    #[inline]
    pub fn width(&self) -> f32 {
        self.w
    }

    #[inline]
    pub fn height(&self) -> f32 {
        self.h
    }

    /// Returns the number of pixels contained in `self`.
    pub fn area(&self) -> f32 {
        self.w * self.h
    }

    pub fn center(&self) -> Vec2f {
        vec2(self.xc, self.yc)
    }

    pub fn size(&self) -> Vec2f {
        vec2(self.w, self.h)
    }

    #[must_use]
    pub fn move_by(&self, x: f32, y: f32) -> Rect {
        Rect {
            xc: self.xc + x,
            yc: self.yc + y,
            ..*self
        }
    }

    #[must_use]
    pub fn move_to(&self, x: f32, y: f32) -> Rect {
        Rect::from_top_left(x, y, self.width(), self.height())
    }

    /// Computes the intersection of `self` and `other`.
    ///
    /// Returns [`None`] when the intersection is empty (ie. the rectangles do not overlap).
    pub fn intersection(&self, other: &Rect) -> Option<Rect> {
        let min = self.top_left().max(other.top_left());
        let max = (self.top_left() + self.size()).min(other.top_left() + other.size());
        if min.x > max.x || min.y > max.y {
            return None;
        }

        Some(Rect::bounding([min, max]).unwrap())
    }

    fn intersection_area(&self, other: &Self) -> f32 {
        self.intersection(other).map_or(0.0, |rect| rect.area())
    }

    fn union_area(&self, other: &Self) -> f32 {
        self.area() + other.area() - self.intersection_area(other)
    }

    /// Computes the Intersection over Union (IOU) of `self` and `other`.
    pub fn iou(&self, other: &Self) -> f32 {
        self.intersection_area(other) / self.union_area(other)
    }

    pub fn contains_point(&self, x: f32, y: f32) -> bool {
        self.x() <= x
            && self.y() <= y
            && self.x() + self.width() >= x
            && self.y() + self.height() >= y
    }

    pub fn corners(&self) -> [[f32; 2]; 4] {
        let [x, y] = [self.x(), self.y()];
        let [w, h] = [self.width(), self.height()];
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    }
}

impl fmt::Debug for Rect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Rect { xc, yc, h, w } = self;
        write!(f, "Rect @ ({xc},{yc})/{w}x{h}")
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
    pub fn bounding<I: IntoIterator<Item = [f32; 2]>>(radians: f32, points: I) -> Option<Self> {
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
        for [x, y] in points {
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

        let [w, h] = [x_max - x_min, y_max - y_min];

        let [x, y] = [cx - w / 2.0, cy - h / 2.0];

        Some(Self::new(Rect::from_top_left(x, y, w, h), radians))
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

    pub fn x_center(&self) -> f32 {
        self.rect.x_center()
    }

    pub fn y_center(&self) -> f32 {
        self.rect.y_center()
    }

    pub fn center(&self) -> [f32; 2] {
        self.rect.center().into()
    }

    /// Grows this rectangle by adding a margin relative to width and height.
    ///
    /// `amount` is the relative amount of the rectangles width and height to add to each side.
    #[must_use]
    pub fn grow_rel(&self, amount: f32) -> Self {
        self.map(|rect| rect.grow_rel(amount))
    }

    /// Symmetrically extends one dimension of `self` so that the resulting rectangle has the given
    /// aspect ratio.
    ///
    /// # Panics
    ///
    /// This method will panic if `self` has a width or height of 0.
    #[must_use]
    pub fn grow_to_fit_aspect(&self, target_aspect: AspectRatio) -> Self {
        self.map(|rect| rect.grow_to_fit_aspect(target_aspect))
    }

    /// Returns the rotated rectangle's corners.
    ///
    /// The order is: top-left, top-right, bottom-right, bottom-left, as seen from the non-rotated
    /// rect: after the rotation is applied, the corners can be rotated anywhere else, but the order
    /// is retained.
    pub fn rotated_corners(&self) -> [[f32; 2]; 4] {
        let corners = self.rect.corners();

        let rotation = Rotation2::new(self.radians);
        let center = Point2::from(self.rect.center().into_array());
        corners.map(|[x, y]| {
            let point = Point2::new(x, y);
            let rel = point - center;
            let rot = rotation * rel;
            let abs = center + rot;
            [abs.x, abs.y]
        })
    }

    pub fn contains_point(&self, x: f32, y: f32) -> bool {
        let [x, y] = self.transform_in(x, y);

        // The rect offset was already compensated for by the transform.
        self.rect.move_to(0.0, 0.0).contains_point(x, y)
    }

    /// Transforms a point from the parent coordinate system into the [`RotatedRect`]'s system.
    pub fn transform_in(&self, x: f32, y: f32) -> [f32; 2] {
        let [x, y] = [x - self.rect.x(), y - self.rect.y()];
        let [cx, cy] = [self.rect.width() / 2.0, self.rect.height() / 2.0];
        let [x, y] = [x - cx, y - cy];
        let [x, y] = [
            x * self.inv_cos - y * self.inv_sin + cx,
            y * self.inv_cos + x * self.inv_sin + cy,
        ];
        [x, y]
    }

    /// Transforms a point from the [`RotatedRect`]'s coordinate system to the parent system.
    pub fn transform_out(&self, x: f32, y: f32) -> [f32; 2] {
        let [cx, cy] = [self.rect.width() / 2.0, self.rect.height() / 2.0];
        let [x, y] = [x - cx, y - cy];
        let [x, y] = [
            x * self.cos - y * self.sin + cx,
            y * self.cos + x * self.sin + cy,
        ];
        [x + self.rect.x(), y + self.rect.y()]
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

    use approx::{assert_relative_eq, AbsDiffEq, RelativeEq};

    use super::*;

    macro_rules! assert_approx_eq {
        ($lhs:expr, $rhs:expr) => {
            ::approx::assert_relative_eq!(&$lhs[..], &$rhs[..], epsilon = 1e-7)
        };
    }

    impl AbsDiffEq for RotatedRect {
        type Epsilon = f32;

        fn default_epsilon() -> Self::Epsilon {
            f32::default_epsilon()
        }

        fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
            self.rect.xc.abs_diff_eq(&other.rect.xc, epsilon)
                && self.rect.yc.abs_diff_eq(&other.rect.yc, epsilon)
                && self.rect.w.abs_diff_eq(&other.rect.w, epsilon)
                && self.rect.h.abs_diff_eq(&other.rect.h, epsilon)
                && self.radians.abs_diff_eq(&other.radians, epsilon)
                && self.sin.abs_diff_eq(&other.sin, epsilon)
                && self.cos.abs_diff_eq(&other.cos, epsilon)
                && self.inv_sin.abs_diff_eq(&other.inv_sin, epsilon)
                && self.inv_cos.abs_diff_eq(&other.inv_cos, epsilon)
        }
    }
    impl RelativeEq for RotatedRect {
        fn default_max_relative() -> Self::Epsilon {
            f32::default_max_relative()
        }

        fn relative_eq(
            &self,
            other: &Self,
            epsilon: Self::Epsilon,
            max_relative: Self::Epsilon,
        ) -> bool {
            self.rect
                .xc
                .relative_eq(&other.rect.xc, epsilon, max_relative)
                && self
                    .rect
                    .yc
                    .relative_eq(&other.rect.yc, epsilon, max_relative)
                && self
                    .rect
                    .w
                    .relative_eq(&other.rect.w, epsilon, max_relative)
                && self
                    .rect
                    .h
                    .relative_eq(&other.rect.h, epsilon, max_relative)
                && self
                    .radians
                    .relative_eq(&other.radians, epsilon, max_relative)
                && self.sin.relative_eq(&other.sin, epsilon, max_relative)
                && self.cos.relative_eq(&other.cos, epsilon, max_relative)
                && self
                    .inv_sin
                    .relative_eq(&other.inv_sin, epsilon, max_relative)
                && self
                    .inv_cos
                    .relative_eq(&other.inv_cos, epsilon, max_relative)
        }
    }

    #[test]
    fn test_contains_point() {
        let rect = Rect::from_top_left(-5.0, 5.0, 10.0, 5.0);
        assert!(rect.contains_point(-5.0, 5.0));
        assert!(rect.contains_point(-5.0 + 9.0, 5.0 + 4.0));
        assert!(!rect.contains_point(-5.0 + 11.0, 5.0 + 4.0));
        assert!(!rect.contains_point(-5.0 + 9.0, 5.0 + 5.0 + 1.0));

        let empty = Rect::from_center(0.0, 0.0, 0.0, 0.0);
        assert!(!empty.contains_point(0.0025, 0.0));
        assert!(!empty.contains_point(0.0, 1.0));
        assert!(!empty.contains_point(0.0, -1.0));
    }

    #[test]
    fn test_intersection() {
        assert_eq!(
            Rect::from_ranges(0.0..=10.0, 0.0..=10.0)
                .intersection(&Rect::from_ranges(5.0..=5.0, 5.0..=5.0)),
            Some(Rect::from_ranges(5.0..=5.0, 5.0..=5.0))
        );
        assert_eq!(
            Rect::from_ranges(5.0..=5.0, 5.0..=5.0)
                .intersection(&Rect::from_ranges(0.0..=10.0, 0.0..=10.0)),
            Some(Rect::from_ranges(5.0..=5.0, 5.0..=5.0))
        );
        assert_eq!(
            Rect::from_ranges(5.0..=5.0, 5.0..=5.0)
                .intersection_area(&Rect::from_ranges(6.0..=10.0, 0.0..=10.0)),
            0.0,
        );
    }

    #[test]
    fn test_geom_zero() {
        let zero = Rect::from_center(0.0, 0.0, 0.0, 0.0);
        assert_eq!(zero.area(), 0.0);

        let also_zero = Rect::from_center(1.0, 0.0, 0.0, 0.0);
        assert_eq!(also_zero.area(), 0.0);

        assert_eq!(zero.intersection_area(&also_zero), 0.0);
        assert_eq!(zero.union_area(&also_zero), 0.0);
    }

    #[test]
    fn test_iou() {
        // Two rects with the same center point, but different sizes.
        let smaller = Rect::from_center(9.0, 9.0, 1.0, 1.0);
        let bigger = Rect::from_center(9.0, 9.0, 2.0, 2.0);

        assert_eq!(smaller.area(), 1.0);
        assert_eq!(bigger.area(), 4.0);

        let intersection = smaller.intersection(&bigger).unwrap();
        assert_eq!(intersection.xc, smaller.xc);
        assert_eq!(intersection.yc, smaller.yc);
        assert_eq!(intersection.w, smaller.w);
        assert_eq!(intersection.h, smaller.h);

        assert_eq!(
            smaller.intersection_area(&bigger),
            bigger.intersection_area(&smaller),
        );
        assert_eq!(smaller.intersection_area(&bigger), 1.0);
        assert_eq!(smaller.union_area(&bigger), bigger.union_area(&smaller));
        assert_eq!(smaller.union_area(&bigger), 4.0);

        assert_eq!(smaller.iou(&bigger), 1.0 / 4.0);
        assert_eq!(bigger.iou(&smaller), 1.0 / 4.0);
    }

    #[test]
    fn test_bounding() {
        assert_eq!(
            Rect::bounding([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]]).unwrap(),
            Rect::from_center(0.0, 0.0, 2.0, 2.0),
        );
        assert_eq!(
            Rect::bounding([[1.0, 1.0], [-1.0, -1.0]]).unwrap(),
            Rect::from_center(0.0, 0.0, 2.0, 2.0),
        );
        assert_eq!(
            Rect::bounding([[-1.0, -1.0], [1.0, 1.0]]).unwrap(),
            Rect::from_center(0.0, 0.0, 2.0, 2.0),
        );
        assert_eq!(
            Rect::bounding([[1.0, 1.0], [2.0, 2.0]]).unwrap(),
            Rect::from_center(1.5, 1.5, 1.0, 1.0),
        );
        assert_eq!(
            Rect::bounding([[0.0, 0.0], [10.0, 0.0]]).unwrap(),
            Rect::from_center(5.0, 0.0, 10.0, 0.0),
        );
    }

    #[test]
    fn test_fit_aspect() {
        assert_eq!(
            Rect::from_center(10.0, 10.0, 50.0, 100.0).grow_to_fit_aspect(AspectRatio::SQUARE),
            Rect::from_center(10.0, 10.0, 100.0, 100.0),
        );
        assert_eq!(
            Rect::from_center(10.0, 10.0, 100.0, 50.0).grow_to_fit_aspect(AspectRatio::SQUARE),
            Rect::from_center(10.0, 10.0, 100.0, 100.0),
        );
        assert_eq!(
            Rect::from_center(10.0, 10.0, 100.0, 98.0).grow_to_fit_aspect(AspectRatio::SQUARE),
            Rect::from_center(10.0, 10.0, 100.0, 100.0),
        );
    }

    #[test]
    fn test_grow_move_center() {
        let orig = Rect::from_top_left(0.0, 0.0, 0.0, 0.0);
        assert_eq!(orig.grow_move_center(0.0, 0.0), orig);
        assert_eq!(
            orig.grow_move_center(1.0, 0.0),
            Rect::from_top_left(0.0, 0.0, 2.0, 0.0)
        );
    }

    #[test]
    fn test_rotated_rect_transform() {
        // Not actually rotated
        let null = RotatedRect::new(Rect::from_top_left(0.0, 0.0, 1.0, 1.0), 0.0);
        assert_eq!(null.transform_in(0.0, 0.0), [0.0, 0.0]);
        assert_eq!(null.transform_out(0.0, 0.0), [0.0, 0.0]);

        assert_eq!(null.transform_in(1.0, -1.0), [1.0, -1.0]);
        assert_eq!(null.transform_out(1.0, -1.0), [1.0, -1.0]);

        let offset = RotatedRect::new(Rect::from_top_left(10.0, 20.0, 1.0, 1.0), 0.0);
        assert_eq!(offset.transform_in(0.0, 0.0), [-10.0, -20.0]);
        assert_eq!(offset.transform_in(10.0, 20.0), [0.0, 0.0]);

        // Rotated clockwise by 90°
        let right = RotatedRect::new(Rect::from_top_left(0.0, 0.0, 1.0, 1.0), TAU / 4.0);
        assert_eq!(right.transform_in(0.5, 0.5), [0.5, 0.5]);
        assert_eq!(right.transform_out(0.5, 0.5), [0.5, 0.5]);
        assert_approx_eq!(right.transform_in(0.0, 0.0), [0.0, 1.0]);
        assert_approx_eq!(right.transform_out(0.0, 0.0), [1.0, 0.0]);

        assert_approx_eq!(right.transform_in(1.0, 0.0), [0.0, 0.0]);
        assert_approx_eq!(right.transform_out(0.0, -1.0), [2.0, 0.0]);

        // Offset, rotated by 180°
        let rect = RotatedRect::new(Rect::from_top_left(10.0, 20.0, 1.0, 1.0), TAU / 2.0);
        assert_approx_eq!(rect.transform_in(10.0, 20.0), [1.0, 1.0]);
        assert_approx_eq!(rect.transform_in(11.0, 21.0), [0.0, 0.0]);
        assert_approx_eq!(rect.transform_out(0.0, 0.0), [11.0, 21.0]);
    }

    #[test]
    fn test_rotated_rect_contains_point() {
        // 1x1 rect at origin
        let rect = RotatedRect::new(Rect::from_top_left(0.0, 0.0, 1.0, 1.0), 1.0);
        assert!(rect.contains_point(0.5, 0.5));
        assert!(!rect.contains_point(0.0, 1.5));
        assert!(!rect.contains_point(1.0, 1.0));
        assert!(!rect.contains_point(1.0, 0.0));
        assert!(!rect.contains_point(0.0, -1.0));

        // 1x1 rect offset
        let rect = RotatedRect::new(Rect::from_top_left(10.0, 20.0, 1.0, 1.0), 1.0);
        assert!(rect.contains_point(10.5, 20.5));
        assert!(!rect.contains_point(9.0, 20.0));
        assert!(!rect.contains_point(10.0, 21.5));

        // Wide rect, flipped
        let rect = RotatedRect::new(Rect::from_top_left(10.0, 20.0, 100.0, 1.0), TAU / 2.0);
        assert!(!rect.contains_point(-20.0, 20.5));
        assert!(!rect.contains_point(9.0, 20.5));
        assert!(rect.contains_point(10.0, 20.5));
        assert!(rect.contains_point(100.0, 20.00005));
        assert!(rect.contains_point(55.0, 20.5));
        assert!(!rect.contains_point(55.0, 21.0));
        assert!(!rect.contains_point(55.0, 19.0));

        // Wide rect, rotated 90°
        let rect = RotatedRect::new(Rect::from_center(0.0, 0.0, 51.0, 1.0), TAU / 4.0);
        assert!(rect.contains_point(0.0, 0.0));
        assert!(rect.contains_point(0.0, 1.0));
        assert!(rect.contains_point(0.0, 25.0));
        assert!(!rect.contains_point(0.0, 26.0));
        assert!(rect.contains_point(0.0, -1.0));
        assert!(rect.contains_point(0.0, -25.0));
        assert!(!rect.contains_point(0.0, -26.0));
        assert!(!rect.contains_point(1.0, 0.0));
        assert!(!rect.contains_point(-1.0, 0.0));

        // Wide rect, offset, rotated 90°
        let rect = RotatedRect::new(Rect::from_center(10.0, 10.0, 51.0, 1.0), TAU / 4.0);
        assert!(rect.contains_point(10.0, 0.0));
        assert!(rect.contains_point(10.0, 1.0));
        assert!(rect.contains_point(10.0, 35.0));
        assert!(!rect.contains_point(10.0, 36.0));
        assert!(rect.contains_point(10.0, -1.0));
        assert!(rect.contains_point(10.0, -15.0));
        assert!(!rect.contains_point(10.0, -16.0));
        assert!(!rect.contains_point(11.0, 0.0));
        assert!(!rect.contains_point(9.0, 0.0));
    }

    #[test]
    fn test_rotated_rect_bounding() {
        #[track_caller]
        fn bounding<I: IntoIterator<Item = [f32; 2]>>(radians: f32, points: I) -> RotatedRect
        where
            I::IntoIter: Clone,
        {
            let points = points.into_iter();
            let rect = RotatedRect::bounding(radians, points.clone()).unwrap();

            let dilated = rect.map(|rect| rect.grow_rel(0.005));
            for [x, y] in points {
                assert!(
                    dilated.contains_point(x, y),
                    "{dilated:?} does not contain {x},{y}"
                );
            }

            rect
        }

        assert!(RotatedRect::bounding(0.0, []).is_none());

        assert_eq!(
            bounding(0.0, [[0.0, 0.0], [1.0, 1.0]]),
            Rect::from_top_left(0.0, 0.0, 1.0, 1.0).into(),
        );
        assert_eq!(
            bounding(0.0, [[0.0, 0.0], [10.0, 0.0]]),
            Rect::from_top_left(0.0, 0.0, 10.0, 0.0).into(),
        );
        assert_relative_eq!(
            bounding(TAU / 2.0, [[0.0, 0.0], [1.0, 1.0]]),
            RotatedRect::new(Rect::from_top_left(0.0, 0.0, 1.0, 1.0), TAU / 2.0),
            epsilon = 1e-7,
        );
        assert_relative_eq!(
            bounding(TAU / 4.0, [[0.0, 0.0], [1.0, 1.0]]),
            RotatedRect::new(Rect::from_top_left(0.0, 0.0, 1.0, 1.0), TAU / 4.0),
            epsilon = 1e-7,
        );
        assert_eq!(
            bounding(TAU / 4.0, [[0.0, 0.0], [9.0, 9.0]]),
            RotatedRect::new(Rect::from_top_left(0.0, 0.0, 9.0, 9.0), TAU / 4.0),
        );
    }
}
