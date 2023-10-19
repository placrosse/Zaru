//! Rectangle types.
//!
//! These are used throughout the library for image manipulation, object detection, regions of
//! interest, etc.

use std::{fmt, ops::RangeInclusive};

use crate::AspectRatio;
use zaru_linalg::{approx::ApproxEq, vec2, Mat2, Mat2f, Vec2f};

/// An axis-aligned rectangle.
///
/// Rectangles are allowed to have zero height and/or width. Negative dimensions are not allowed.
#[derive(Clone, Copy, PartialEq)]
pub struct Rect {
    center: Vec2f,
    size: Vec2f,
}

impl Rect {
    /// Creates a rectangle extending outwards from a center point.
    #[inline]
    pub fn from_center(x_center: f32, y_center: f32, width: f32, height: f32) -> Self {
        Self {
            center: vec2(x_center, y_center),
            size: vec2(width, height),
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
            center: self.center,
            size: self.size * scale,
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
            size: vec2(self.size.w + left + right, self.size.h + top + bottom),
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
            res.size.w += inc_w;
        } else {
            let target_height = self.width() / target_aspect.as_f32();
            let inc_h = target_height - self.height();
            res.size.h += inc_h;
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
    pub fn top_left(&self) -> Vec2f {
        self.center - self.size * 0.5
    }

    /// Returns the X coordinate of the left side of the rectangle.
    #[inline]
    pub fn x(&self) -> f32 {
        self.top_left().x
    }

    /// Returns the Y coordinate of the top side of the rectangle.
    #[inline]
    pub fn y(&self) -> f32 {
        self.top_left().y
    }

    #[inline]
    pub fn width(&self) -> f32 {
        self.size.w
    }

    #[inline]
    pub fn height(&self) -> f32 {
        self.size.h
    }

    /// Returns the number of pixels contained in `self`.
    #[inline]
    pub fn area(&self) -> f32 {
        self.size.w * self.size.h
    }

    #[inline]
    pub fn center(&self) -> Vec2f {
        self.center
    }

    #[inline]
    pub fn size(&self) -> Vec2f {
        self.size
    }

    #[must_use]
    pub fn move_by(&self, offset: impl Into<Vec2f>) -> Rect {
        Rect {
            center: self.center + offset.into(),
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

    pub fn contains_point(&self, point: impl Into<Vec2f>) -> bool {
        let p: Vec2f = point.into();
        self.x() <= p.x
            && self.y() <= p.y
            && self.x() + self.width() >= p.x
            && self.y() + self.height() >= p.y
    }

    pub fn corners(&self) -> [Vec2f; 4] {
        let [x, y] = [self.x(), self.y()];
        let [w, h] = [self.width(), self.height()];
        [
            vec2(x, y),
            vec2(x + w, y),
            vec2(x + w, y + h),
            vec2(x, y + h),
        ]
    }
}

impl fmt::Debug for Rect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Rect @ ({},{})/{}x{}",
            self.center().x,
            self.center().y,
            self.size().w,
            self.size().h
        )
    }
}

impl ApproxEq for Rect {
    type Tolerance = f32;

    fn abs_diff_eq(&self, other: &Self, abs_tolerance: Self::Tolerance) -> bool {
        self.center.abs_diff_eq(&other.center, abs_tolerance)
            && self.size.abs_diff_eq(&other.size, abs_tolerance)
    }

    fn rel_diff_eq(&self, other: &Self, rel_tolerance: Self::Tolerance) -> bool {
        self.center.rel_diff_eq(&other.center, rel_tolerance)
            && self.size.rel_diff_eq(&other.size, rel_tolerance)
    }

    fn ulps_diff_eq(&self, other: &Self, ulps_tolerance: u32) -> bool {
        self.center.ulps_diff_eq(&other.center, ulps_tolerance)
            && self.size.ulps_diff_eq(&other.size, ulps_tolerance)
    }
}

/// A [`Rect`], rotated around its center.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotatedRect {
    rect: Rect,
    radians: f32,
}

impl RotatedRect {
    /// Creates a new rotated rectangle.
    ///
    /// `radians` is the clockwise rotation to apply to the [`Rect`].
    #[inline]
    pub fn new(rect: Rect, radians: f32) -> Self {
        Self { rect, radians }
    }

    /// Approximates the rotated bounding rectangle that encompasses `points`.
    ///
    /// Returns [`None`] if `points` is an empty iterator.
    pub fn bounding<T: Into<Vec2f>, I: IntoIterator<Item = T>>(
        radians: f32,
        points: I,
    ) -> Option<Self> {
        let mut points = points.into_iter().peekable();

        // Make sure we have at least 1 point.
        points.peek()?;

        // Approach: we rotate all points so that we can compute width, height, and center point of
        // the rectangle, then we rotate the center back into the right coordinate system.
        // Note that, since we rotate the center back into the original coordinate system before
        // using it, it doesn't matter what point we rotate everything around. We pick the origin
        // for convenience. Picking a point closer to the centroid of the points could potentially
        // reduce rounding errors, but for now, this works fine.

        let cw = Mat2f::rotation_clockwise(radians);
        let mut min = Vec2f::splat(f32::MAX);
        let mut max = Vec2f::splat(f32::MIN);
        for point in points {
            let p: Vec2f = point.into();
            let p = cw * p;
            min = min.min(p);
            max = max.max(p);
        }

        // Center in rotated frame.
        let center = (min + max) * 0.5;

        // Center in non-rotated frame (original coordinates).
        let center = center.rotate_counterclockwise(radians);

        let size = max - min;

        Some(Self::new(
            Rect::from_center(center.x, center.y, size.w, size.h),
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

    pub fn center(&self) -> Vec2f {
        self.rect.center()
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

    /// Returns the rotated rectangle's corners in the parent's coordinate system.
    ///
    /// The order is: top-left, top-right, bottom-right, bottom-left, as seen from the non-rotated
    /// rect: after the rotation is applied, the corners can be rotated anywhere else, but the order
    /// is retained.
    pub fn rotated_corners(&self) -> [Vec2f; 4] {
        let corners = self.rect.corners();

        let rot = Mat2::rotation_counterclockwise(self.radians);
        corners.map(|p| {
            let rel = p - self.rect.center();
            let rot = rot * rel;
            self.rect().center() + rot
        })
    }

    pub fn contains_point(&self, point: impl Into<Vec2f>) -> bool {
        let pt = self.transform_in(point.into());

        // The rect offset was already compensated for by the transform.
        self.rect.move_to(0.0, 0.0).contains_point(pt)
    }

    /// Transforms a point from the parent coordinate system into the [`RotatedRect`]'s system.
    ///
    /// The origin of the inner coordinate system is formed by the top left corner of the rectangle.
    pub fn transform_in(&self, pt: impl Into<Vec2f>) -> Vec2f {
        self.transform_in_impl(pt.into())
    }
    fn transform_in_impl(&self, pt: Vec2f) -> Vec2f {
        let center = self.rect.size() * 0.5;
        let pos = pt - self.rect.top_left() - center;
        pos.rotate_clockwise(self.radians) + center
    }

    /// Transforms a point from the [`RotatedRect`]'s coordinate system to the parent system.
    ///
    /// The origin of the inner coordinate system is formed by the top left corner of the rectangle.
    pub fn transform_out(&self, pt: impl Into<Vec2f>) -> Vec2f {
        self.transform_out_impl(pt.into())
    }
    fn transform_out_impl(&self, pt: Vec2f) -> Vec2f {
        let center: Vec2f = self.rect.size() * 0.5;
        (pt - center).rotate_counterclockwise(self.radians) + center + self.rect.top_left()
    }
}

impl ApproxEq for RotatedRect {
    type Tolerance = f32;

    fn abs_diff_eq(&self, other: &Self, abs_tolerance: Self::Tolerance) -> bool {
        self.rect.abs_diff_eq(&other.rect, abs_tolerance)
            && self.radians.abs_diff_eq(&other.radians, abs_tolerance)
    }

    fn rel_diff_eq(&self, other: &Self, rel_tolerance: Self::Tolerance) -> bool {
        self.rect.rel_diff_eq(&other.rect, rel_tolerance)
            && self.radians.rel_diff_eq(&other.radians, rel_tolerance)
    }

    fn ulps_diff_eq(&self, other: &Self, ulps_tolerance: u32) -> bool {
        self.rect.ulps_diff_eq(&other.rect, ulps_tolerance)
            && self.radians.ulps_diff_eq(&other.radians, ulps_tolerance)
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

    use zaru_linalg::assert_approx_eq;

    use super::*;

    #[test]
    fn test_contains_point() {
        let rect = Rect::from_top_left(-5.0, 5.0, 10.0, 5.0);
        assert!(rect.contains_point([-5.0, 5.0]));
        assert!(rect.contains_point([-5.0 + 9.0, 5.0 + 4.0]));
        assert!(!rect.contains_point([-5.0 + 11.0, 5.0 + 4.0]));
        assert!(!rect.contains_point([-5.0 + 9.0, 5.0 + 5.0 + 1.0]));

        let empty = Rect::from_center(0.0, 0.0, 0.0, 0.0);
        assert!(!empty.contains_point([0.0025, 0.0]));
        assert!(!empty.contains_point([0.0, 1.0]));
        assert!(!empty.contains_point([0.0, -1.0]));
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
        assert_eq!(intersection.center(), smaller.center());
        assert_eq!(intersection.size(), smaller.size());

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
        assert_eq!(null.transform_in([0.0, 0.0]), vec2(0.0, 0.0));
        assert_eq!(null.transform_out([0.0, 0.0]), vec2(0.0, 0.0));

        assert_eq!(null.transform_in([1.0, -1.0]), vec2(1.0, -1.0));
        assert_eq!(null.transform_out([1.0, -1.0]), vec2(1.0, -1.0));

        let offset = RotatedRect::new(Rect::from_top_left(10.0, 20.0, 1.0, 1.0), 0.0);
        assert_eq!(offset.transform_in([0.0, 0.0]), vec2(-10.0, -20.0));
        assert_eq!(offset.transform_in([10.0, 20.0]), vec2(0.0, 0.0));

        // Rotated clockwise by 90°
        let right = RotatedRect::new(Rect::from_top_left(0.0, 0.0, 1.0, 1.0), TAU / 4.0);
        assert_eq!(right.transform_in([0.5, 0.5]), vec2(0.5, 0.5));
        assert_eq!(right.transform_out([0.5, 0.5]), vec2(0.5, 0.5));
        assert_approx_eq!(right.transform_in([0.0, 0.0]), vec2(0.0, 1.0));
        assert_approx_eq!(right.transform_out([0.0, 0.0]), vec2(1.0, 0.0));

        assert_approx_eq!(right.transform_in([1.0, 0.0]), vec2(0.0, 0.0));
        assert_approx_eq!(right.transform_out([0.0, -1.0]), vec2(2.0, 0.0));

        // Offset, rotated by 180°
        let rect = RotatedRect::new(Rect::from_top_left(10.0, 20.0, 1.0, 1.0), TAU / 2.0);
        assert_approx_eq!(rect.transform_in([10.0, 20.0]), vec2(1.0, 1.0));
        assert_approx_eq!(rect.transform_in([11.0, 21.0]), vec2(0.0, 0.0));
        assert_approx_eq!(rect.transform_out([0.0, 0.0]), vec2(11.0, 21.0));
    }

    #[test]
    fn test_rotated_rect_contains_point() {
        // 1x1 rect at origin
        let rect = RotatedRect::new(Rect::from_top_left(0.0, 0.0, 1.0, 1.0), 1.0);
        assert!(rect.contains_point([0.5, 0.5]));
        assert!(!rect.contains_point([0.0, 1.5]));
        assert!(!rect.contains_point([1.0, 1.0]));
        assert!(!rect.contains_point([1.0, 0.0]));
        assert!(!rect.contains_point([0.0, -1.0]));

        // 1x1 rect offset
        let rect = RotatedRect::new(Rect::from_top_left(10.0, 20.0, 1.0, 1.0), 1.0);
        assert!(rect.contains_point([10.5, 20.5]));
        assert!(!rect.contains_point([9.0, 20.0]));
        assert!(!rect.contains_point([10.0, 21.5]));

        // Wide rect, flipped
        let rect = RotatedRect::new(Rect::from_top_left(10.0, 20.0, 100.0, 1.0), TAU / 2.0);
        assert!(!rect.contains_point([-20.0, 20.5]));
        assert!(!rect.contains_point([9.0, 20.5]));
        assert!(rect.contains_point([10.0, 20.5]));
        assert!(rect.contains_point([100.0, 20.00005]));
        assert!(rect.contains_point([55.0, 20.5]));
        assert!(!rect.contains_point([55.0, 21.0]));
        assert!(!rect.contains_point([55.0, 19.0]));

        // Wide rect, rotated 90°
        let rect = RotatedRect::new(Rect::from_center(0.0, 0.0, 51.0, 1.0), TAU / 4.0);
        assert!(rect.contains_point([0.0, 0.0]));
        assert!(rect.contains_point([0.0, 1.0]));
        assert!(rect.contains_point([0.0, 25.0]));
        assert!(!rect.contains_point([0.0, 26.0]));
        assert!(rect.contains_point([0.0, -1.0]));
        assert!(rect.contains_point([0.0, -25.0]));
        assert!(!rect.contains_point([0.0, -26.0]));
        assert!(!rect.contains_point([1.0, 0.0]));
        assert!(!rect.contains_point([-1.0, 0.0]));

        // Wide rect, offset, rotated 90°
        let rect = RotatedRect::new(Rect::from_center(10.0, 10.0, 51.0, 1.0), TAU / 4.0);
        assert!(rect.contains_point([10.0, 0.0]));
        assert!(rect.contains_point([10.0, 1.0]));
        assert!(rect.contains_point([10.0, 35.0]));
        assert!(!rect.contains_point([10.0, 36.0]));
        assert!(rect.contains_point([10.0, -1.0]));
        assert!(rect.contains_point([10.0, -15.0]));
        assert!(!rect.contains_point([10.0, -16.0]));
        assert!(!rect.contains_point([11.0, 0.0]));
        assert!(!rect.contains_point([9.0, 0.0]));
    }

    #[test]
    fn test_rotated_rect_bounding() {
        #[track_caller]
        fn bounding<I: IntoIterator<Item = impl Into<Vec2f>>>(
            radians: f32,
            points: I,
        ) -> RotatedRect
        where
            I::IntoIter: Clone,
        {
            let points = points.into_iter();
            let rect = RotatedRect::bounding(radians, points.clone()).unwrap();

            let dilated = rect.map(|rect| rect.grow_rel(0.005));
            for point in points {
                let point = point.into();
                assert!(
                    dilated.contains_point(point),
                    "{dilated:?} does not contain {point}"
                );
            }

            rect
        }

        assert!(RotatedRect::bounding::<Vec2f, _>(0.0, []).is_none());

        assert_eq!(
            bounding(0.0, [[0.0, 0.0], [1.0, 1.0]]),
            Rect::from_top_left(0.0, 0.0, 1.0, 1.0).into(),
        );
        assert_eq!(
            bounding(0.0, [[0.0, 0.0], [10.0, 0.0]]),
            Rect::from_top_left(0.0, 0.0, 10.0, 0.0).into(),
        );
        assert_approx_eq!(
            bounding(TAU / 2.0, [[0.0, 0.0], [1.0, 1.0]]),
            RotatedRect::new(Rect::from_top_left(0.0, 0.0, 1.0, 1.0), TAU / 2.0),
        );
        assert_approx_eq!(
            bounding(TAU / 4.0, [[0.0, 0.0], [1.0, 1.0]]),
            RotatedRect::new(Rect::from_top_left(0.0, 0.0, 1.0, 1.0), TAU / 4.0),
        );
        assert_eq!(
            bounding(TAU / 4.0, [[0.0, 0.0], [9.0, 9.0]]),
            RotatedRect::new(Rect::from_top_left(0.0, 0.0, 9.0, 9.0), TAU / 4.0),
        );
    }

    #[test]
    fn corners() {
        let rect = Rect::from_center(1.0, 1.0, 4.0, 2.0);
        assert_eq!(
            rect.corners(),
            [[-1.0, 0.0], [3.0, 0.0], [3.0, 2.0], [-1.0, 2.0]]
        );
    }
}
