//! Common functionality for object detection.
//!
//! The functionality defined in this module (and submodules) is meant to be reusable across
//! different detectors.

pub mod nms;
pub mod ssd;

use crate::{image::Rect, nn::point_to_img, resolution::Resolution};

/// A detected object.
///
/// A [`RawDetection`] consists of a [`BoundingRect`] enclosing the detected object, a confidence
/// value, and an optional set of located keypoints.
///
/// Per convention, the confidence value lies between 0.0 and 1.0, which can be achieved by passing
/// the raw network output through [`crate::num::sigmoid`] (but the network documentation should be
/// consulted). The confidence value is used when performing non-maximum suppression with
/// [`nms::SuppressionMode::Average`], so it has to have the expected range when making use of that.
///
/// Called "raw" because it does not reside in any defined coordinate system. Detector
/// implementations typically provide a wrapper around this type that allows accessing the detection
/// as a [`Rect`] in input image coordinates.
#[derive(Debug, Clone)]
pub struct RawDetection {
    confidence: f32,
    rect: BoundingRect,
    keypoints: Vec<Keypoint>,
}

impl RawDetection {
    pub fn new(confidence: f32, rect: BoundingRect) -> Self {
        Self {
            confidence,
            rect,
            keypoints: Vec::new(),
        }
    }

    pub fn with_keypoints(confidence: f32, rect: BoundingRect, keypoints: Vec<Keypoint>) -> Self {
        Self {
            confidence,
            rect,
            keypoints,
        }
    }

    pub fn push_keypoint(&mut self, keypoint: Keypoint) {
        self.keypoints.push(keypoint);
    }

    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    pub fn set_confidence(&mut self, confidence: f32) {
        self.confidence = confidence;
    }

    pub fn bounding_rect(&self) -> BoundingRect {
        self.rect
    }

    pub fn set_bounding_rect(&mut self, rect: BoundingRect) {
        self.rect = rect;
    }

    pub fn keypoints(&self) -> &[Keypoint] {
        &self.keypoints
    }

    pub fn keypoints_mut(&mut self) -> &mut Vec<Keypoint> {
        &mut self.keypoints
    }
}

/// A 2D keypoint produced as part of a [`RawDetection`].
///
/// A keypoint by itself is just a point in an unspecified coordinate system. Keypoints are often,
/// but not always, inside the detection bounding box.
///
/// The meaning of a keypoint depends on the specific detector and on its index in the keypoint
/// list. Typically keypoints are used to crop/rotate a detected object for further processing.
///
/// Not all detectors output keypoints. Some may just output bounding rectangles with confidence
/// scores.
#[derive(Debug, Clone, Copy)]
pub struct Keypoint {
    x: f32,
    y: f32,
}

impl Keypoint {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn x(&self) -> f32 {
        self.x
    }

    pub fn y(&self) -> f32 {
        self.y
    }
}

/// Axis-aligned bounding rectangle of a detected object.
///
/// This primarily differs from [`Rect`] in that it uses float coordinates instead of integers.
#[derive(Debug, Clone, Copy)]
pub struct BoundingRect {
    xc: f32,
    yc: f32,
    w: f32,
    h: f32,
}

impl BoundingRect {
    /// Creates a bounding rectangle centered at `(xc,yc)`.
    pub fn from_center(xc: f32, yc: f32, w: f32, h: f32) -> Self {
        Self { xc, yc, w, h }
    }

    pub(crate) fn grow_rel(&self, left: f32, right: f32, top: f32, bottom: f32) -> Self {
        let left = left * self.w;
        let right = right * self.w;
        let top = top * self.h;
        let bottom = bottom * self.h;
        Self {
            xc: self.xc + (right - left) * 0.5,
            yc: self.yc + (bottom - top) * 0.5,
            w: self.w + left + right,
            h: self.h + top + bottom,
        }
    }

    /// Uniformly scales the size of `self` by `scale`.
    #[cfg(test)]
    fn scale(&self, scale: f32) -> Self {
        Self {
            xc: self.xc,
            yc: self.yc,
            w: self.w * scale,
            h: self.h * scale,
        }
    }

    pub(crate) fn to_rect(&self, full_res: &Resolution) -> Rect {
        let top_left = (self.xc - self.w / 2.0, self.yc - self.h / 2.0);
        let bottom_right = (self.xc + self.w / 2.0, self.yc + self.h / 2.0);

        let top_left = point_to_img(top_left.0, top_left.1, full_res);
        let bottom_right = point_to_img(bottom_right.0, bottom_right.1, full_res);

        Rect::from_corners(top_left, bottom_right)
    }

    fn top_left(&self) -> (f32, f32) {
        (self.xc - self.w / 2.0, self.yc - self.h / 2.0)
    }

    fn bottom_right(&self) -> (f32, f32) {
        (self.xc + self.w / 2.0, self.yc + self.h / 2.0)
    }

    /// Returns the amount of area covered by `self`.
    pub fn area(&self) -> f32 {
        self.w * self.h
    }

    fn intersection(&self, other: &Self) -> Self {
        let top_left_1 = self.top_left();
        let top_left_2 = other.top_left();
        let top_left = (
            top_left_1.0.max(top_left_2.0),
            top_left_1.1.max(top_left_2.1),
        );

        let bot_right_1 = self.bottom_right();
        let bot_right_2 = other.bottom_right();
        let bot_right = (
            bot_right_1.0.min(bot_right_2.0),
            bot_right_1.1.min(bot_right_2.1),
        );

        Self {
            xc: (top_left.0 + bot_right.0) / 2.0,
            yc: (top_left.1 + bot_right.1) / 2.0,
            w: bot_right.0 - top_left.0,
            h: bot_right.1 - top_left.1,
        }
    }

    fn intersection_area(&self, other: &Self) -> f32 {
        self.intersection(other).area()
    }

    fn union_area(&self, other: &Self) -> f32 {
        self.area() + other.area() - self.intersection_area(other)
    }

    /// Computes the Intersection over Union (IOU) of `self` and `other`.
    pub fn iou(&self, other: &Self) -> f32 {
        self.intersection_area(other) / self.union_area(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounding_rect_zero() {
        let zero = BoundingRect {
            xc: 0.0,
            yc: 0.0,
            w: 0.0,
            h: 0.0,
        };
        assert_eq!(zero.area(), 0.0);

        let also_zero = BoundingRect {
            xc: 1.0,
            yc: 0.0,
            w: 0.0,
            h: 0.0,
        };
        assert_eq!(also_zero.area(), 0.0);

        assert_eq!(zero.intersection(&also_zero).area(), 0.0);
        assert_eq!(zero.union_area(&also_zero), 0.0);
    }

    #[test]
    fn test_intersection() {
        let a = BoundingRect {
            xc: 1.0,
            yc: 0.0,
            w: 1.0,
            h: 1.0,
        };
        let b = BoundingRect {
            xc: 2.0,
            yc: 0.0,
            w: 1.0,
            h: 1.0,
        };
        assert_eq!(a.intersection(&b).area(), 0.0);
        assert_eq!(b.intersection(&a).area(), 0.0);

        let c = BoundingRect {
            xc: 1.5,
            yc: 0.0,
            w: 1.0,
            h: 1.0,
        };
        let ac = a.intersection(&c);
        assert_eq!(ac.xc, 1.25);
        assert_eq!(ac.yc, 0.0);
        assert_eq!(ac.w, 0.5);
        assert_eq!(ac.h, 1.0);
    }

    #[test]
    fn test_bounding_rect() {
        // Two rects with the same center point, but different sizes.
        let smaller = BoundingRect {
            xc: 9.0,
            yc: 9.0,
            w: 1.0,
            h: 1.0,
        };
        let bigger = BoundingRect {
            xc: 9.0,
            yc: 9.0,
            w: 2.0,
            h: 2.0,
        };

        assert_eq!(smaller.area(), 1.0);
        assert_eq!(bigger.area(), 4.0);

        let intersection = smaller.intersection(&bigger);
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
}
