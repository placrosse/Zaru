//! Common functionality for object detection.
//!
//! The functionality defined in this module (and submodules) is meant to be reusable across
//! different detectors.

pub mod nms;
pub mod ssd;

use std::{fmt::Debug, marker::PhantomData};

use zaru_image::{
    draw, AsImageView, AsImageViewMut, Color, ImageView, ImageViewMut, Rect, Resolution,
    RotatedRect,
};
use zaru_nn::{Cnn, Outputs};
use zaru_utils::timer::Timer;

use self::nms::NonMaxSuppression;

/// Trait implemented by neural networks that detect objects in an input image.
pub trait Network: Send + Sync + 'static {
    /// The type used to represent the object classes this network can distinguish.
    ///
    /// Networks that only handle a single object class can set this to `()`.
    type Classes: Classes;

    /// Returns the [`Cnn`] to use for detection.
    fn cnn(&self) -> &Cnn;

    /// Extracts all detections with confidence above `threshold` from the network's output.
    ///
    /// Keypoint and detection positions are expected to be in the coordinate system of the
    /// network's input.
    fn extract(
        &self,
        outputs: &Outputs,
        threshold: f32,
        detections: &mut Detections<Self::Classes>,
    );
}

/// A collection of per-class object detections.
#[derive(Debug)]
pub struct Detections<C: Classes = ()> {
    // FIXME: make this sparse, networks can have thousands of classes
    vec: Vec<Vec<Detection>>,
    _p: PhantomData<C>,
}

impl<C: Classes> Detections<C> {
    pub fn new() -> Self {
        Self {
            vec: Vec::new(),
            _p: PhantomData,
        }
    }

    /// Returns the total number of detections across all object classes.
    pub fn len(&self) -> usize {
        self.vec.iter().map(|v| v.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.vec.iter().all(|v| v.is_empty())
    }

    pub fn clear(&mut self) {
        for class in &mut self.vec {
            class.clear();
        }
    }

    pub fn push(&mut self, class: C, detection: Detection) {
        let raw_class = class.as_u32() as usize;
        if self.vec.len() <= raw_class {
            self.vec.resize_with(raw_class + 1, Vec::new);
        }

        self.vec[raw_class].push(detection);
    }

    /// Returns an iterator yielding all detections alongside their class.
    pub fn all_detections(&self) -> impl Iterator<Item = (C, &Detection)> {
        self.vec
            .iter()
            .enumerate()
            .flat_map(|(i, v)| v.iter().map(move |det| (C::from_u32(i as u32), det)))
    }

    pub fn all_detections_mut(&mut self) -> impl Iterator<Item = (C, &mut Detection)> {
        self.vec
            .iter_mut()
            .enumerate()
            .flat_map(|(i, v)| v.iter_mut().map(move |det| (C::from_u32(i as u32), det)))
    }

    /// Returns an iterator that yields all detections of the given class.
    pub fn for_class(&self, class: C) -> impl Iterator<Item = &Detection> {
        self.vec
            .get(class.as_u32() as usize)
            .into_iter()
            .flat_map(|v| v.iter())
    }
}

impl Detections {
    /// Returns an iterator yielding the stored detections.
    pub fn iter(&self) -> impl Iterator<Item = &Detection> {
        self.vec.iter().flat_map(|v| v.iter())
    }
}

/// Types that represent object classes.
///
/// Some object detectors are able to detect and distinguish between a number of different types of
/// objects. A type implementing this trait can be used to represent the different object classes.
///
/// For networks that only detect objects of a single type, `()` can be used as the [`Classes`]
/// implementor.
pub trait Classes: Send + Sync + 'static {
    /// Casts an instance of `self` to a raw `u32`.
    fn as_u32(&self) -> u32;

    /// Casts a raw `u32` to an instance of `Self`.
    ///
    /// The library never passes invalid values for `raw` to this method, so any (safe) behavior is
    /// permitted in that case (eg. panicking or returning a default value).
    fn from_u32(raw: u32) -> Self;
}

impl Classes for () {
    #[inline]
    fn as_u32(&self) -> u32 {
        0
    }

    #[inline]
    fn from_u32(_: u32) -> Self {
        ()
    }
}

/// A generic object detector.
///
/// This type wraps a [`Network`] for object detection.
pub struct Detector<C: Classes> {
    network: Box<dyn Network<Classes = C>>,
    detections: Detections<C>,
    t_infer: Timer,
    t_extract: Timer,
    t_nms: Timer,
    thresh: f32,
    nms: NonMaxSuppression,
    _p: PhantomData<C>,
}

impl<C: Classes> Detector<C> {
    pub const DEFAULT_THRESHOLD: f32 = 0.5;

    pub fn new<N: Network<Classes = C>>(network: N) -> Self {
        Self {
            network: Box::new(network),
            detections: Detections::new(),
            t_infer: Timer::new("infer"),
            t_extract: Timer::new("extract"),
            t_nms: Timer::new("nms"),
            thresh: Self::DEFAULT_THRESHOLD,
            nms: NonMaxSuppression::new(),
            _p: PhantomData,
        }
    }

    pub fn input_resolution(&self) -> Resolution {
        self.network.cnn().input_resolution()
    }

    #[inline]
    pub fn set_threshold(&mut self, thresh: f32) {
        self.thresh = thresh;
    }

    pub fn nms_mut(&mut self) -> &mut NonMaxSuppression {
        &mut self.nms
    }

    pub fn detect<V: AsImageView>(&mut self, image: &V) -> &Detections<C> {
        self.detect_impl(image.as_view())
    }

    fn detect_impl(&mut self, image: ImageView<'_>) -> &Detections<C> {
        self.detections.clear();

        let cnn = self.network.cnn();
        let input_res = cnn.input_resolution();

        // If the input image's aspect ratio doesn't match the CNN's input, create an oversized view
        // that does.
        let rect = image
            .rect()
            .grow_to_fit_aspect(input_res.aspect_ratio().unwrap());
        let view = image.view(rect);
        let outputs = self.t_infer.time(|| cnn.estimate(&view)).unwrap();
        log::trace!("inference result: {:?}", outputs);

        self.t_extract.time(|| {
            self.network
                .extract(&outputs, self.thresh, &mut self.detections)
        });

        self.t_nms.time(|| {
            for class in &mut self.detections.vec {
                // FIXME: do this in-place
                let iter = self.nms.process(class);
                class.clear();
                class.extend(iter);
            }
        });

        // Map all coordinates back into the input image.
        let scale = rect.width() as f32 / input_res.width() as f32;
        for (_, det) in self.detections.all_detections_mut() {
            // Map all coordinates from the network's input coordinate system to `rect`'s system.
            det.rect.xc *= scale;
            det.rect.yc *= scale;
            det.rect.w *= scale;
            det.rect.h *= scale;
            for kp in &mut det.keypoints {
                kp.x *= scale;
                kp.y *= scale;
            }

            // Now remove the offset added by the oversized rectangle (this compensates for
            // "black bars" added to adjust the aspect ratio).
            det.rect.xc += rect.x() as f32;
            det.rect.yc += rect.y() as f32;
            for kp in &mut det.keypoints {
                kp.x += rect.x() as f32;
                kp.y += rect.y() as f32;
            }
        }

        &self.detections
    }

    pub fn timers(&self) -> impl Iterator<Item = &Timer> + '_ {
        [&self.t_infer, &self.t_extract, &self.t_nms].into_iter()
    }
}

/// A detected object.
///
/// A [`Detection`] consists of a [`BoundingRect`] enclosing the detected object, a confidence
/// value, an optional rotation angle of the object, and a possibly empty set of located keypoints.
///
/// Per convention, the confidence value lies between 0.0 and 1.0, which can be achieved by passing
/// the raw network output through [`crate::num::sigmoid`] (but the network documentation should be
/// consulted). The confidence value is used when performing non-maximum suppression with
/// [`nms::SuppressionMode::Average`], so it has to have the expected range when making use of that.
#[derive(Debug, Clone)]
pub struct Detection {
    confidence: f32,
    angle: f32,
    rect: BoundingRect,
    keypoints: Vec<Keypoint>,
}

impl Detection {
    pub fn new(confidence: f32, rect: BoundingRect) -> Self {
        Self {
            confidence,
            angle: 0.0,
            rect,
            keypoints: Vec::new(),
        }
    }

    pub fn with_keypoints(confidence: f32, rect: BoundingRect, keypoints: Vec<Keypoint>) -> Self {
        Self {
            confidence,
            angle: 0.0,
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

    /// Returns the angle of the detected object, in radians, clockwise.
    ///
    /// Note that not all networks support computing the object angle. If it is not supported, an
    /// angle of 0.0 will be returned.
    pub fn angle(&self) -> f32 {
        self.angle
    }

    /// Sets the angle of the detected object, in radians, clockwise.
    pub fn set_angle(&mut self, angle: f32) {
        self.angle = angle;
    }

    /// Returns the axis-aligned bounding rectangle containing the detected object.
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

    pub fn draw<I: AsImageViewMut>(&self, image: &mut I) {
        self.draw_impl(&mut image.as_view_mut());
    }

    fn draw_impl(&self, image: &mut ImageViewMut<'_>) {
        draw::rotated_rect(
            image,
            RotatedRect::new(self.bounding_rect().to_rect(), self.angle()),
        )
        .color(Color::from_rgb8(170, 0, 0));
        for lm in self.keypoints() {
            draw::marker(image, lm.x() as _, lm.y() as _);
        }

        let color = match self.confidence() {
            0.8.. => Color::GREEN,
            0.4..=0.8 => Color::YELLOW,
            _ => Color::RED,
        };
        draw::text(
            image,
            self.bounding_rect().xc as _,
            (self.bounding_rect().yc + self.bounding_rect().h * 0.5) as _,
            &format!("conf={:.01}", self.confidence()),
        )
        .align_top()
        .color(color);
    }
}

/// A 2D keypoint produced as part of a [`Detection`].
///
/// Keypoints are often, but not always, inside the detection bounding box and indicate the
/// approximate location of some object landmark.
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

    pub fn to_rect(&self) -> Rect {
        Rect::from_center(
            self.xc as _,
            self.yc as _,
            self.w.ceil() as _,
            self.h.ceil() as _,
        )
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