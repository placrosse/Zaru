//! Face detection module.
//!
//! This uses one of the "BlazeFace" neural networks also used in MediaPipe's [Face Detection]
//! module.
//!
//! [Face Detection]: https://google.github.io/mediapipe/solutions/face_detection

use nalgebra::{Rotation2, Vector2};

use crate::{
    filter::{AlphaBetaFilter, Ema, Filter},
    image::{self, AsImageView, AsImageViewMut, Color, ImageView, ImageViewMut, Rect},
    nn::{point_to_img, Cnn, CnnInputFormat, NeuralNetwork},
    num::TotalF32,
    resolution::Resolution,
    timer::Timer,
};

use self::{
    avg::DetectionFilter,
    nma::NonMaxAvg,
    ssd::{Anchor, AnchorParams, Anchors, LayerInfo},
};

mod avg;
mod nma;
mod ssd;

/// Detection confidence threshold.
///
/// Minimum confidence at which a detection is used as the "seed" of a non-maximum suppression round
/// (or in this case, non-maximum averaging).
///
/// Tested thresholds for short-range model:
/// - 0.5 creates false positives when blocking the camera with my hand.
const SEED_THRESH: f32 = 0.6;
/// Thresholds for detections to *contribute* to an NMS/NMA round started by another detection with
/// at least `SEED_THRESH`.
const CONTRIB_THRESH: f32 = 0.0;
/// Minimum Intersection-over-Union value to consider two detections to overlap.
const IOU_THRESH: f32 = 0.3;

/// If this is `true`, an [`AlphaBetaFilter`] is used to smooth detections across frames. If it is
/// `false` an exponentially-weighted moving average is used instead ([`Ema`]).
///
/// Typically, the exponential moving average performs well enough for face detection.
const USE_AB_FILTER: bool = false;

/// Alpha parameter of the exponentially-weighted moving average filter.
const EMA_ALPHA: f32 = 0.25;

/// Alpha parameter of the alpha beta filter.
const AB_FILTER_ALPHA: f32 = 0.2;
/// Beta parameter of the alpha beta filter.
const AB_FILTER_BETA: f32 = 0.03;

const MODEL_PATH: &str = "models/face_detection_short_range.onnx";

/// Neural-Network based face detector.
pub struct Detector {
    anchors: Anchors,
    model: Cnn,
    t_resize: Timer,
    t_infer: Timer,
    t_filter: Timer,
    nma: NonMaxAvg,
    avg: Box<dyn Filter<RawDetection> + Send>,
    raw_detections: Vec<RawDetection>,
    detections: Vec<Detection>,
}

impl Detector {
    /// Creates a new face detector.
    pub fn new() -> Self {
        let anchors = Anchors::calculate(&AnchorParams {
            layers: &[LayerInfo::new(2, 16, 16), LayerInfo::new(6, 8, 8)],
        });

        let avg: Box<dyn Filter<RawDetection> + Send> = if USE_AB_FILTER {
            Box::new(DetectionFilter::new(AlphaBetaFilter::new(
                AB_FILTER_ALPHA,
                AB_FILTER_BETA,
            )))
        } else {
            Box::new(DetectionFilter::new(Ema::new(EMA_ALPHA)))
        };

        Self {
            anchors,
            // FIXME share model globally
            model: Cnn::new(
                NeuralNetwork::load(MODEL_PATH).unwrap(),
                CnnInputFormat::NHWC,
            )
            .unwrap(),
            t_resize: Timer::new("resize"),
            t_infer: Timer::new("infer"),
            t_filter: Timer::new("filter"),
            nma: NonMaxAvg::new(SEED_THRESH, IOU_THRESH),
            avg,
            raw_detections: Vec::new(),
            detections: Vec::new(),
        }
    }

    /// Runs face detections on an input image, returning the filtered detections.
    ///
    /// The image will be scaled to the input size expected by the neural network, and detections
    /// will be back-mapped to input image coordinates.
    pub fn detect<V: AsImageView>(&mut self, image: &V) -> &[Detection] {
        self.detect_impl(image.as_view())
    }

    fn detect_impl(&mut self, image: ImageView<'_>) -> &[Detection] {
        self.raw_detections.clear();
        self.detections.clear();

        let full_res = image.resolution();

        let mut image = image.reborrow();
        let resized;
        if image.resolution() != self.model.input_resolution() {
            resized = self
                .t_resize
                .time(|| image.aspect_aware_resize(self.model.input_resolution()));
            image = resized.as_view();
        }
        let result = self.t_infer.time(|| self.model.infer(&image)).unwrap();
        log::trace!("inference result: {:?}", result);

        self.t_filter.time(|| {
            let boxes = &result[0];

            let conf = result[1].as_slice::<f32>().unwrap();
            for (index, &conf) in conf.iter().enumerate() {
                if conf < CONTRIB_THRESH {
                    continue;
                }

                let tensor_view = boxes.view_at_prefix(&[0, index]).unwrap();
                let box_params = &tensor_view.as_slice::<f32>().unwrap()[..16];
                self.raw_detections.push(RawDetection::extract(
                    &self.anchors[index],
                    box_params,
                    conf,
                ));
            }

            let detections = self.nma.average(&mut self.raw_detections);
            // TODO >1
            match detections.iter().max_by_key(|det| TotalF32(det.confidence)) {
                Some(det) => {
                    let raw = self.avg.push(*det);
                    self.detections.push(Detection { raw, full_res });
                }
                None => self.avg.reset(),
            }
        });

        &self.detections
    }

    /// Returns profiling timers for image resizing, neural inference, and detection filtering.
    pub fn timers(&self) -> impl IntoIterator<Item = &Timer> + '_ {
        [&self.t_resize, &self.t_infer, &self.t_filter]
    }
}

/// A detected face, consisting of a bounding box and landmarks.
#[derive(Debug)]
pub struct Detection {
    raw: RawDetection,
    full_res: Resolution,
}

impl Detection {
    /// Returns the raw bounding box of the face as output by the network (adjusted for the input image).
    ///
    /// This box is *very* tight and does not include the head boundary or much of the forehead.
    pub fn bounding_rect_raw(&self) -> Rect {
        self.raw.bounding_box_raw().to_rect(&self.full_res)
    }

    /// Returns the bounding box of the detected face, adjusted to include the whole head boundary.
    pub fn bounding_rect_loose(&self) -> Rect {
        self.raw.bounding_box_loose().to_rect(&self.full_res)
    }

    /// Returns the confidence of this detection.
    pub fn confidence(&self) -> f32 {
        self.raw.confidence
    }

    /// Estimated clockwise rotation of the face.
    pub fn rotation_radians(&self) -> f32 {
        let left_eye = self.left_eye();
        let right_eye = self.right_eye();
        let left_to_right_eye = Vector2::new(
            (right_eye.0 - left_eye.0) as f32,
            (right_eye.1 - left_eye.1) as f32,
        );
        Rotation2::rotation_between(&Vector2::x(), &left_to_right_eye).angle()
    }

    /// Returns the coordinates of the left eye's landmark (from the perspective of the input image,
    /// not the depicted person).
    pub fn left_eye(&self) -> (i32, i32) {
        self.raw.landmarks[0].to_img_coord(&self.full_res)
    }

    /// Returns the coordinates of the right eye's landmark (from the perspective of the input image,
    /// not the depicted person).
    pub fn right_eye(&self) -> (i32, i32) {
        self.raw.landmarks[1].to_img_coord(&self.full_res)
    }

    /// Draws the bounding box and landmarks of this detection onto an image.
    ///
    /// # Panics
    ///
    /// The image must have the same resolution as the image the detection was performed on,
    /// otherwise this method will panic.
    pub fn draw<I: AsImageViewMut>(&self, image: &mut I) {
        self.draw_impl(&mut image.as_view_mut());
    }

    fn draw_impl(&self, image: &mut ImageViewMut<'_>) {
        let res = Resolution::new(image.width(), image.height());
        assert_eq!(
            res, self.full_res,
            "attempted to draw `Detection` onto canvas with mismatched size",
        );

        image::draw_rect(image, self.bounding_rect_raw());
        for mark in &self.raw.landmarks {
            let (x, y) = mark.to_img_coord(&self.full_res);
            image::draw_marker(image, x, y);
        }

        image::draw_rect(image, self.bounding_rect_loose()).color(Color::from_rgb8(0, 255, 0));
    }
}

#[derive(Debug, Clone, Copy)]
struct RawDetection {
    bounding_box: BoundingBox,
    landmarks: [Landmark; 6],
    confidence: f32,
}

impl RawDetection {
    fn extract(anchor: &Anchor, box_params: &[f32], confidence: f32) -> Self {
        assert_eq!(box_params.len(), 16);

        let xc = box_params[0] / 128.0 + anchor.x_center();
        let yc = box_params[1] / 128.0 + anchor.y_center();
        let w = box_params[2] / 128.0;
        let h = box_params[3] / 128.0;
        let pt = |x, y| Landmark {
            x: x / 128.0 + anchor.x_center(),
            y: y / 128.0 + anchor.y_center(),
        };

        RawDetection {
            bounding_box: BoundingBox { xc, yc, w, h },
            landmarks: [
                pt(box_params[4], box_params[5]),
                pt(box_params[6], box_params[7]),
                pt(box_params[8], box_params[9]),
                pt(box_params[10], box_params[11]),
                pt(box_params[12], box_params[13]),
                pt(box_params[14], box_params[15]),
            ],
            confidence,
        }
    }

    /// Returns the raw bounding box of the face as output by the network.
    ///
    /// This box is *very* tight and does not include the head boundary or much of the forehead.
    fn bounding_box_raw(&self) -> BoundingBox {
        self.bounding_box
    }

    fn bounding_box_loose(&self) -> BoundingBox {
        // apply "forehead adjustment"

        const LOOSEN_LEFT: f32 = 0.08;
        const LOOSEN_RIGHT: f32 = 0.08;
        const LOOSEN_TOP: f32 = 0.55;
        const LOOSEN_BOTTOM: f32 = 0.2;

        self.bounding_box
            .grow_rel(LOOSEN_LEFT, LOOSEN_RIGHT, LOOSEN_TOP, LOOSEN_BOTTOM)
    }
}

#[derive(Debug, Clone, Copy)]
struct Landmark {
    x: f32,
    y: f32,
}

impl Landmark {
    fn to_img_coord(&self, full_res: &Resolution) -> (i32, i32) {
        point_to_img(self.x, self.y, full_res)
    }
}

/// An axis-aligned rectangle in a `[0.0, 1.0]` coordinate space.
#[derive(Debug, Clone, Copy)]
struct BoundingBox {
    xc: f32,
    yc: f32,
    w: f32,
    h: f32,
}

impl BoundingBox {
    fn grow_rel(&self, left: f32, right: f32, top: f32, bottom: f32) -> Self {
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

    fn to_rect(&self, full_res: &Resolution) -> Rect {
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

    fn area(&self) -> f32 {
        self.w * self.h
    }

    fn intersection(&self, other: &BoundingBox) -> BoundingBox {
        let top_left_1 = self.top_left();
        let top_left_2 = other.top_left();
        let top_left = (
            top_left_1.0.max(top_left_2.0),
            top_left_1.1.max(top_left_2.1),
        );

        let bot_right_1 = self.bottom_right();
        let bot_right_2 = other.bottom_right();
        let bot_right = (
            bot_right_1.0.max(bot_right_2.0),
            bot_right_1.1.max(bot_right_2.1),
        );

        BoundingBox {
            xc: (top_left.0 + bot_right.0) / 2.0,
            yc: (top_left.1 + bot_right.1) / 2.0,
            w: bot_right.0 - top_left.0,
            h: bot_right.1 - top_left.1,
        }
    }

    fn intersection_area(&self, other: &BoundingBox) -> f32 {
        self.intersection(other).area()
    }

    fn union_area(&self, other: &BoundingBox) -> f32 {
        self.area() + other.area() - self.intersection_area(other)
    }

    fn iou(&self, other: &BoundingBox) -> f32 {
        self.intersection_area(other) / self.union_area(other)
    }
}
