//! Palm detection.

use once_cell::sync::Lazy;

use crate::{
    detection::{
        nms::NonMaxSuppression,
        ssd::{Anchor, AnchorParams, Anchors, LayerInfo},
        BoundingRect, RawDetection,
    },
    image::{self, AsImageView, AsImageViewMut, Color, ImageView, ImageViewMut, Rect},
    nn::{create_linear_color_mapper, point_to_img, Cnn, CnnInputShape, NeuralNetwork},
    num::sigmoid,
    resolution::Resolution,
    timer::Timer,
};

pub struct PalmDetector {
    cnn: &'static Cnn,
    anchors: Anchors,
    nms: NonMaxSuppression,
    thresh: f32,
    t_resize: Timer,
    t_infer: Timer,
    t_nms: Timer,
    raw_detections: Vec<RawDetection>,
    detections: Vec<Detection>,
}

impl PalmDetector {
    const DEFAULT_THRESH: f32 = 0.5;

    pub fn new<N: PalmDetectionNetwork>(network: N) -> Self {
        drop(network);
        Self {
            cnn: N::cnn(),
            anchors: Anchors::calculate(&AnchorParams {
                layers: &[LayerInfo::new(2, 24, 24), LayerInfo::new(6, 12, 12)],
            }),
            nms: NonMaxSuppression::new(),
            thresh: Self::DEFAULT_THRESH,
            t_resize: Timer::new("resize"),
            t_infer: Timer::new("infer"),
            t_nms: Timer::new("NMS"),
            raw_detections: Vec::new(),
            detections: Vec::new(),
        }
    }

    /// Returns the expected input resolution of the internal neural network.
    pub fn input_resolution(&self) -> Resolution {
        self.cnn.input_resolution()
    }

    pub fn detect<A: AsImageView>(&mut self, image: A) -> &[Detection] {
        self.detect_impl(image.as_view())
    }

    fn detect_impl(&mut self, image: ImageView<'_>) -> &[Detection] {
        self.raw_detections.clear();
        self.detections.clear();

        let full_res = image.resolution();
        let input_resolution = self.input_resolution();

        let mut image = image.reborrow();
        let resized;
        if image.resolution() != input_resolution {
            resized = self
                .t_resize
                .time(|| image.aspect_aware_resize(input_resolution));
            image = resized.as_view();
        }
        let result = self.t_infer.time(|| self.cnn.estimate(&image)).unwrap();
        log::trace!("inference result: {:?}", result);

        let num_anchors = self.anchors.anchor_count();
        self.t_nms.time(|| {
            let boxes = &result[0];
            let confidences = &result[1];

            assert_eq!(boxes.shape(), &[1, num_anchors, 18]);
            assert_eq!(confidences.shape(), &[1, num_anchors, 1]);

            for (index, view) in confidences.index([0]).iter().enumerate() {
                let conf = sigmoid(view.as_slice()[0]);

                if conf < self.thresh {
                    continue;
                }

                let tensor_view = boxes.index([0, index]);
                let box_params = tensor_view.as_slice();
                self.raw_detections.push(extract_detection(
                    &self.anchors[index],
                    input_resolution,
                    box_params,
                    conf,
                ));
            }

            let detections = self.nms.process(&mut self.raw_detections);
            for raw in detections {
                self.detections.push(Detection { raw, full_res });
            }
        });

        &self.detections
    }

    pub fn timers(&self) -> impl Iterator<Item = &Timer> + '_ {
        [&self.t_resize, &self.t_infer, &self.t_nms].into_iter()
    }
}

fn extract_detection(
    anchor: &Anchor,
    input_res: Resolution,
    box_params: &[f32],
    confidence: f32,
) -> RawDetection {
    assert_eq!(box_params.len(), 18);

    let input_w = input_res.width() as f32;
    let input_h = input_res.height() as f32;

    let xc = box_params[0] / input_w + anchor.x_center();
    let yc = box_params[1] / input_h + anchor.y_center();
    let w = box_params[2] / input_w;
    let h = box_params[3] / input_h;
    let lm = |x, y| {
        crate::detection::Keypoint::new(
            x / input_w + anchor.x_center(),
            y / input_h + anchor.y_center(),
        )
    };

    RawDetection::with_keypoints(
        confidence,
        BoundingRect::from_center(xc, yc, w, h),
        vec![
            lm(box_params[4], box_params[5]),
            lm(box_params[6], box_params[7]),
            lm(box_params[8], box_params[9]),
            lm(box_params[10], box_params[11]),
            lm(box_params[12], box_params[13]),
            lm(box_params[14], box_params[15]),
            lm(box_params[16], box_params[17]),
        ],
    )
}

pub struct Detection {
    raw: RawDetection,
    full_res: Resolution,
}

impl Detection {
    pub fn confidence(&self) -> f32 {
        self.raw.confidence()
    }

    pub fn bounding_rect(&self) -> Rect {
        self.raw.bounding_rect().to_rect(&self.full_res)
    }

    /// Draws the bounding box of this detection onto an image.
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

        image::draw_rect(image, self.bounding_rect()).color(Color::BLUE);
        for (i, lm) in self.raw.keypoints().iter().enumerate() {
            let (x, y) = point_to_img(lm.x(), lm.y(), &self.full_res);
            image::draw_marker(image, x, y).color(Color::BLUE);
            image::draw_text(image, x, y - 5, &i.to_string()).color(Color::BLUE);
        }
    }
}

pub trait PalmDetectionNetwork {
    fn cnn() -> &'static Cnn;
}

/// A "lightweight" palm detection network.
///
/// **Note**: This network is still *extremely* heavy compared to other "light" networks. CPU
/// inference seems to take around 10 times as long as for the short-range *face* detection network,
/// for example.
pub struct LiteNetwork;

impl PalmDetectionNetwork for LiteNetwork {
    fn cnn() -> &'static Cnn {
        const MODEL_DATA: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/3rdparty/onnx/palm_detection_lite.onnx"
        ));

        static MODEL: Lazy<Cnn> = Lazy::new(|| {
            Cnn::new(
                NeuralNetwork::from_onnx(MODEL_DATA)
                    .unwrap()
                    .load()
                    .unwrap(),
                CnnInputShape::NCHW,
                create_linear_color_mapper(0.0..=1.0),
            )
            .unwrap()
        });

        &MODEL
    }
}

/// The full-range palm detection network.
///
/// This is about 15% slower than [`LiteNetwork`].
pub struct FullNetwork;

impl PalmDetectionNetwork for FullNetwork {
    fn cnn() -> &'static Cnn {
        const MODEL_DATA: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/3rdparty/onnx/palm_detection_full.onnx"
        ));

        static MODEL: Lazy<Cnn> = Lazy::new(|| {
            Cnn::new(
                NeuralNetwork::from_onnx(MODEL_DATA)
                    .unwrap()
                    .load()
                    .unwrap(),
                CnnInputShape::NCHW,
                create_linear_color_mapper(0.0..=1.0),
            )
            .unwrap()
        });

        &MODEL
    }
}