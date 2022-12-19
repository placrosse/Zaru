//! Human body detection.

use once_cell::sync::Lazy;
use zaru_utils::num::sigmoid;

use crate::{
    detection::{
        nms::NonMaxSuppression,
        ssd::{Anchor, AnchorParams, Anchors, LayerInfo},
        BoundingRect, RawDetection,
    },
    nn::{create_linear_color_mapper, point_to_img, Cnn, CnnInputShape, NeuralNetwork},
    timer::Timer,
};
use zaru_image::{
    draw, AsImageView, AsImageViewMut, Color, ImageView, ImageViewMut, Rect, Resolution,
};

static MODEL: Lazy<Cnn> = Lazy::new(|| {
    let model_data = include_blob::include_bytes!("../../3rdparty/onnx/pose_detection.onnx");
    Cnn::new(
        NeuralNetwork::from_onnx(model_data)
            .unwrap()
            .load()
            .unwrap(),
        CnnInputShape::NCHW,
        create_linear_color_mapper(-1.0..=1.0),
    )
    .unwrap()
});

pub struct PoseDetector {
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

impl PoseDetector {
    const DEFAULT_THRESH: f32 = 0.5;

    pub fn new() -> Self {
        Self {
            cnn: &MODEL,
            anchors: Anchors::calculate(&AnchorParams {
                layers: &[
                    LayerInfo::new(2, 28, 28),
                    LayerInfo::new(2, 14, 14),
                    LayerInfo::new(6, 7, 7),
                ],
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

        let mut image = image;
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

            assert_eq!(confidences.shape(), &[1, num_anchors, 1]);
            assert_eq!(boxes.shape(), &[1, num_anchors, 12]);

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
    assert_eq!(box_params.len(), 12);

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

    pub fn keypoints(&self) -> impl Iterator<Item = (i32, i32)> + '_ {
        self.raw
            .keypoints()
            .iter()
            .map(|lm| point_to_img(lm.x(), lm.y(), &self.full_res))
    }

    pub fn keypoint_hips(&self) -> (i32, i32) {
        let lm = self.raw.keypoints()[0];
        point_to_img(lm.x(), lm.y(), &self.full_res)
    }

    /// Draws the bounding box and keypoints of this detection onto an image.
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

        draw::rect(image, self.bounding_rect()).color(Color::GREEN);
        for (i, lm) in self.raw.keypoints().iter().enumerate() {
            let (x, y) = point_to_img(lm.x(), lm.y(), &self.full_res);
            draw::marker(image, x, y).color(Color::GREEN);
            draw::text(image, x, y - 3, &i.to_string())
                .align_bottom()
                .color(Color::GREEN);
        }
    }
}
