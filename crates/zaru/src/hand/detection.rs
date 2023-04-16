//! Palm detection.

// TODO(GPU/wonnx): support mode=linear for `Resize` node

use crate::nn::{ColorMapper, Outputs};
use crate::num::sigmoid;
use crate::{image::Resolution, rect::Rect};
use include_blob::include_blob;
use nalgebra::{Point2, Rotation2, Vector2};
use once_cell::sync::Lazy;

use crate::{
    detection::{
        ssd::{Anchor, AnchorParams, Anchors, LayerInfo},
        Detection, Detections, Network,
    },
    nn::{Cnn, CnnInputShape, NeuralNetwork},
};

/// A keypoint of a [`Detection`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keypoint {
    Wrist = 0,
    IndexFingerMcp = 1,
    MiddleFingerMcp = 2,
    RingFingerMcp = 3,
    PinkyMcp = 4,
    ThumbCmc = 5,
    ThumbMcp = 6,
}

/// A list of all [`Keypoint`]s.
pub const ALL_KEYPOINTS: &[Keypoint] = &[
    Keypoint::Wrist,
    Keypoint::IndexFingerMcp,
    Keypoint::MiddleFingerMcp,
    Keypoint::RingFingerMcp,
    Keypoint::PinkyMcp,
    Keypoint::ThumbCmc,
    Keypoint::ThumbMcp,
];

/// A "lightweight" palm detection network.
///
/// **Note**: This network is still *extremely* heavy compared to other "light" networks. CPU
/// inference seems to take around 10 times as long as for the short-range *face* detection network,
/// for example.
pub struct LiteNetwork;

static LITE_MODEL: Lazy<Cnn> = Lazy::new(|| {
    let model_data = include_blob!("../../3rdparty/onnx/palm_detection_lite.onnx");
    Cnn::new(
        NeuralNetwork::from_onnx(model_data)
            .unwrap()
            .load()
            .unwrap(),
        CnnInputShape::NCHW,
        ColorMapper::linear(0.0..=1.0),
    )
    .unwrap()
});

impl Network for LiteNetwork {
    type Classes = ();

    fn cnn(&self) -> &Cnn {
        &LITE_MODEL
    }

    fn extract(&self, outputs: &Outputs, threshold: f32, detections: &mut Detections) {
        extract_outputs(
            self.cnn().input_resolution(),
            outputs,
            threshold,
            detections,
        );
    }
}

/// The full-range palm detection network.
///
/// This is about 15% slower than [`LiteNetwork`].
pub struct FullNetwork;

static FULL_MODEL: Lazy<Cnn> = Lazy::new(|| {
    let model_data = include_blob!("../../3rdparty/onnx/palm_detection_full.onnx");
    Cnn::new(
        NeuralNetwork::from_onnx(model_data)
            .unwrap()
            .load()
            .unwrap(),
        CnnInputShape::NCHW,
        ColorMapper::linear(0.0..=1.0),
    )
    .unwrap()
});

impl Network for FullNetwork {
    type Classes = ();

    fn cnn(&self) -> &Cnn {
        &FULL_MODEL
    }

    fn extract(&self, outputs: &Outputs, threshold: f32, detections: &mut Detections) {
        extract_outputs(
            self.cnn().input_resolution(),
            outputs,
            threshold,
            detections,
        );
    }
}

fn extract_outputs(
    input_res: Resolution,
    outputs: &Outputs,
    thresh: f32,
    detections: &mut Detections,
) {
    static ANCHORS: Lazy<Anchors> = Lazy::new(|| {
        Anchors::calculate(&AnchorParams {
            layers: &[LayerInfo::new(2, 24, 24), LayerInfo::new(6, 12, 12)],
        })
    });

    let num_anchors = ANCHORS.anchor_count();
    let boxes = &outputs[0];
    let confidences = &outputs[1];

    assert_eq!(boxes.shape(), &[1, num_anchors, 18]);
    assert_eq!(confidences.shape(), &[1, num_anchors, 1]);

    for (index, view) in confidences.index([0]).iter().enumerate() {
        let conf = sigmoid(view.as_slice()[0]);

        if conf < thresh {
            continue;
        }

        let tensor_view = boxes.index([0, index]);
        let box_params = tensor_view.as_slice();
        detections.push(
            (),
            extract_detection(&ANCHORS[index], input_res, box_params, conf),
        );
    }
}

fn extract_detection(
    anchor: &Anchor,
    input_res: Resolution,
    box_params: &[f32],
    confidence: f32,
) -> Detection {
    assert_eq!(box_params.len(), 18);

    let input_w = input_res.width() as f32;
    let input_h = input_res.height() as f32;

    let xc = box_params[0] + anchor.x_center() * input_w;
    let yc = box_params[1] + anchor.y_center() * input_h;
    let w = box_params[2];
    let h = box_params[3];
    let lm = |x, y| {
        crate::detection::Keypoint::new(
            x + anchor.x_center() * input_w,
            y + anchor.y_center() * input_h,
        )
    };

    let mut det = Detection::with_keypoints(
        confidence,
        Rect::from_center(xc, yc, w, h),
        vec![
            lm(box_params[4], box_params[5]),
            lm(box_params[6], box_params[7]),
            lm(box_params[8], box_params[9]),
            lm(box_params[10], box_params[11]),
            lm(box_params[12], box_params[13]),
            lm(box_params[14], box_params[15]),
            lm(box_params[16], box_params[17]),
        ],
    );

    let a = det.keypoints()[Keypoint::MiddleFingerMcp as usize];
    let finger = Point2::new(a.x(), a.y());
    let b = det.keypoints()[Keypoint::Wrist as usize];
    let wrist = Point2::new(b.x(), b.y());

    let rel = wrist - finger;
    det.set_angle(Rotation2::rotation_between(&Vector2::y(), &rel).angle());

    det
}
