//! Face detection module.
//!
//! This uses one of the "BlazeFace" neural networks also used in MediaPipe's [Face Detection]
//! module.
//!
//! [Face Detection]: https://google.github.io/mediapipe/solutions/face_detection

use crate::nn::{ColorMapper, Outputs};
use crate::num::sigmoid;
use crate::{image::Resolution, rect::Rect};
use include_blob::include_blob;
use nalgebra::{Rotation2, Vector2};
use once_cell::sync::Lazy;

use crate::{
    detection::{
        ssd::{Anchor, AnchorParams, Anchors, LayerInfo},
        Detection, Detections, Network,
    },
    nn::{Cnn, CnnInputShape, NeuralNetwork},
};

pub enum Keypoint {
    LeftEye = 0,
    RightEye = 1,
}

/// A small and efficient face detection network, best for faces in <3m of the camera.
pub struct ShortRangeNetwork;

static SHORT_RANGE_MODEL: Lazy<Cnn> = Lazy::new(|| {
    let model_data = include_blob!("../../3rdparty/onnx/face_detection_short_range.onnx");
    Cnn::new(
        NeuralNetwork::from_onnx(model_data)
            .unwrap()
            .load()
            .unwrap(),
        CnnInputShape::NCHW,
        ColorMapper::linear(-1.0..=1.0),
    )
    .unwrap()
});

impl Network for ShortRangeNetwork {
    type Classes = ();

    fn cnn(&self) -> &Cnn {
        &SHORT_RANGE_MODEL
    }

    fn extract(&self, outputs: &Outputs, thresh: f32, detections: &mut Detections) {
        static ANCHORS: Lazy<Anchors> = Lazy::new(|| {
            Anchors::calculate(&AnchorParams {
                layers: &[LayerInfo::new(2, 16, 16), LayerInfo::new(6, 8, 8)],
            })
        });

        let res = SHORT_RANGE_MODEL.input_resolution();
        extract_outputs(res, &ANCHORS, outputs, thresh, detections);
    }
}

/// A larger detection network with a greater detection range, but slower inference speed (around 5
/// times that of [`ShortRangeNetwork`]).
pub struct FullRangeNetwork;

static FULL_RANGE_MODEL: Lazy<Cnn> = Lazy::new(|| {
    let model_data = include_blob!("../../3rdparty/onnx/face_detection_full_range.onnx");
    Cnn::new(
        NeuralNetwork::from_onnx(model_data)
            .unwrap()
            .load()
            .unwrap(),
        CnnInputShape::NCHW,
        ColorMapper::linear(-1.0..=1.0),
    )
    .unwrap()
});

impl Network for FullRangeNetwork {
    type Classes = ();

    fn cnn(&self) -> &Cnn {
        &FULL_RANGE_MODEL
    }

    fn extract(&self, outputs: &Outputs, thresh: f32, detections: &mut Detections) {
        static ANCHORS: Lazy<Anchors> = Lazy::new(|| {
            Anchors::calculate(&AnchorParams {
                layers: &[LayerInfo::new(1, 48, 48)],
            })
        });

        let res = FULL_RANGE_MODEL.input_resolution();
        extract_outputs(res, &ANCHORS, outputs, thresh, detections);
    }
}

fn extract_outputs(
    input_res: Resolution,
    anchors: &Anchors,
    outputs: &Outputs,
    thresh: f32,
    detections: &mut Detections,
) {
    let num_anchors = anchors.anchor_count();
    let boxes = &outputs[0];
    let confidences = &outputs[1];

    assert_eq!(boxes.shape(), &[1, num_anchors, 16]);
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
            extract_detection(&anchors[index], input_res, box_params, conf),
        );
    }
}

fn extract_detection(
    anchor: &Anchor,
    input_res: Resolution,
    box_params: &[f32],
    confidence: f32,
) -> Detection {
    assert_eq!(box_params.len(), 16);

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
        ],
    );

    let left_eye = det.keypoints()[Keypoint::LeftEye as usize];
    let right_eye = det.keypoints()[Keypoint::RightEye as usize];
    let left_to_right_eye =
        Vector2::new(right_eye.x() - left_eye.x(), right_eye.y() - left_eye.y());
    let angle = Rotation2::rotation_between(&Vector2::x(), &left_to_right_eye).angle();
    det.set_angle(angle);

    det
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{detection::Detector, test};

    #[test]
    fn detects_face() {
        let mut det = Detector::new(ShortRangeNetwork);
        let detections = det.detect(test::sad_linus_full());
        let detection = detections.iter().next().expect("no detection");

        assert!(detection.confidence() >= 0.9, "{}", detection.confidence());
        let angle = detection.angle().to_degrees();
        assert!(angle < 5.0, "{angle}");
    }
}
