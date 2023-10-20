//! Face detection module.
//!
//! This uses one of the "BlazeFace" neural networks also used in MediaPipe's [Face Detection]
//! module.
//!
//! [Face Detection]: https://google.github.io/mediapipe/solutions/face_detection

use std::sync::OnceLock;

use crate::nn::{ColorMapper, Outputs};
use crate::num::sigmoid;
use crate::{image::Resolution, rect::Rect};
use include_blob::include_blob;
use zaru_linalg::{vec2, Vec2};

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

impl Network for ShortRangeNetwork {
    type Classes = ();

    fn cnn(&self) -> &Cnn {
        static MODEL: OnceLock<Cnn> = OnceLock::new();
        MODEL.get_or_init(|| {
            let model_data = include_blob!("../../3rdparty/onnx/face_detection_short_range.onnx");
            Cnn::new(
                NeuralNetwork::from_onnx(model_data).load().unwrap(),
                CnnInputShape::NCHW,
                ColorMapper::linear(-1.0..=1.0),
            )
            .unwrap()
        })
    }

    fn extract(&self, outputs: &Outputs, thresh: f32, detections: &mut Detections) {
        static ANCHORS: OnceLock<Anchors> = OnceLock::new();

        let anchors = ANCHORS.get_or_init(|| {
            Anchors::calculate(&AnchorParams {
                layers: &[LayerInfo::new(2, 16, 16), LayerInfo::new(6, 8, 8)],
            })
        });
        let res = self.cnn().input_resolution();
        extract_outputs(res, anchors, outputs, thresh, detections);
    }
}

/// A larger detection network with a greater detection range, but slower inference speed (around 5
/// times that of [`ShortRangeNetwork`]).
pub struct FullRangeNetwork;

// TODO(GPU/wonnx): support mode=linear for `Resize` node

impl Network for FullRangeNetwork {
    type Classes = ();

    fn cnn(&self) -> &Cnn {
        static MODEL: OnceLock<Cnn> = OnceLock::new();
        MODEL.get_or_init(|| {
            let model_data = include_blob!("../../3rdparty/onnx/face_detection_full_range.onnx");
            Cnn::new(
                NeuralNetwork::from_onnx(model_data).load().unwrap(),
                CnnInputShape::NCHW,
                ColorMapper::linear(-1.0..=1.0),
            )
            .unwrap()
        })
    }

    fn extract(&self, outputs: &Outputs, thresh: f32, detections: &mut Detections) {
        static ANCHORS: OnceLock<Anchors> = OnceLock::new();

        let anchors = ANCHORS.get_or_init(|| {
            Anchors::calculate(&AnchorParams {
                layers: &[LayerInfo::new(1, 48, 48)],
            })
        });
        let res = self.cnn().input_resolution();
        extract_outputs(res, anchors, outputs, thresh, detections);
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

    let input_size = vec2(input_res.width() as f32, input_res.height() as f32);
    let center = vec2(box_params[0], box_params[1]) + anchor.center() * input_size;

    let lm = |x, y| crate::detection::Keypoint::new(vec2(x, y) + center * input_size);

    let size = vec2(box_params[2], box_params[3]);
    let mut det = Detection::with_keypoints(
        confidence,
        Rect::from_center(center.x, center.y, size.w, size.h),
        vec![
            lm(box_params[4], box_params[5]),
            lm(box_params[6], box_params[7]),
            lm(box_params[8], box_params[9]),
            lm(box_params[10], box_params[11]),
            lm(box_params[12], box_params[13]),
            lm(box_params[14], box_params[15]),
        ],
    );

    let left_eye = det.keypoints()[Keypoint::LeftEye as usize].position();
    let right_eye = det.keypoints()[Keypoint::RightEye as usize].position();
    let left_to_right_eye = right_eye - left_eye;
    det.set_angle(left_to_right_eye.signed_angle_to(Vec2::X));

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

        assert!(detection.confidence() >= 0.8, "{}", detection.confidence());
        let angle = detection.angle().to_degrees();
        assert!(angle.abs() < 5.0, "{angle}");
    }
}
