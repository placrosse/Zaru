//! Human body detection.

// TODO(GPU/wonnx): support mode=linear for `Resize` node

use std::sync::OnceLock;

use include_blob::include_blob;
use zaru_linalg::vec2;

use crate::image::Resolution;
use crate::nn::{ColorMapper, Outputs};
use crate::num::sigmoid;
use crate::rect::Rect;
use crate::{
    detection::{
        ssd::{Anchor, AnchorParams, Anchors, LayerInfo},
        Detection, Detections, Network,
    },
    nn::{Cnn, CnnInputShape, NeuralNetwork},
};

/// Body pose detection network.
///
/// Use with [`Detector`](crate::detection::Detector).
///
/// This network detects human bodies and computes several keypoints documented in [`Keypoint`].
pub struct PoseNetwork;

impl Network for PoseNetwork {
    type Classes = ();

    fn cnn(&self) -> &Cnn {
        static MODEL: OnceLock<Cnn> = OnceLock::new();
        MODEL.get_or_init(|| {
            let model_data = include_blob!("../../3rdparty/onnx/pose_detection.onnx");
            Cnn::new(
                NeuralNetwork::from_onnx(model_data).load().unwrap(),
                CnnInputShape::NCHW,
                ColorMapper::linear(-1.0..=1.0),
            )
            .unwrap()
        })
    }

    fn extract(&self, outputs: &Outputs, threshold: f32, detections: &mut Detections) {
        static ANCHORS: OnceLock<Anchors> = OnceLock::new();

        let anchors = ANCHORS.get_or_init(|| {
            Anchors::calculate(&AnchorParams {
                layers: &[
                    LayerInfo::new(2, 28, 28),
                    LayerInfo::new(2, 14, 14),
                    LayerInfo::new(6, 7, 7),
                ],
            })
        });

        extract_outputs(
            self.cnn().input_resolution(),
            anchors,
            outputs,
            threshold,
            detections,
        );
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

    assert_eq!(confidences.shape(), &[1, num_anchors, 1]);
    assert_eq!(boxes.shape(), &[1, num_anchors, 12]);

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
    assert_eq!(box_params.len(), 12);

    let input_size = vec2(input_res.width() as f32, input_res.height() as f32);
    let center = vec2(box_params[0], box_params[1]) + anchor.center() * input_size;

    let lm = |x, y| crate::detection::Keypoint::new(vec2(x, y) + center * input_size);

    let size = vec2(box_params[2], box_params[3]);
    Detection::with_keypoints(
        confidence,
        Rect::from_center(center.x, center.y, size.w, size.h),
        vec![
            lm(box_params[4], box_params[5]),
            lm(box_params[6], box_params[7]),
            lm(box_params[8], box_params[9]),
            lm(box_params[10], box_params[11]),
        ],
    )
}

/// Keypoints estimated by the detection network.
#[derive(Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Keypoint {
    Hips = 0,
}
