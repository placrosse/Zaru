//! Human body detection.

use crate::nn::Outputs;
use crate::num::sigmoid;
use crate::rect::Rect;
use include_blob::include_blob;
use once_cell::sync::Lazy;

use crate::image::Resolution;
use crate::{
    detection::{
        ssd::{Anchor, AnchorParams, Anchors, LayerInfo},
        Detection, Detections, Network,
    },
    nn::{create_linear_color_mapper, Cnn, CnnInputShape, NeuralNetwork},
};

static MODEL: Lazy<Cnn> = Lazy::new(|| {
    let model_data = include_blob!("../../3rdparty/onnx/pose_detection.onnx");
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

/// Body pose detection network.
///
/// Use with [`Detector`](crate::detection::Detector).
///
/// This network detects human bodies and computes several keypoints documented in [`Keypoint`].
pub struct PoseNetwork;

impl Network for PoseNetwork {
    type Classes = ();

    fn cnn(&self) -> &Cnn {
        &MODEL
    }

    fn extract(&self, outputs: &Outputs, threshold: f32, detections: &mut Detections) {
        static ANCHORS: Lazy<Anchors> = Lazy::new(|| {
            Anchors::calculate(&AnchorParams {
                layers: &[
                    LayerInfo::new(2, 28, 28),
                    LayerInfo::new(2, 14, 14),
                    LayerInfo::new(6, 7, 7),
                ],
            })
        });

        extract_outputs(
            MODEL.input_resolution(),
            &ANCHORS,
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

    Detection::with_keypoints(
        confidence,
        Rect::from_center(xc, yc, w, h),
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
