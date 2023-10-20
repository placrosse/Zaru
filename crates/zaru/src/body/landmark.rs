//! Body pose landmark prediction. Not yet fully implemented.

// TODO(GPU/wonnx): support mode=linear for `Resize` node

use crate::image::{draw, AsImageViewMut, Color, ImageViewMut};
use crate::landmark::{Confidence, Estimate, Landmark, Landmarks, Network};
use crate::nn::{ColorMapper, Outputs};
use crate::num::sigmoid;
use include_blob::include_blob;
use once_cell::sync::Lazy;

use crate::{
    nn::{Cnn, CnnInputShape, NeuralNetwork},
    slice::SliceExt,
};

#[derive(Clone)]
pub struct LandmarkResult {
    pose_presence: f32,
    landmarks: Landmarks,
}

impl Default for LandmarkResult {
    fn default() -> Self {
        Self {
            pose_presence: 0.0,
            landmarks: Landmarks::new(33 + 6),
        }
    }
}

impl Estimate for LandmarkResult {
    #[inline]
    fn landmarks_mut(&mut self) -> &mut Landmarks {
        &mut self.landmarks
    }
}

impl Confidence for LandmarkResult {
    #[inline]
    fn confidence(&self) -> f32 {
        self.pose_presence
    }
}

impl LandmarkResult {
    pub fn pose_landmarks(&self) -> impl Iterator<Item = Landmark> + '_ {
        (0..33).map(|i| self.landmarks.get(i))
    }

    pub fn aux_landmarks(&self) -> impl Iterator<Item = Landmark> + '_ {
        (33..33 + 6).map(|i| self.landmarks.get(i))
    }

    pub fn get(&self, i: LandmarkIdx) -> Landmark {
        self.landmarks.get(i as usize)
    }

    #[inline]
    pub fn presence(&self) -> f32 {
        self.pose_presence
    }

    pub fn draw<I: AsImageViewMut>(&self, target: &mut I) {
        self.draw_impl(&mut target.as_view_mut());
    }

    fn draw_impl(&self, target: &mut ImageViewMut<'_>) {
        for (a, b) in COARSE_CONNECTIVITY {
            let a = self.get(*a);
            let b = self.get(*b);
            draw::line(target, a.position().truncate(), b.position().truncate());
        }

        for lm in self.pose_landmarks() {
            draw::marker(target, lm.position().truncate()).size(9);
        }
        for lm in self.aux_landmarks() {
            draw::marker(target, lm.position().truncate()).color(Color::YELLOW);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LandmarkIdx {
    Nose = 0,
    LeftEyeInner = 1,
    LeftEye = 2,
    LeftEyeOuter = 3,
    RightEyeInner = 4,
    RightEye = 5,
    RightEyeOuter = 6,
    LeftEar = 7,
    RightEar = 8,
    MouthLeft = 9,
    MouthRight = 10,
    LeftShoulder = 11,
    RightShoulder = 12,
    LeftElbow = 13,
    RightElbow = 14,
    LeftWrist = 15,
    RightWrist = 16,
    LeftPinky = 17,
    RightPinky = 18,
    LeftIndex = 19,
    RightIndex = 20,
    LeftThumb = 21,
    RightThumb = 22,
    LeftHip = 23,
    RightHip = 24,
    LeftKnee = 25,
    RightKnee = 26,
    LeftAnkle = 27,
    RightAnkle = 28,
    LeftHeel = 29,
    RightHeel = 30,
    LeftFootIndex = 31,
    RightFootIndex = 32,
}

const COARSE_CONNECTIVITY: &[(LandmarkIdx, LandmarkIdx)] = {
    use LandmarkIdx::*;
    &[
        (LeftShoulder, RightShoulder),
        (LeftShoulder, LeftElbow),
        (LeftElbow, LeftWrist),
        (RightShoulder, RightElbow),
        (RightElbow, RightWrist),
        (LeftShoulder, LeftHip),
        (LeftHip, LeftAnkle),
        (LeftAnkle, LeftHeel),
        (LeftAnkle, LeftFootIndex),
        (RightShoulder, RightHip),
        (RightHip, RightAnkle),
        (RightAnkle, RightHeel),
        (RightAnkle, RightFootIndex),
    ]
};

pub struct LiteNetwork;

impl Network for LiteNetwork {
    type Output = LandmarkResult;

    fn cnn(&self) -> &Cnn {
        static MODEL: Lazy<Cnn> = Lazy::new(|| {
            let model_data = include_blob!("../../3rdparty/onnx/pose_landmark_lite.onnx");
            Cnn::new(
                NeuralNetwork::from_onnx(model_data)
                    .with_output_selection([0, 1])
                    .load()
                    .unwrap(),
                CnnInputShape::NCHW,
                ColorMapper::linear(0.0..=1.0),
            )
            .unwrap()
        });

        &MODEL
    }

    fn extract(&self, outputs: &Outputs, estimate: &mut Self::Output) {
        extract(outputs, estimate);
    }
}

pub struct FullNetwork;

impl Network for FullNetwork {
    type Output = LandmarkResult;

    fn cnn(&self) -> &Cnn {
        static MODEL: Lazy<Cnn> = Lazy::new(|| {
            let model_data = include_blob!("../../3rdparty/onnx/pose_landmark_full.onnx");
            Cnn::new(
                NeuralNetwork::from_onnx(model_data)
                    .with_output_selection([0, 1])
                    .load()
                    .unwrap(),
                CnnInputShape::NCHW,
                ColorMapper::linear(0.0..=1.0),
            )
            .unwrap()
        });

        &MODEL
    }

    fn extract(&self, outputs: &Outputs, estimate: &mut Self::Output) {
        extract(outputs, estimate);
    }
}

// NB: There's also a "heavy" network, but it's >25 MB, so we don't support it. The full network
// should already perform pretty well.

fn extract(outputs: &Outputs, estimate: &mut LandmarkResult) {
    let screen_landmarks = &outputs[0];
    let pose_flag = &outputs[1];

    // Other outputs are turned off during load.
    /*let segmentation = &outputs[2];
    let heatmap = &outputs[3];
    let world_landmarks = &outputs[4];*/

    // 33 pose landmarks (`LandmarkIdx`), 6 auxiliary landmarks -> 39 total
    assert_eq!(screen_landmarks.shape(), &[1, 39 * 5]); // 5 values each
    assert_eq!(pose_flag.shape(), &[1, 1]);
    /*assert_eq!(segmentation.shape(), &[1, 256, 256, 1]);
    assert_eq!(heatmap.shape(), &[1, 64, 64, 39]);
    assert_eq!(world_landmarks.shape(), &[1, 39 * 3]); // 3 values each*/

    estimate.pose_presence = pose_flag.index([0, 0]).as_singular();

    for (i, &[x, y, z, visibility, presence]) in screen_landmarks
        .index([0])
        .as_slice()
        .array_chunks_exact::<5>()
        .enumerate()
    {
        estimate.landmarks.set(
            i,
            Landmark::new([x, y, z])
                .with_visibility(sigmoid(visibility))
                .with_presence(sigmoid(presence)),
        );
    }
}
