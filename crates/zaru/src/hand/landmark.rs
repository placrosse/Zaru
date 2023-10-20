//! Hand landmark prediction.

use crate::image::{draw, AsImageViewMut, Color, ImageViewMut};
use crate::iter::zip_exact;
use crate::nn::ColorMapper;
use include_blob::include_blob;
use nalgebra::{Point2, Rotation2, Vector2};
use once_cell::sync::Lazy;
use zaru_linalg::{vec2, Vec3, Vec3f};

use crate::{
    landmark::{Confidence, Estimate, Landmarks, Network},
    nn::{Cnn, CnnInputShape, NeuralNetwork, Outputs},
    slice::SliceExt,
};

/// Landmark results estimated by [`LiteNetwork`] and [`FullNetwork`].
#[derive(Clone)]
pub struct LandmarkResult {
    landmarks: Landmarks,
    presence: f32,
    raw_handedness: f32,
}

impl Default for LandmarkResult {
    fn default() -> Self {
        LandmarkResult {
            landmarks: Landmarks::new(21),
            presence: 0.0,
            raw_handedness: 0.0,
        }
    }
}

impl LandmarkResult {
    /// Returns the 3D landmark positions in the input image's coordinate system.
    pub fn landmark_positions(&self) -> impl Iterator<Item = Vec3f> + '_ {
        (0..self.landmarks.len()).map(|index| self.landmark_position(index))
    }

    /// Returns a landmark's position in the input image's coordinate system.
    pub fn landmark_position(&self, index: usize) -> Vec3f {
        self.landmarks.positions()[index]
    }

    /// Returns an iterator over the landmarks that surround the palm.
    pub fn palm_landmarks(&self) -> impl Iterator<Item = Vec3f> + '_ {
        PALM_LANDMARKS
            .iter()
            .map(|lm| self.landmark_position(*lm as usize))
    }

    /// Computes the center position of the hand's palm by averaging some of the landmarks.
    pub fn palm_center(&self) -> Vec3f {
        let mut pos = Vec3::ZERO;
        let mut count = 0;
        for lm in self.palm_landmarks() {
            pos += lm;
            count += 1;
        }

        pos / count as f32
    }

    /// Computes the clockwise rotation of the palm compared to an upright position.
    ///
    /// A rotation of 0° means that fingers are pointed upwards.
    pub fn rotation_radians(&self) -> f32 {
        let p = self.landmark_position(LandmarkIdx::MiddleFingerMcp as usize);
        let finger = Point2::new(p.x, p.y);
        let p = self.landmark_position(LandmarkIdx::Wrist as usize);
        let wrist = Point2::new(p.x, p.y);

        let rel = wrist - finger;
        Rotation2::rotation_between(&Vector2::y(), &rel).angle()
    }

    /// Returns the estimated handedness of the hand in the image.
    ///
    /// This assumes that the camera image is passed in as-is, and the returned value should only be
    /// relied on when the `presence` is over some threshold.
    pub fn handedness(&self) -> Handedness {
        if self.raw_handedness > 0.5 {
            Handedness::Right
        } else {
            Handedness::Left
        }
    }

    pub fn draw<I: AsImageViewMut>(&self, target: &mut I) {
        self.draw_impl(&mut target.as_view_mut());
    }

    fn draw_impl(&self, target: &mut ImageViewMut<'_>) {
        let hand = match self.handedness() {
            Handedness::Left => "L",
            Handedness::Right => "R",
        };

        let palm = self.palm_center().truncate();

        let a = self
            .landmark_position(LandmarkIdx::MiddleFingerMcp as usize)
            .truncate();
        let b = self
            .landmark_position(LandmarkIdx::Wrist as usize)
            .truncate();
        draw::line(target, a, b).color(Color::from_rgb8(127, 127, 127));
        draw::text(
            target,
            b,
            &format!("{:.1} deg", self.rotation_radians().to_degrees()),
        )
        .align_top();

        draw::text(target, palm - vec2(0.0, 5.0), hand);
        draw::text(
            target,
            palm + vec2(0.0, 5.0),
            &format!("confidence={:.2}", self.confidence()),
        );

        for (a, b) in CONNECTIVITY {
            let a = self.landmark_position(*a as usize).truncate();
            let b = self.landmark_position(*b as usize).truncate();

            draw::line(target, a, b).color(Color::GREEN);
        }
        for pos in self.landmark_positions() {
            draw::marker(target, pos.truncate());
        }
    }
}

impl Estimate for LandmarkResult {
    #[inline]
    fn landmarks_mut(&mut self) -> &mut Landmarks {
        &mut self.landmarks
    }

    fn angle_radians(&self) -> Option<f32> {
        Some(self.rotation_radians())
    }
}

impl Confidence for LandmarkResult {
    #[inline]
    fn confidence(&self) -> f32 {
        self.presence
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Handedness {
    Left,
    Right,
}

/// Names for the hand pose landmarks.
///
/// # Terminology
///
/// - **CMC**: [Carpometacarpal joint], the lowest joint of the thumb, located near the wrist.
/// - **MCP**: [Metacarpophalangeal joint], the lower joint forming the knuckles near the palm of
///   the hand.
/// - **PIP**: Proximal Interphalangeal joint, the joint between the MCP and DIP.
/// - **DIP**: Distal Interphalangeal joint, the highest joint of a finger.
/// - **Tip**: This landmark is just placed on the tip of the finger, above the DIP.
///
/// [Carpometacarpal joint]: https://en.wikipedia.org/wiki/Carpometacarpal_joint
/// [Metacarpophalangeal joint]: https://en.wikipedia.org/wiki/Metacarpophalangeal_joint
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LandmarkIdx {
    Wrist,
    ThumbCmc,
    ThumbMcp,
    ThumbIp,
    ThumbTip,
    IndexFingerMcp,
    IndexFingerPip,
    IndexFingerDip,
    IndexFingerTip,
    MiddleFingerMcp,
    MiddleFingerPip,
    MiddleFingerDip,
    MiddleFingerTip,
    RingFingerMcp,
    RingFingerPip,
    RingFingerDip,
    RingFingerTip,
    PinkyMcp,
    PinkyPip,
    PinkyDip,
    PinkyTip,
}

const PALM_LANDMARKS: &[LandmarkIdx] = {
    use LandmarkIdx::*;
    &[
        Wrist,
        ThumbCmc,
        IndexFingerMcp,
        MiddleFingerMcp,
        RingFingerMcp,
        PinkyMcp,
    ]
};

const CONNECTIVITY: &[(LandmarkIdx, LandmarkIdx)] = {
    use LandmarkIdx::*;
    &[
        // Surround the palm:
        (Wrist, ThumbCmc),
        (ThumbCmc, IndexFingerMcp),
        (IndexFingerMcp, MiddleFingerMcp),
        (MiddleFingerMcp, RingFingerMcp),
        (RingFingerMcp, PinkyMcp),
        (PinkyMcp, Wrist),
        // Thumb:
        (ThumbCmc, ThumbMcp),
        (ThumbMcp, ThumbIp),
        (ThumbIp, ThumbTip),
        // Index:
        (IndexFingerMcp, IndexFingerPip),
        (IndexFingerPip, IndexFingerDip),
        (IndexFingerDip, IndexFingerTip),
        // Middle:
        (MiddleFingerMcp, MiddleFingerPip),
        (MiddleFingerPip, MiddleFingerDip),
        (MiddleFingerDip, MiddleFingerTip),
        // Ring:
        (RingFingerMcp, RingFingerPip),
        (RingFingerPip, RingFingerDip),
        (RingFingerDip, RingFingerTip),
        // Pinky:
        (PinkyMcp, PinkyPip),
        (PinkyPip, PinkyDip),
        (PinkyDip, PinkyTip),
    ]
};

/// A lightweight but fairly inaccurate landmark estimation network.
///
/// Takes a bit over 20ms to run on my machine, so it can't hit 60 FPS, but it is faster than
/// [`FullNetwork`].
#[derive(Clone, Copy)]
pub struct LiteNetwork;

impl Network for LiteNetwork {
    type Output = LandmarkResult;

    fn cnn(&self) -> &Cnn {
        static MODEL: Lazy<Cnn> = Lazy::new(|| {
            let model_data = include_blob!("../../3rdparty/onnx/hand_landmark_lite.onnx");
            Cnn::new(
                NeuralNetwork::from_onnx(model_data).load().unwrap(),
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

/// A somewhat more accurate landmark estimation network that takes about 25-30% longer to infer
/// than [`LiteNetwork`] (on CPU).
#[derive(Clone, Copy)]
pub struct FullNetwork;

impl Network for FullNetwork {
    type Output = LandmarkResult;

    fn cnn(&self) -> &Cnn {
        static MODEL: Lazy<Cnn> = Lazy::new(|| {
            let model_data = include_blob!("../../3rdparty/onnx/hand_landmark_full.onnx");
            Cnn::new(
                NeuralNetwork::from_onnx(model_data).load().unwrap(),
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

fn extract(outputs: &Outputs, estimate: &mut LandmarkResult) {
    let screen_landmarks = &outputs[0];
    let presence_flag = &outputs[1];
    let handedness = &outputs[2];
    let metric_landmarks = &outputs[3];

    assert_eq!(screen_landmarks.shape(), &[1, 63]);
    assert_eq!(presence_flag.shape(), &[1, 1]);
    assert_eq!(handedness.shape(), &[1, 1]);
    assert_eq!(metric_landmarks.shape(), &[1, 63]);

    estimate.presence = presence_flag.index([0, 0]).as_singular();
    estimate.raw_handedness = handedness.index([0, 0]).as_singular();
    for (&[x, y, z], out) in zip_exact(
        screen_landmarks
            .index([0])
            .as_slice()
            .array_chunks_exact::<3>(),
        estimate.landmarks.positions_mut(),
    ) {
        out[0] = x;
        out[1] = y;
        out[2] = z;
    }
}
