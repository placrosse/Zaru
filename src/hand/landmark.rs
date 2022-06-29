//! Hand landmark prediction.

use nalgebra::{Point2, Rotation2, Vector2};
use once_cell::sync::Lazy;

use crate::{
    image::{self, AsImageView, AsImageViewMut, Color, ImageView, ImageViewMut},
    iter::zip_exact,
    landmark::{Estimation, Estimator, Landmarks},
    nn::{create_linear_color_mapper, unadjust_aspect_ratio, Cnn, CnnInputShape, NeuralNetwork},
    resolution::Resolution,
    timer::Timer,
};

#[derive(Clone)]
pub struct Landmarker {
    cnn: &'static Cnn,
    t_resize: Timer,
    t_infer: Timer,
    result_buffer: LandmarkResult,
}

impl Landmarker {
    pub fn new<N: LandmarkNetwork>(network: N) -> Self {
        drop(network);
        Self {
            cnn: N::cnn(),
            t_resize: Timer::new("resize"),
            t_infer: Timer::new("infer"),
            result_buffer: LandmarkResult {
                landmarks: Landmarks::new(21),
                presence: 0.0,
                raw_handedness: 0.0,
            },
        }
    }

    /// Returns the expected input resolution of the internal neural network.
    pub fn input_resolution(&self) -> Resolution {
        self.cnn.input_resolution()
    }

    /// Computes hand landmarks in `image`.
    pub fn compute<V: AsImageView>(&mut self, image: &V) -> &LandmarkResult {
        self.compute_impl(image.as_view())
    }

    fn compute_impl(&mut self, image: ImageView<'_>) -> &LandmarkResult {
        let input_res = self.input_resolution();
        let full_res = image.resolution();
        let aspect_ratio = full_res.aspect_ratio().unwrap();

        let mut image = image;
        let resized;
        if image.resolution() != input_res {
            resized = self.t_resize.time(|| image.aspect_aware_resize(input_res));
            image = resized.as_view();
        }
        let outputs = self.t_infer.time(|| self.cnn.estimate(&image)).unwrap();
        log::trace!("cnn outputs: {:?}", outputs);

        let screen_landmarks = &outputs[0];
        let presence_flag = &outputs[1];
        let handedness = &outputs[2];
        let metric_landmarks = &outputs[3];

        assert_eq!(screen_landmarks.shape(), &[1, 63]);
        assert_eq!(presence_flag.shape(), &[1, 1]);
        assert_eq!(handedness.shape(), &[1, 1]);
        assert_eq!(metric_landmarks.shape(), &[1, 63]);

        self.result_buffer.presence = presence_flag.index([0, 0]).as_singular();
        self.result_buffer.raw_handedness = handedness.index([0, 0]).as_singular();
        for (coords, out) in zip_exact(
            screen_landmarks.index([0]).as_slice().chunks(3),
            self.result_buffer.landmarks.positions_mut(),
        ) {
            let [x, y, z] = [coords[0], coords[1], coords[2]];
            let (x, y) = unadjust_aspect_ratio(
                x / input_res.width() as f32,
                y / input_res.height() as f32,
                aspect_ratio,
            );
            let (x, y) = (x * full_res.width() as f32, y * full_res.height() as f32);

            out[0] = x;
            out[1] = y;
            out[2] = z;
        }

        &self.result_buffer
    }

    pub fn timers(&self) -> impl Iterator<Item = &Timer> + '_ {
        [&self.t_resize, &self.t_infer].into_iter()
    }
}

/// Landmark results returned by [`Landmarker::compute`].
#[derive(Clone)]
pub struct LandmarkResult {
    landmarks: Landmarks,
    presence: f32,
    raw_handedness: f32,
}

impl LandmarkResult {
    pub fn move_by(&mut self, x: f32, y: f32, z: f32) {
        for pos in self.landmarks.positions_mut() {
            pos[0] += x;
            pos[1] += y;
            pos[2] += z;
        }
    }

    /// Returns the 3D landmark positions in the input image's coordinate system.
    pub fn landmark_positions(&self) -> impl Iterator<Item = [f32; 3]> + '_ {
        (0..self.landmark_count()).map(|index| self.landmark_position(index))
    }

    /// Returns a landmark's position in the input image's coordinate system.
    pub fn landmark_position(&self, index: usize) -> [f32; 3] {
        self.landmarks.positions()[index]
    }

    /// Returns an iterator over the landmarks that surround the palm.
    pub fn palm_landmarks(&self) -> impl Iterator<Item = [f32; 3]> + '_ {
        PALM_LANDMARKS
            .iter()
            .map(|lm| self.landmark_position(*lm as usize))
    }

    /// Computes the center position of the hand's palm by averaging some of the landmarks.
    pub fn palm_center(&self) -> [f32; 3] {
        let mut pos = (0.0, 0.0, 0.0);
        let mut count = 0;
        for [x, y, z] in self.palm_landmarks() {
            pos.0 += x;
            pos.1 += y;
            pos.2 += z;
            count += 1;
        }

        [
            pos.0 / count as f32,
            pos.1 / count as f32,
            pos.2 / count as f32,
        ]
    }

    /// Computes the clockwise rotation of the palm compared to an upright position.
    ///
    /// A rotation of 0° means that fingers are pointed upwards.
    pub fn rotation_radians(&self) -> f32 {
        let [x, y, _] = self.landmark_position(LandmarkIdx::MiddleFingerMcp as usize);
        let finger = Point2::new(x as f32, y as f32);
        let [x, y, _] = self.landmark_position(LandmarkIdx::Wrist as usize);
        let wrist = Point2::new(x as f32, y as f32);

        let rel = wrist - finger;
        Rotation2::rotation_between(&Vector2::y(), &rel).angle()
    }

    #[inline]
    pub fn landmark_count(&self) -> usize {
        self.landmarks.len()
    }

    /// Returns the presence flag, indicating the confidence of whether a hand was in the input
    /// image.
    ///
    /// The value is between 0.0 and 1.0, with higher values indicating higher confidence that a
    /// hand was present.
    pub fn presence(&self) -> f32 {
        self.presence
    }

    /// Returns the estimated handedness of the hand in the image.
    ///
    /// This assumes that the camera image is passed in as-is, and the returned value should only be
    /// relied on when `presence` is over some threshold.
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

        let [palm_x, palm_y, _] = self.palm_center();
        let (palm_x, palm_y) = (palm_x as i32, palm_y as i32);

        let [a_x, a_y, _] = self.landmark_position(LandmarkIdx::MiddleFingerMcp as usize);
        let [b_x, b_y, _] = self.landmark_position(LandmarkIdx::Wrist as usize);
        image::draw_line(target, a_x as _, a_y as _, b_x as _, b_y as _)
            .color(Color::from_rgb8(127, 127, 127));
        image::draw_text(
            target,
            b_x as _,
            b_y as _,
            &format!("{:.1} deg", self.rotation_radians().to_degrees()),
        )
        .align_top();

        image::draw_text(target, palm_x, palm_y - 5, hand);
        image::draw_text(
            target,
            palm_x,
            palm_y + 5,
            &format!("presence={:.2}", self.presence()),
        );

        for (a, b) in CONNECTIVITY {
            let [a_x, a_y, _] = self.landmark_position(*a as usize);
            let [b_x, b_y, _] = self.landmark_position(*b as usize);

            image::draw_line(target, a_x as _, a_y as _, b_x as _, b_y as _).color(Color::GREEN);
        }
        for [x, y, _] in self.landmark_positions() {
            image::draw_marker(target, x as i32, y as i32);
        }
    }
}

impl Estimation for LandmarkResult {
    fn confidence(&self) -> f32 {
        self.presence()
    }

    fn landmarks(&self) -> &crate::landmark::Landmarks {
        &self.landmarks
    }

    fn angle_radians(&self) -> Option<f32> {
        Some(self.rotation_radians())
    }
}

impl Estimator for Landmarker {
    type Estimation = LandmarkResult;

    fn estimate<V: AsImageView>(&mut self, image: &V) -> &Self::Estimation {
        self.compute(image)
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

pub trait LandmarkNetwork {
    fn cnn() -> &'static Cnn;
}

/// A lightweight but fairly inaccurate landmark estimation network.
///
/// Takes a bit over 20ms to run on my machine, so it can't hit 60 FPS, but it is faster than
/// [`FullNetwork`].
pub struct LiteNetwork;

impl LandmarkNetwork for LiteNetwork {
    fn cnn() -> &'static Cnn {
        const MODEL_DATA: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/3rdparty/onnx/hand_landmark_lite.onnx"
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

/// A somewhat more accurate landmark estimation network that takes about 25-30% longer to infer
/// than [`LiteNetwork`] (on CPU).
pub struct FullNetwork;

impl LandmarkNetwork for FullNetwork {
    fn cnn() -> &'static Cnn {
        const MODEL_DATA: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/3rdparty/onnx/hand_landmark_full.onnx"
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
