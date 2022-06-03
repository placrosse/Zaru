//! Body pose landmark prediction. Not yet fully implemented.

#![allow(warnings)]

use once_cell::sync::Lazy;

use crate::{
    image::{AsImageView, ImageView},
    nn::{create_linear_color_mapper, Cnn, CnnInputShape, NeuralNetwork},
    resolution::{AspectRatio, Resolution},
    timer::Timer,
};

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
                landmarks: Landmarks {
                    positions: [(0.0, 0.0, 0.0); 33],
                },
                face_flag: 0.0,
                orig_res: Resolution::new(1, 1),
                orig_aspect: AspectRatio::SQUARE,
                input_res: N::cnn().input_resolution(),
            },
        }
    }

    /// Returns the expected input resolution of the internal neural network.
    pub fn input_resolution(&self) -> Resolution {
        self.cnn.input_resolution()
    }

    /// Computes facial landmarks in `image`.
    ///
    /// `image` must be a cropped image of a face. When using [`Detector`], the
    /// rectangle returned by [`Detection::bounding_rect_loose`] produces good results.
    ///
    /// The image should depict a face that is mostly upright. Results will be poor if the face is
    /// rotated too much.
    ///
    /// [`Detector`]: super::detector::Detector
    /// [`Detection::bounding_rect_loose`]: super::detector::Detection::bounding_rect_loose
    pub fn compute<V: AsImageView>(&mut self, image: &V) -> &LandmarkResult {
        self.compute_impl(image.as_view())
    }

    fn compute_impl(&mut self, image: ImageView<'_>) -> &LandmarkResult {
        let input_res = self.input_resolution();
        let full_res = image.resolution();
        self.result_buffer.orig_res = full_res;
        self.result_buffer.orig_aspect = full_res.aspect_ratio();

        let mut image = image.reborrow();
        let resized;
        if image.resolution() != input_res {
            resized = self.t_resize.time(|| image.aspect_aware_resize(input_res));
            image = resized.as_view();
        }
        let result = self.t_infer.time(|| self.cnn.estimate(&image)).unwrap();
        log::trace!("inference result: {:?}", result);

        todo!()
    }
}

/// Landmark results returned by [`Landmarker::compute`].
#[derive(Clone)]
pub struct LandmarkResult {
    landmarks: Landmarks,
    face_flag: f32,
    orig_res: Resolution,
    orig_aspect: AspectRatio,
    input_res: Resolution,
}

/// Raw landmark positions.
#[derive(Clone)]
pub struct Landmarks {
    positions: [(f32, f32, f32); 33],
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

pub trait LandmarkNetwork {
    fn cnn() -> &'static Cnn;
}

pub struct LiteNetwork;

impl LandmarkNetwork for LiteNetwork {
    fn cnn() -> &'static Cnn {
        const MODEL_DATA: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/3rdparty/onnx/pose_landmark_lite.onnx"
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

pub struct FullNetwork;

impl LandmarkNetwork for FullNetwork {
    fn cnn() -> &'static Cnn {
        const MODEL_DATA: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/3rdparty/onnx/pose_landmark_full.onnx"
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

// NB: There's also a "heavy" network, but it's >25 MB, so we don't support it. The full network
// should already perform pretty well.
