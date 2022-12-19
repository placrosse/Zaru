//! Body pose landmark prediction. Not yet fully implemented.

// TODO: port to the new `landmark` module

use once_cell::sync::Lazy;
use zaru_image::{
    draw, AsImageView, AsImageViewMut, AspectRatio, Color, ImageView, ImageViewMut, Resolution,
};
use zaru_utils::{iter::zip_exact, num::sigmoid};

use crate::{
    nn::{create_linear_color_mapper, unadjust_aspect_ratio, Cnn, CnnInputShape, NeuralNetwork},
    slice::SliceExt,
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
                landmarks: [Landmark {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    presence: 0.0,
                    visibility: 0.0,
                }; 39],
                pose_presence: 0.0,
                orig_res: Resolution::new(1, 1),
                orig_aspect: AspectRatio::SQUARE,
            },
        }
    }

    /// Returns the expected input resolution of the internal neural network.
    pub fn input_resolution(&self) -> Resolution {
        self.cnn.input_resolution()
    }

    /// Computes body landmarks in `image`.
    pub fn compute<V: AsImageView>(&mut self, image: &V) -> &LandmarkResult {
        self.compute_impl(image.as_view())
    }

    fn compute_impl(&mut self, image: ImageView<'_>) -> &LandmarkResult {
        let input_res = self.input_resolution();
        let full_res = image.resolution();
        let orig_aspect = full_res.aspect_ratio().unwrap();
        self.result_buffer.orig_res = full_res;
        self.result_buffer.orig_aspect = orig_aspect;

        let mut image = image;
        let resized;
        if image.resolution() != input_res {
            resized = self.t_resize.time(|| image.aspect_aware_resize(input_res));
            image = resized.as_view();
        }
        let outputs = self.t_infer.time(|| self.cnn.estimate(&image)).unwrap();
        log::trace!("cnn outputs: {:?}", outputs);

        let screen_landmarks = &outputs[0];
        let pose_flag = &outputs[1];
        let segmentation = &outputs[2];
        let heatmap = &outputs[3];
        let world_landmarks = &outputs[4];

        // 33 pose landmarks (`LandmarkIdx`), 6 auxiliary landmarks
        assert_eq!(screen_landmarks.shape(), &[1, 195]); // 39 landmarks * 5 values
        assert_eq!(pose_flag.shape(), &[1, 1]);
        assert_eq!(segmentation.shape(), &[1, 256, 256, 1]);
        assert_eq!(heatmap.shape(), &[1, 64, 64, 39]);
        assert_eq!(world_landmarks.shape(), &[1, 117]); // 39 landmarks * 3 values

        self.result_buffer.pose_presence = pose_flag.index([0, 0]).as_singular();

        for (&[x, y, z, visibility, presence], out) in zip_exact(
            screen_landmarks
                .index([0])
                .as_slice()
                .array_chunks_exact::<5>(),
            &mut self.result_buffer.landmarks,
        ) {
            let (x, y) = unadjust_aspect_ratio(
                x / input_res.width() as f32,
                y / input_res.height() as f32,
                orig_aspect,
            );
            let (x, y) = (x * full_res.width() as f32, y * full_res.height() as f32);

            out.x = x;
            out.y = y;
            out.z = z;
            out.visibility = visibility;
            out.presence = presence;
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
    orig_res: Resolution,
    orig_aspect: AspectRatio,

    pose_presence: f32,
    landmarks: [Landmark; 39],
}

impl LandmarkResult {
    pub fn pose_landmarks(&self) -> &[Landmark] {
        &self.landmarks[..33]
    }

    pub fn aux_landmarks(&self) -> &[Landmark] {
        &self.landmarks[33..]
    }

    pub fn presence(&self) -> f32 {
        self.pose_presence
    }

    pub fn draw<I: AsImageViewMut>(&self, target: &mut I) {
        self.draw_impl(&mut target.as_view_mut());
    }

    fn draw_impl(&self, target: &mut ImageViewMut<'_>) {
        for (a, b) in COARSE_CONNECTIVITY {
            let a = &self.landmarks[*a as usize];
            let b = &self.landmarks[*b as usize];
            draw::line(target, a.x() as _, a.y() as _, b.x() as _, b.y() as _).stroke_width(3);
        }

        for lm in self.pose_landmarks() {
            draw::marker(target, lm.x() as _, lm.y() as _).size(9);
        }
        for lm in self.aux_landmarks() {
            draw::marker(target, lm.x() as _, lm.y() as _).color(Color::YELLOW);
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

#[derive(Debug, Clone, Copy)]
pub struct Landmark {
    x: f32,
    y: f32,
    z: f32,
    visibility: f32,
    presence: f32,
}

impl Landmark {
    pub fn x(&self) -> f32 {
        self.x
    }

    pub fn y(&self) -> f32 {
        self.y
    }

    pub fn z(&self) -> f32 {
        self.z
    }

    pub fn visibility(&self) -> f32 {
        sigmoid(self.visibility)
    }

    pub fn presence(&self) -> f32 {
        sigmoid(self.presence)
    }
}

pub trait LandmarkNetwork {
    fn cnn() -> &'static Cnn;
}

pub struct LiteNetwork;

impl LandmarkNetwork for LiteNetwork {
    fn cnn() -> &'static Cnn {
        static MODEL: Lazy<Cnn> = Lazy::new(|| {
            let model_data = include_blob::include_bytes!("3rdparty/onnx/pose_landmark_lite.onnx");
            Cnn::new(
                NeuralNetwork::from_onnx(model_data)
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
        static MODEL: Lazy<Cnn> = Lazy::new(|| {
            let model_data = include_blob::include_bytes!("3rdparty/onnx/pose_landmark_full.onnx");
            Cnn::new(
                NeuralNetwork::from_onnx(model_data)
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
