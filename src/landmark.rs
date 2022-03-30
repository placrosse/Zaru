//! Facial landmark detection.
//!
//! Also known as *face alignment* or *registration*.
//!
//! This uses one of the neural networks also used in MediaPipe's [Face Mesh] pipeline.
//!
//! [Face Mesh]: https://google.github.io/mediapipe/solutions/face_mesh.html

mod reference_data;

use std::ops::Index;

use crate::{
    image::{AsImageView, ImageView},
    iter::zip_exact,
    nn::{unadjust_aspect_ratio, Cnn, CnnInputFormat, NeuralNetwork},
    resolution::{AspectRatio, Resolution},
    timer::Timer,
};

const MODEL_PATH: &str = "onnx/face_landmark.onnx";

/// A neural network based facial landmark predictor.
pub struct Landmarker {
    model: Cnn,
    t_resize: Timer,
    t_infer: Timer,
    /// Large, so keep one around and return by ref.
    result_buffer: LandmarkResult,
}

impl Landmarker {
    /// Creates a new facial landmark calculator.
    pub fn new() -> Self {
        let model = Cnn::new(
            NeuralNetwork::load(MODEL_PATH).unwrap(),
            CnnInputFormat::NHWC,
        )
        .unwrap();
        Self {
            t_resize: Timer::new("resize"),
            t_infer: Timer::new("infer"),
            result_buffer: LandmarkResult {
                landmarks: Landmarks {
                    positions: [Pos(0.0, 0.0, 0.0); 468],
                },
                face_flag: 0.0,
                orig_res: Resolution::new(1, 1),
                orig_aspect: AspectRatio::SQUARE,
                input_res: model.input_resolution(),
            },
            model,
        }
    }

    /// Returns the expected input resolution of the internal neural network.
    pub fn input_resolution(&self) -> Resolution {
        self.model.input_resolution()
    }

    /// Computes facial landmarks in `image`.
    ///
    /// `image` must be a cropped image of a face. When using [`crate::detector::Detector`], the
    /// rectangle returned by [`Detection::bounding_rect_loose`] produces good results.
    ///
    /// The image should depict a face that is mostly upright. Results will be poor if the face is
    /// rotated too much.
    ///
    /// [`Detection::bounding_rect_loose`]: crate::detector::Detection::bounding_rect_loose
    pub fn compute<V: AsImageView>(&mut self, image: &V) -> &LandmarkResult {
        self.compute_impl(image.as_view())
    }

    fn compute_impl(&mut self, image: ImageView<'_>) -> &LandmarkResult {
        let input_res = self.model.input_resolution();
        let full_res = image.resolution();
        self.result_buffer.orig_res = full_res;
        self.result_buffer.orig_aspect = full_res.aspect_ratio();

        let mut image = image.reborrow();
        let resized;
        if image.resolution() != input_res {
            resized = self.t_resize.time(|| image.aspect_aware_resize(input_res));
            image = resized.as_view();
        }
        let result = self.t_infer.time(|| self.model.infer(&image)).unwrap();
        log::trace!("inference result: {:?}", result);

        self.result_buffer.face_flag = result[1].as_slice::<f32>().unwrap()[0];
        for (coords, out) in zip_exact(
            result[0].as_slice::<f32>().unwrap().chunks(3),
            &mut self.result_buffer.landmarks.positions,
        ) {
            out.0 = coords[0];
            out.1 = coords[1];
            out.2 = coords[2];
        }

        &self.result_buffer
    }

    /// Returns profiling timers for image resizing and neural inference.
    pub fn timers(&self) -> impl IntoIterator<Item = &Timer> + '_ {
        [&self.t_resize, &self.t_infer]
    }
}

/// Landmark position in 3D space.
#[derive(Debug, Clone, Copy)]
pub struct Pos(f32, f32, f32);

impl Pos {
    #[inline]
    pub fn x(&self) -> f32 {
        self.0
    }

    #[inline]
    pub fn y(&self) -> f32 {
        self.1
    }

    #[inline]
    pub fn z(&self) -> f32 {
        self.2
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

impl LandmarkResult {
    /// Returns the 3D landmark positions in the input image's coordinate system.
    pub fn landmark_positions(&self) -> impl Iterator<Item = Pos> + '_ {
        (0..self.landmark_count()).map(|index| self.landmark_position(index))
    }

    /// Returns a landmark's position in the input image's coordinate system.
    pub fn landmark_position(&self, index: usize) -> Pos {
        let Pos(x, y, z) = self.landmarks.positions[index];
        let (x, y) = unadjust_aspect_ratio(
            x / self.input_res.width() as f32,
            y / self.input_res.height() as f32,
            self.orig_aspect,
        );
        let (x, y) = (
            x * self.orig_res.width() as f32,
            y * self.orig_res.height() as f32,
        );
        Pos(x, y, z)
    }

    #[inline]
    pub fn landmark_count(&self) -> usize {
        self.landmarks.positions.len()
    }

    #[inline]
    pub fn raw_landmarks(&self) -> &Landmarks {
        &self.landmarks
    }

    /// Returns the confidence that the input image contains a proper face.
    ///
    /// This can be used to estimate the fit quality, or to re-run face detection if that isn't done
    /// each frame. Typical values are >20.0 when a good landmark fit is produced, between 10 and 20
    /// when the face is rotated a bit too far, and <10 when the face is rotated much too far or
    /// there is no face in the input image.
    #[inline]
    pub fn face_confidence(&self) -> f32 {
        self.face_flag
    }
}

/// Raw face landmark positions.
#[derive(Clone)]
pub struct Landmarks {
    positions: [Pos; 468],
}

impl Landmarks {
    /// Landmark positions from the reference/canonical face model.
    pub const REFERENCE_LANDMARKS: &'static Self = &Self {
        positions: reference_data::POSITIONS,
    };
}

impl Index<Idx> for Landmarks {
    type Output = Pos;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.positions[index as usize]
    }
}

impl Index<usize> for Landmarks {
    type Output = Pos;

    fn index(&self, index: usize) -> &Self::Output {
        &self.positions[index]
    }
}

/// Assigns a name to certain important landmark indices.
///
/// "Left" and "Right" are relative to the input image, not from the PoV of the depicted person.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Idx {
    MouthLeft = 78,
    MouthRight = 308,
    MouthTop = 13,
    MouthBottom = 14,
    LeftEyeLeftCorner = 33,
    LeftEyeRightCorner = 133,
    LeftEyeTop = 159,
    LeftEyeBottom = 145,
    RightEyeLeftCorner = 362,
    RightEyeRightCorner = 263,
    RightEyeTop = 386,
    RightEyeBottom = 374,
    RightEyebrowLeftCorner = 295,
    LeftEyebrowRightCorner = 65,
    // idle thought: you could use this data to render an AR unibrow :)
}

impl Into<usize> for Idx {
    #[inline]
    fn into(self) -> usize {
        self as usize
    }
}
