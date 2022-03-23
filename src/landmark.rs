//! Facial landmark detection.
//!
//! Also known as *face alignment* or *registration*.
//!
//! This uses one of the neural networks also used in MediaPipe's [Face Mesh] pipeline.
//!
//! [Face Mesh]: https://google.github.io/mediapipe/solutions/face_mesh.html

use crate::{
    image::{AsImageView, ImageView},
    nn::{unadjust_aspect_ratio, Cnn, CnnInputFormat, NeuralNetwork},
    resolution::Resolution,
    timer::Timer,
};

const MODEL_PATH: &str = "models/face_landmark.onnx";

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
        Self {
            model: Cnn::new(
                NeuralNetwork::load(MODEL_PATH).unwrap(),
                CnnInputFormat::NHWC,
            )
            .unwrap(),
            t_resize: Timer::new("resize"),
            t_infer: Timer::new("infer"),
            result_buffer: LandmarkResult {
                landmarks: [(0.0, 0.0, 0.0); 468],
                face_flag: 0.0,
            },
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
    /// [`Detection::bounding_rect_loose`]: crate::detector::Detection::bounding_rect_loose
    pub fn compute<V: AsImageView>(&mut self, image: &V) -> &LandmarkResult {
        self.compute_impl(image.as_view())
    }

    fn compute_impl(&mut self, image: ImageView<'_>) -> &LandmarkResult {
        let input_res = self.model.input_resolution();
        let full_res = image.resolution();
        let orig_aspect = full_res.aspect_ratio();

        let mut image = image.reborrow();
        let resized;
        if image.resolution() != input_res {
            resized = self.t_resize.time(|| image.aspect_aware_resize(input_res));
            image = resized.as_view();
        }
        let result = self.t_infer.time(|| self.model.infer(&image)).unwrap();
        log::trace!("inference result: {:?}", result);

        self.result_buffer.face_flag = result[1].as_slice::<f32>().unwrap()[0];
        for (coords, (res_x, res_y, res_z)) in result[0]
            .as_slice::<f32>()
            .unwrap()
            .chunks(3)
            .zip(&mut self.result_buffer.landmarks)
        {
            let (x, y, z) = (coords[0], coords[1], coords[2]);
            let x = x / input_res.width() as f32;
            let y = y / input_res.height() as f32;
            let (x, y) = unadjust_aspect_ratio(x, y, orig_aspect);
            *res_x = x * full_res.width() as f32;
            *res_y = y * full_res.height() as f32;
            *res_z = z;
        }

        &self.result_buffer
    }

    /// Returns profiling timers for image resizing and neural inference.
    pub fn timers(&self) -> impl IntoIterator<Item = &Timer> + '_ {
        [&self.t_resize, &self.t_infer]
    }
}

/// Landmark results returned by [`Landmarker::compute`].
pub struct LandmarkResult {
    /// Landmarks scaled to fit the input image.
    landmarks: [(f32, f32, f32); 468],
    face_flag: f32,
}

impl LandmarkResult {
    /// Returns the 3D landmarks fitted to the face.
    ///
    /// X and Y coordinates correspond to the input image's coordinate system, Z coordinates are
    /// raw model output.
    pub fn landmarks(&self) -> &[(f32, f32, f32)] {
        &self.landmarks
    }

    /// Returns the confidence that the input image contains a proper face.
    ///
    /// This can be used to estimate the fit quality, or to re-run face detection if that isn't done
    /// each frame. Typical values are >20.0 when a good landmark fit is produced, between 10 and 20
    /// when the face is rotated a bit too far, and <10 when the face is rotated much too far or
    /// there is no face in the input image.
    pub fn face_confidence(&self) -> f32 {
        self.face_flag
    }
}
