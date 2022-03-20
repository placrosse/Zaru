//! Facial landmark detection.

use crate::{
    image::{AsImageView, ImageView},
    nn::{Cnn, CnnInputFormat, NeuralNetwork},
    timer::Timer,
};

const MODEL_PATH: &str = "models/face_landmark.onnx";

pub struct Landmarker {
    model: Cnn,
    t_resize: Timer,
    t_infer: Timer,
    /// Large, so keep one around and return by ref.
    result_buffer: LandmarkResult,
}

impl Landmarker {
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

    pub fn detect<V: AsImageView>(&mut self, image: &V) -> &LandmarkResult {
        self.detect_impl(image.as_view())
    }

    fn detect_impl(&mut self, image: ImageView<'_>) -> &LandmarkResult {
        let mut image = image.reborrow();
        let resized;
        if image.resolution() != self.model.input_resolution() {
            resized = self
                .t_resize
                .time(|| image.aspect_aware_resize(self.model.input_resolution()));
            image = resized.as_view();
        }
        let result = self.t_infer.time(|| self.model.infer(&image)).unwrap();
        log::trace!("inference result: {:?}", result);

        todo!()
    }
}

pub struct LandmarkResult {
    landmarks: [(f32, f32, f32); 468],
    face_flag: f32,
}
