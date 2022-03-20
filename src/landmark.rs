//! Facial landmark detection.

use crate::{
    image::{AsImageView, ImageView},
    nn::{Cnn, CnnInputFormat, NeuralNetwork},
    resolution::Resolution,
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

    pub fn input_resolution(&self) -> Resolution {
        self.model.input_resolution()
    }

    pub fn timers(&self) -> impl IntoIterator<Item = &Timer> + '_ {
        [&self.t_resize, &self.t_infer]
    }

    pub fn compute<V: AsImageView>(&mut self, image: &V) -> &LandmarkResult {
        self.compute_impl(image.as_view())
    }

    fn compute_impl(&mut self, image: ImageView<'_>) -> &LandmarkResult {
        let input_res = self.model.input_resolution();
        let full_res = image.resolution();

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
            *res_x = x / input_res.width() as f32 * full_res.width() as f32;
            *res_y = y / input_res.height() as f32 * full_res.height() as f32;
            *res_z = z;
        }

        &self.result_buffer
    }
}

pub struct LandmarkResult {
    /// Landmarks scaled to fit the input image.
    landmarks: [(f32, f32, f32); 468],
    face_flag: f32,
}

impl LandmarkResult {
    pub fn landmarks(&self) -> &[(f32, f32, f32)] {
        &self.landmarks
    }

    pub fn face_confidence(&self) -> f32 {
        self.face_flag
    }
}
