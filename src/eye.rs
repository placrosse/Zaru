//! Eye and iris landmark computation.

use crate::{
    image::{AsImageView, ImageView},
    nn::{unadjust_aspect_ratio, Cnn, CnnInputFormat, NeuralNetwork},
    resolution::Resolution,
    timer::Timer,
};

const MODEL_PATH: &str = "onnx/iris_landmark.onnx";

pub struct EyeLandmarker {
    model: Cnn,
    t_resize: Timer,
    t_infer: Timer,
    result_buf: EyeLandmarks,
}

impl EyeLandmarker {
    pub fn new() -> Self {
        Self {
            model: Cnn::new(
                NeuralNetwork::load(MODEL_PATH).unwrap(),
                CnnInputFormat::NHWC,
            )
            .unwrap(),
            t_resize: Timer::new("resize"),
            t_infer: Timer::new("infer"),
            result_buf: EyeLandmarks {
                eye_contour: [(0.0, 0.0); 71],
                iris_contour: [(0.0, 0.0); 5],
                full_res: Resolution::new(1, 1),
            },
        }
    }

    /// Returns the expected input resolution of the internal neural network.
    pub fn input_resolution(&self) -> Resolution {
        self.model.input_resolution()
    }

    /// Computes eye landmarks on a cropped image of a left eye.
    ///
    /// Landmarks of a right eye can be computed by flipping both the image and the returned
    /// landmarks.
    pub fn compute<V: AsImageView>(&mut self, image: &V) -> &mut EyeLandmarks {
        self.compute_impl(image.as_view())
    }

    fn compute_impl(&mut self, image: ImageView<'_>) -> &mut EyeLandmarks {
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

        self.result_buf.full_res = full_res;
        for (coords, (out_x, out_y)) in result[0]
            .as_slice::<f32>()
            .unwrap()
            .chunks(2)
            .zip(&mut self.result_buf.eye_contour)
        {
            let (x, y) = (coords[0], coords[1]);
            let x = x / input_res.width() as f32;
            let y = y / input_res.height() as f32;
            let (x, y) = unadjust_aspect_ratio(x, y, orig_aspect);
            *out_x = x * full_res.width() as f32;
            *out_y = y * full_res.height() as f32;
        }

        for (coords, (out_x, out_y)) in result[1]
            .as_slice::<f32>()
            .unwrap()
            .chunks(2)
            .zip(&mut self.result_buf.iris_contour)
        {
            let (x, y) = (coords[0], coords[1]);
            let x = x / input_res.width() as f32;
            let y = y / input_res.height() as f32;
            let (x, y) = unadjust_aspect_ratio(x, y, orig_aspect);
            *out_x = x * full_res.width() as f32;
            *out_y = y * full_res.height() as f32;
        }

        &mut self.result_buf
    }

    /// Returns profiling timers for image resizing and neural inference.
    pub fn timers(&self) -> impl IntoIterator<Item = &Timer> + '_ {
        [&self.t_resize, &self.t_infer]
    }
}

/// Computed landmarks of an eye and its iris.
///
/// All coordinates use the coordinate system of the input image passed to
/// [`EyeLandmarker::compute`].
pub struct EyeLandmarks {
    eye_contour: [(f32, f32); 71],
    iris_contour: [(f32, f32); 5],
    full_res: Resolution,
}

impl EyeLandmarks {
    pub fn eye_contour(&self) -> &[(f32, f32)] {
        &self.eye_contour
    }

    pub fn iris_contour(&self) -> &[(f32, f32)] {
        // FIXME replace with center and radius/diameter
        &self.iris_contour
    }

    /// Flips all landmark coordinates along the X axis.
    pub fn flip_horizontal_in_place(&mut self) {
        let half_width = self.full_res.width() as f32 / 2.0;
        for (x, _) in &mut self.eye_contour {
            *x = -(*x - half_width) + half_width;
        }
        for (x, _) in &mut self.iris_contour {
            *x = -(*x - half_width) + half_width;
        }
    }
}
