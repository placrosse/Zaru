//! Eye and iris landmark computation.
//!
//! This uses the neural network from MediaPipe's [Iris] pipeline.
//!
//! [Iris]: https://google.github.io/mediapipe/solutions/iris

use nalgebra::Point2;
use once_cell::sync::Lazy;

use crate::{
    image::{self, AsImageView, AsImageViewMut, Color, ImageView, ImageViewMut},
    iter::zip_exact,
    nn::{unadjust_aspect_ratio, Cnn, CnnInputShape, NeuralNetwork},
    resolution::Resolution,
    timer::Timer,
};

const MODEL_DATA: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/3rdparty/onnx/iris_landmark.onnx"
));

static MODEL: Lazy<Cnn> = Lazy::new(|| {
    Cnn::new(
        NeuralNetwork::from_onnx(MODEL_DATA).unwrap(),
        CnnInputShape::NHWC,
    )
    .unwrap()
});

/// An eye and iris landmark predictor.
pub struct EyeLandmarker {
    model: &'static Cnn,
    t_resize: Timer,
    t_infer: Timer,
    result_buf: EyeLandmarks,
}

impl EyeLandmarker {
    /// Creates a new eye landmarker.
    pub fn new() -> Self {
        Self {
            model: &MODEL,
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
    /// landmarks (via [`ImageViewMut::flip_horizontal_in_place`] and
    /// [`EyeLandmarks::flip_horizontal_in_place`], respectively).
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
        let result = self.t_infer.time(|| self.model.estimate(&image)).unwrap();
        log::trace!("inference result: {:?}", result);

        self.result_buf.full_res = full_res;
        for (coords, (out_x, out_y)) in zip_exact(
            result[0].as_slice::<f32>().unwrap().chunks(3), // x, y, and z coordinates
            &mut self.result_buf.eye_contour,
        ) {
            let (x, y) = (coords[0], coords[1]);
            let x = x / input_res.width() as f32;
            let y = y / input_res.height() as f32;
            let (x, y) = unadjust_aspect_ratio(x, y, orig_aspect);
            *out_x = x * full_res.width() as f32;
            *out_y = y * full_res.height() as f32;
        }

        for (coords, (out_x, out_y)) in zip_exact(
            result[1].as_slice::<f32>().unwrap().chunks(3), // x, y, and z coordinates
            &mut self.result_buf.iris_contour,
        ) {
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
#[derive(Clone)]
pub struct EyeLandmarks {
    eye_contour: [(f32, f32); 71],
    iris_contour: [(f32, f32); 5],
    full_res: Resolution,
}

impl EyeLandmarks {
    /// Returns the center coordinates of the iris.
    pub fn iris_center(&self) -> (f32, f32) {
        self.iris_contour[0]
    }

    /// Returns the outer landmarks of the iris.
    pub fn iris_contour(&self) -> &[(f32, f32)] {
        &self.iris_contour[1..]
    }

    /// Computes the iris diameter from the landmarks.
    pub fn iris_diameter(&self) -> f32 {
        let (cx, cy) = self.iris_center();
        let center = Point2::new(cx, cy);

        // Average data from all landmarks.
        let mut acc_radius = 0.0;
        for (x, y) in self.iris_contour() {
            acc_radius += nalgebra::distance(&center, &Point2::new(*x, *y));
        }
        let diameter = acc_radius / self.iris_contour().len() as f32 * 2.0;
        diameter
    }

    pub fn eye_contour(&self) -> &[(f32, f32)] {
        &self.eye_contour
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

    /// Draws the eye landmarks onto an image.
    pub fn draw<I: AsImageViewMut>(&self, image: &mut I) {
        self.draw_impl(image.as_view_mut());
    }

    fn draw_impl(&self, mut image: ImageViewMut<'_>) {
        assert_eq!(image.resolution(), self.full_res);
        for (x, y) in self.eye_contour().iter().take(16) {
            image::draw_marker(&mut image, *x as _, *y as _)
                .size(1)
                .color(Color::MAGENTA);
        }
        for (x, y) in self.eye_contour().iter().skip(16) {
            image::draw_marker(&mut image, *x as _, *y as _)
                .size(1)
                .color(Color::GREEN);
        }

        let (x, y) = self.iris_center();
        image::draw_circle(&mut image, x as _, y as _, self.iris_diameter() as u32)
            .color(Color::RED);
    }
}
