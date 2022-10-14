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
    landmark::Landmarks,
    nn::{create_linear_color_mapper, unadjust_aspect_ratio, Cnn, CnnInputShape, NeuralNetwork},
    resolution::Resolution,
    timer::Timer,
};

const MODEL_DATA: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/3rdparty/onnx/iris_landmark.onnx"
));

static MODEL: Lazy<Cnn> = Lazy::new(|| {
    Cnn::new(
        NeuralNetwork::from_onnx(MODEL_DATA)
            .unwrap()
            .load()
            .unwrap(),
        CnnInputShape::NCHW,
        create_linear_color_mapper(-1.0..=1.0),
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
                landmarks: Landmarks::new(76),
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
        let orig_aspect = full_res.aspect_ratio().unwrap();

        let mut image = image;
        let resized;
        if image.resolution() != input_res {
            resized = self.t_resize.time(|| image.aspect_aware_resize(input_res));
            image = resized.as_view();
        }
        let result = self.t_infer.time(|| self.model.estimate(&image)).unwrap();
        log::trace!("inference result: {:?}", result);
        let eye_contour = &result[0];
        let iris_contour = &result[1];

        self.result_buf.full_res = full_res;
        for (coords, [out_x, out_y, out_z]) in zip_exact(
            eye_contour.index([0]).as_slice().chunks(3), // x, y, and z coordinates
            self.result_buf.landmarks.positions_mut()[5..].iter_mut(),
        ) {
            let (x, y) = (coords[0], coords[1]);
            let x = x / input_res.width() as f32;
            let y = y / input_res.height() as f32;
            let (x, y) = unadjust_aspect_ratio(x, y, orig_aspect);
            *out_x = x * full_res.width() as f32;
            *out_y = y * full_res.height() as f32;
            *out_z = coords[2];
        }

        for (coords, [out_x, out_y, out_z]) in zip_exact(
            iris_contour.index([0]).as_slice().chunks(3), // x, y, and z coordinates
            self.result_buf.landmarks.positions_mut()[..5].iter_mut(),
        ) {
            let (x, y) = (coords[0], coords[1]);
            let x = x / input_res.width() as f32;
            let y = y / input_res.height() as f32;
            let (x, y) = unadjust_aspect_ratio(x, y, orig_aspect);
            *out_x = x * full_res.width() as f32;
            *out_y = y * full_res.height() as f32;
            *out_z = coords[2];
        }

        &mut self.result_buf
    }

    /// Returns profiling timers for image resizing and neural inference.
    pub fn timers(&self) -> impl Iterator<Item = &Timer> + '_ {
        [&self.t_resize, &self.t_infer].into_iter()
    }
}

/// Computed landmarks of an eye and its iris.
///
/// All coordinates use the coordinate system of the input image passed to
/// [`EyeLandmarker::compute`].
#[derive(Clone)]
pub struct EyeLandmarks {
    landmarks: Landmarks,
    full_res: Resolution,
}

impl EyeLandmarks {
    pub fn landmarks(&self) -> &Landmarks {
        &self.landmarks
    }

    pub fn landmarks_mut(&mut self) -> &mut Landmarks {
        &mut self.landmarks
    }

    /// Returns the center coordinates of the iris.
    pub fn iris_center(&self) -> [f32; 3] {
        self.landmarks.positions()[0]
    }

    /// Returns the outer landmarks of the iris.
    pub fn iris_contour(&self) -> impl Iterator<Item = [f32; 3]> + '_ {
        self.landmarks.positions()[1..=4].iter().copied()
    }

    /// Computes the iris diameter from the landmarks.
    pub fn iris_diameter(&self) -> f32 {
        let [cx, cy, _] = self.iris_center();
        let center = Point2::new(cx, cy);

        // Average data from all landmarks.
        let mut acc_radius = 0.0;
        for [x, y, _] in self.iris_contour() {
            acc_radius += nalgebra::distance(&center, &Point2::new(x, y));
        }
        let diameter = acc_radius / self.iris_contour().count() as f32 * 2.0;
        diameter
    }

    pub fn eye_contour(&self) -> impl Iterator<Item = [f32; 3]> + '_ {
        self.landmarks.positions()[5..].iter().copied()
    }

    /// Flips all landmark coordinates along the X axis.
    pub fn flip_horizontal_in_place(&mut self) {
        let half_width = self.full_res.width() as f32 / 2.0;
        self.landmarks
            .map_positions(|[x, y, z]| [-(x - half_width) + half_width, y, z]);
    }

    /// Draws the eye landmarks onto an image.
    pub fn draw<I: AsImageViewMut>(&self, image: &mut I) {
        self.draw_impl(image.as_view_mut());
    }

    fn draw_impl(&self, mut image: ImageViewMut<'_>) {
        let [x, y, _] = self.iris_center();
        image::draw_marker(&mut image, x as _, y as _)
            .size(3)
            .color(Color::CYAN);
        image::draw_circle(&mut image, x as _, y as _, self.iris_diameter() as u32)
            .color(Color::CYAN);

        for [x, y, _] in self.eye_contour().take(16) {
            image::draw_marker(&mut image, x as _, y as _)
                .size(1)
                .color(Color::MAGENTA);
        }
        for [x, y, _] in self.eye_contour().skip(16) {
            image::draw_marker(&mut image, x as _, y as _)
                .size(1)
                .color(Color::GREEN);
        }
    }
}
