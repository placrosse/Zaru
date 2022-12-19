//! Eye and iris landmark computation.
//!
//! This uses the neural network from MediaPipe's [Iris] pipeline.
//!
//! [Iris]: https://google.github.io/mediapipe/solutions/iris

use nalgebra::Point2;
use once_cell::sync::Lazy;

use zaru_image::{draw, AsImageViewMut, Color, ImageViewMut, Resolution};
use zaru_utils::iter::zip_exact;

use crate::{
    landmark::{Estimation, Landmarks, Network},
    nn::{create_linear_color_mapper, Cnn, CnnInputShape, NeuralNetwork, Outputs},
    slice::SliceExt,
};

const MODEL_DATA: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../3rdparty/onnx/iris_landmark.onnx"
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

/// A [`Network`] that computes eye landmarks on a cropped image of a left eye.
///
/// Landmarks of a right eye can be computed by flipping both the image and the returned
/// landmarks (via [`ImageViewMut::flip_horizontal_in_place`] and
/// [`EyeLandmarks::flip_horizontal_in_place`], respectively).
#[derive(Clone, Copy)]
pub struct EyeNetwork;

impl Network for EyeNetwork {
    type Output = EyeLandmarks;

    fn cnn(&self) -> &Cnn {
        &MODEL
    }

    fn extract(&self, outputs: &Outputs, estimation: &mut Self::Output) {
        let eye_contour = &outputs[0];
        let iris_contour = &outputs[1];

        for (&[x, y, z], [out_x, out_y, out_z]) in zip_exact(
            eye_contour.index([0]).as_slice().array_chunks_exact::<3>(), // x, y, and z coordinates
            estimation.landmarks.positions_mut()[5..].iter_mut(),
        ) {
            *out_x = x;
            *out_y = y;
            *out_z = z;
        }

        for (&[x, y, z], [out_x, out_y, out_z]) in zip_exact(
            iris_contour.index([0]).as_slice().array_chunks_exact::<3>(), // x, y, and z coordinates
            estimation.landmarks.positions_mut()[..5].iter_mut(),
        ) {
            *out_x = x;
            *out_y = y;
            *out_z = z;
        }
    }
}

/// Landmarks of an eye and its iris, estimated by [`EyeNetwork`].
#[derive(Clone)]
pub struct EyeLandmarks {
    landmarks: Landmarks,
}

impl Default for EyeLandmarks {
    fn default() -> Self {
        Self {
            landmarks: Landmarks::new(Self::NUM_LANDMARKS),
        }
    }
}

impl EyeLandmarks {
    pub const NUM_LANDMARKS: usize = 76;

    #[inline]
    pub fn landmarks(&self) -> &Landmarks {
        &self.landmarks
    }

    #[inline]
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
    pub fn flip_horizontal_in_place(&mut self, full_res: Resolution) {
        let half_width = full_res.width() as f32 / 2.0;
        self.landmarks
            .map_positions(|[x, y, z]| [-(x - half_width) + half_width, y, z]);
    }

    /// Draws the eye landmarks onto an image.
    pub fn draw<I: AsImageViewMut>(&self, image: &mut I) {
        self.draw_impl(image.as_view_mut());
    }

    fn draw_impl(&self, mut image: ImageViewMut<'_>) {
        let [x, y, _] = self.iris_center();
        draw::marker(&mut image, x as _, y as _)
            .size(3)
            .color(Color::CYAN);
        draw::circle(&mut image, x as _, y as _, self.iris_diameter() as u32).color(Color::CYAN);

        for [x, y, _] in self.eye_contour().take(16) {
            draw::marker(&mut image, x as _, y as _)
                .size(1)
                .color(Color::MAGENTA);
        }
        for [x, y, _] in self.eye_contour().skip(16) {
            draw::marker(&mut image, x as _, y as _)
                .size(1)
                .color(Color::GREEN);
        }
    }
}

impl Estimation for EyeLandmarks {
    fn landmarks_mut(&mut self) -> &mut Landmarks {
        &mut self.landmarks
    }
}
