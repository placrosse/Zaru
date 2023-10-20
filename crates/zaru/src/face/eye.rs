//! Eye and iris landmark computation.
//!
//! This uses the neural network from MediaPipe's [Iris] pipeline.
//!
//! [Iris]: https://google.github.io/mediapipe/solutions/iris

// TODO: this is rather useless with the FaceMeshV2 model, remove it

use std::sync::OnceLock;

use include_blob::include_blob;
use zaru_linalg::Vec3f;

use crate::image::{draw, AsImageViewMut, Color, ImageViewMut, Resolution};
use crate::iter::zip_exact;

use crate::nn::ColorMapper;
use crate::{
    landmark::{Estimate, Landmarks, Network},
    nn::{Cnn, CnnInputShape, NeuralNetwork, Outputs},
    slice::SliceExt,
};

/// A [`Network`] that computes eye landmarks on a cropped image of a left eye.
///
/// Landmarks of a right eye can be computed by flipping both the input image and the returned
/// landmarks with [`EyeLandmarks::flip_horizontal_in_place`].
#[derive(Clone, Copy)]
pub struct EyeNetwork;

impl Network for EyeNetwork {
    type Output = EyeLandmarks;

    fn cnn(&self) -> &Cnn {
        static MODEL: OnceLock<Cnn> = OnceLock::new();
        MODEL.get_or_init(|| {
            let model_data = include_blob!("../../3rdparty/onnx/iris_landmark.onnx");
            Cnn::new(
                NeuralNetwork::from_onnx(model_data).load().unwrap(),
                CnnInputShape::NCHW,
                ColorMapper::linear(-1.0..=1.0),
            )
            .unwrap()
        })
    }

    fn extract(&self, outputs: &Outputs, estimate: &mut Self::Output) {
        let eye_contour = &outputs[0];
        let iris_contour = &outputs[1];

        for (&[x, y, z], out) in zip_exact(
            eye_contour.index([0]).as_slice().array_chunks_exact::<3>(), // x, y, and z coordinates
            estimate.landmarks.positions_mut()[5..].iter_mut(),
        ) {
            *out = [x, y, z].into();
        }

        for (&[x, y, z], out) in zip_exact(
            iris_contour.index([0]).as_slice().array_chunks_exact::<3>(), // x, y, and z coordinates
            estimate.landmarks.positions_mut()[..5].iter_mut(),
        ) {
            *out = [x, y, z].into();
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
    pub fn iris_center(&self) -> Vec3f {
        self.landmarks.positions()[0]
    }

    /// Returns the outer landmarks of the iris.
    pub fn iris_contour(&self) -> impl Iterator<Item = Vec3f> + '_ {
        self.landmarks.positions()[1..=4].iter().copied()
    }

    /// Computes the iris diameter from the landmarks.
    pub fn iris_diameter(&self) -> f32 {
        let center = self.iris_center();

        // Average data from all landmarks.
        let mut radius = 0.0;
        for p in self.iris_contour() {
            radius += (center - p).length();
        }
        radius / self.iris_contour().count() as f32 * 2.0
    }

    pub fn eye_contour(&self) -> impl Iterator<Item = Vec3f> + '_ {
        self.landmarks.positions()[5..].iter().copied()
    }

    /// Flips all landmark coordinates along the X axis.
    pub fn flip_horizontal_in_place(&mut self, full_res: Resolution) {
        let half_width = full_res.width() as f32 / 2.0;
        self.landmarks
            .map_positions(|p| [-(p.x - half_width) + half_width, p.y, p.z].into());
    }

    /// Draws the eye landmarks onto an image.
    pub fn draw<I: AsImageViewMut>(&self, image: &mut I) {
        self.draw_impl(image.as_view_mut());
    }

    fn draw_impl(&self, mut image: ImageViewMut<'_>) {
        draw::marker(&mut image, self.iris_center().truncate())
            .size(3)
            .color(Color::CYAN);

        for p in self.eye_contour().take(16) {
            draw::marker(&mut image, p.truncate())
                .size(1)
                .color(Color::MAGENTA);
        }
        for p in self.eye_contour().skip(16) {
            draw::marker(&mut image, p.truncate())
                .size(1)
                .color(Color::GREEN);
        }
    }
}

impl Estimate for EyeLandmarks {
    fn landmarks_mut(&mut self) -> &mut Landmarks {
        &mut self.landmarks
    }
}
