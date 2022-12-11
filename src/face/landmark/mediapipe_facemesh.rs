//! A wrapper around MediaPipe's [Face Mesh] landmark predictor network.
//!
//! The [`reference_positions`] function returns the positions of the landmarks for a "reference
//! face" that faces the camera head-on.
//!
//! [Face Mesh]: https://google.github.io/mediapipe/solutions/face_mesh.html

// NOTE: MediaPipe also has a `face_landmarks_with_attention` network, which outputs more accurate
// eye and mouth landmarks. However, it uses custom ops, and so can't be converted to a
// non-TensorFlow format.

use itertools::Itertools;
use nalgebra::{Rotation2, Vector2};
use once_cell::sync::Lazy;

use crate::{
    image::{self, AsImageViewMut, Color, ImageViewMut, RotatedRect},
    iter::zip_exact,
    landmark::{self, Landmarks},
    nn::{create_linear_color_mapper, Cnn, CnnInputShape, NeuralNetwork, Outputs},
    num::{sigmoid, TotalF32},
};

const MODEL_DATA: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/3rdparty/onnx/face_landmark.onnx"
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

/// Estimates facial landmarks using the MediaPipe Face Mesh network.
///
/// The input image must be a cropped image of a face. When using [`Detector`], the
/// rectangle returned by [`Detection::bounding_rect_loose`] produces good results.
///
/// The image should depict a face that is mostly upright. Results will be poor if the face is
/// rotated too much. A [`LandmarkTracker`] can be used to automatically follow the rotation of a
/// face and pass an upright view to the network.
///
/// [`Detector`]: super::super::detection::Detector
/// [`Detection::bounding_rect_loose`]: super::super::detection::Detection::bounding_rect_loose
/// [`LandmarkTracker`]: crate::landmark::LandmarkTracker
pub struct MediaPipeFaceMesh;

impl landmark::Network for MediaPipeFaceMesh {
    type Output = LandmarkResult;

    fn cnn(&self) -> &Cnn {
        &MODEL
    }

    fn extract(&self, output: &Outputs, estimation: &mut Self::Output) {
        estimation.face_flag = sigmoid(output[1].index([0, 0, 0, 0]).as_singular());
        for (coords, out) in zip_exact(
            output[0].index([0, 0, 0]).as_slice().chunks(3),
            estimation.landmarks.positions_mut(),
        ) {
            let [x, y, z] = [coords[0], coords[1], coords[2]];
            out[0] = x;
            out[1] = y;
            out[2] = z;
        }
    }
}

/// Landmark results estimated by [`MediaPipeFaceMesh`].
#[derive(Clone)]
pub struct LandmarkResult {
    landmarks: Landmarks,
    face_flag: f32,
}

impl Default for LandmarkResult {
    fn default() -> Self {
        Self {
            landmarks: Landmarks::new(Self::NUM_LANDMARKS),
            face_flag: 0.0,
        }
    }
}

impl LandmarkResult {
    pub const NUM_LANDMARKS: usize = 468;

    /// Returns the 3D landmark positions in the input image's coordinate system.
    pub fn landmark_positions(&self) -> impl Iterator<Item = (f32, f32, f32)> + '_ {
        (0..self.landmark_count()).map(|index| self.landmark_position(index))
    }

    /// Returns a landmark's position in the input image's coordinate system.
    pub fn landmark_position(&self, index: usize) -> (f32, f32, f32) {
        let [x, y, z] = self.landmarks.landmark(index).position();
        (x, y, z)
    }

    #[inline]
    pub fn landmark_count(&self) -> usize {
        self.landmarks.len()
    }

    #[inline]
    pub fn landmarks(&self) -> &Landmarks {
        &self.landmarks
    }

    #[inline]
    pub fn landmarks_mut(&mut self) -> &mut Landmarks {
        &mut self.landmarks
    }

    /// Returns the confidence that the input image contains a proper face.
    ///
    /// The returned value is in range 0.0 to 1.0.
    ///
    /// This can be used to estimate the fit quality, or to re-run face detection if that isn't done
    /// each frame.
    #[inline]
    pub fn face_confidence(&self) -> f32 {
        self.face_flag
    }

    pub fn rotation_radians(&self) -> f32 {
        let left_eye = self.landmark_position(LandmarkIdx::LeftEyeOuterCorner as _);
        let right_eye = self.landmark_position(LandmarkIdx::RightEyeOuterCorner as _);
        let left_to_right_eye = Vector2::new(
            (right_eye.0 - left_eye.0) as f32,
            (right_eye.1 - left_eye.1) as f32,
        );
        Rotation2::rotation_between(&Vector2::x(), &left_to_right_eye).angle()
    }

    /// Returns a [`RotatedRect`] containing the left eye.
    pub fn left_eye(&self) -> RotatedRect {
        RotatedRect::bounding(
            self.rotation_radians(),
            [
                LandmarkIdx::LeftEyeBottom,
                LandmarkIdx::LeftEyeOuterCorner,
                LandmarkIdx::LeftEyeInnerCorner,
                LandmarkIdx::LeftEyeTop,
            ]
            .into_iter()
            .map(|idx| {
                let pos = self.landmark_position(idx as usize);
                (pos.0 as i32, pos.1 as i32)
            }),
        )
        .unwrap()
    }

    /// Returns a [`RotatedRect`] containing the right eye.
    pub fn right_eye(&self) -> RotatedRect {
        RotatedRect::bounding(
            self.rotation_radians(),
            [
                LandmarkIdx::RightEyeBottom,
                LandmarkIdx::RightEyeInnerCorner,
                LandmarkIdx::RightEyeOuterCorner,
                LandmarkIdx::RightEyeTop,
            ]
            .into_iter()
            .map(|idx| {
                let pos = self.landmark_position(idx as usize);
                (pos.0 as i32, pos.1 as i32)
            }),
        )
        .unwrap()
    }

    /// Draws the landmark result onto an image.
    ///
    /// # Panics
    ///
    /// The image must have the same resolution as the image the detection was performed on,
    /// otherwise this method will panic.
    pub fn draw<I: AsImageViewMut>(&self, image: &mut I) {
        self.draw_impl(&mut image.as_view_mut());
    }

    fn draw_impl(&self, image: &mut ImageViewMut<'_>) {
        for (x, y, _z) in self.landmark_positions() {
            image::draw_marker(image, x as _, y as _).size(3);
        }

        let color = match self.face_confidence() {
            0.75.. => Color::GREEN,
            0.5..=0.75 => Color::YELLOW,
            _ => Color::RED,
        };
        let (x_min, x_max) = self
            .landmark_positions()
            .map(|(x, _, _)| TotalF32(x))
            .minmax()
            .into_option()
            .unwrap();
        let x = (x_min.0 + x_max.0) / 2.0;
        let y = self
            .landmark_positions()
            .map(|(_, y, _)| TotalF32(y))
            .min()
            .unwrap()
            .0;
        image::draw_text(
            image,
            x as i32,
            y as i32 - 3,
            &format!("lm_conf={:.01}", self.face_confidence()),
        )
        .align_bottom()
        .color(color);

        let left_eye = self.landmark_position(LandmarkIdx::LeftEyeOuterCorner as _);
        let right_eye = self.landmark_position(LandmarkIdx::RightEyeOuterCorner as _);
        image::draw_line(
            image,
            left_eye.0 as _,
            left_eye.1 as _,
            right_eye.0 as _,
            right_eye.1 as _,
        )
        .color(Color::WHITE);
        image::draw_text(
            image,
            ((left_eye.0 + right_eye.0) / 2.0) as i32,
            ((left_eye.1 + right_eye.1) / 2.0) as i32,
            &format!("{:.1} deg", self.rotation_radians().to_degrees()),
        )
        .align_bottom()
        .color(Color::WHITE);
    }
}

impl landmark::Estimation for LandmarkResult {
    fn landmarks_mut(&mut self) -> &mut Landmarks {
        &mut self.landmarks
    }

    fn angle_radians(&self) -> Option<f32> {
        Some(self.rotation_radians())
    }
}

impl landmark::Confidence for LandmarkResult {
    fn confidence(&self) -> f32 {
        self.face_confidence()
    }
}

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/3rdparty/3d/canonical_face_model.rs"
));

/// Returns an iterator over the vertices of the reference face model.
///
/// Each point yielded by the returned iterator corresponds to the same point in the sequence
/// of landmarks in the [`LandmarkResult`], but the scale and coordinate system does not: The points
/// returned by this function have Y pointing up, and X and Y are in a smaller range around `(0,0)`,
/// while [`LandmarkResult`] contains points that have Y point down, and X and Y are in term of the
/// input image's coordinates.
pub fn reference_positions() -> impl Iterator<Item = (f32, f32, f32)> {
    REFERENCE_POSITIONS.iter().copied()
}

/// Assigns a name to certain important landmark indices.
///
/// "Left" and "Right" are relative to the input image, not from the PoV of the depicted person.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LandmarkIdx {
    MouthLeft = 78,
    MouthRight = 308,
    MouthTop = 13,
    MouthBottom = 14,
    LeftEyeOuterCorner = 33,
    LeftEyeInnerCorner = 133,
    LeftEyeTop = 159,
    LeftEyeBottom = 145,
    RightEyeInnerCorner = 362,
    RightEyeOuterCorner = 263,
    RightEyeTop = 386,
    RightEyeBottom = 374,
    RightEyebrowInnerCorner = 295,
    LeftEyebrowInnerCorner = 65,
}
// FIXME: these are swapped or otherwise messed up

impl Into<usize> for LandmarkIdx {
    #[inline]
    fn into(self) -> usize {
        self as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        image::{AsImageView, ImageView},
        landmark::Estimator,
        procrustes::ProcrustesAnalyzer,
        test,
    };

    #[track_caller]
    fn check_angle(expected_radians: f32, actual_radians: f32) {
        let expected_degrees = expected_radians.to_degrees();
        let actual_degrees = actual_radians.to_degrees();
        assert!(
            (actual_degrees - expected_degrees).abs() < 5.0,
            "expected angle: {}°, actual angle: {}°",
            expected_degrees,
            actual_degrees,
        );
    }

    fn check_landmarks(image: ImageView<'_>, degrees: f32) {
        let expected_radians = degrees.to_radians();

        let mut lm = Estimator::new(MediaPipeFaceMesh);
        let landmarks = lm.estimate(&image);
        assert!(landmarks.face_confidence() > 0.9);
        check_angle(expected_radians, landmarks.rotation_radians());
        check_angle(expected_radians, landmarks.left_eye().rotation_radians());
        check_angle(expected_radians, landmarks.right_eye().rotation_radians());

        if degrees.abs() <= 45.0f32 {
            assert!(landmarks.left_eye().center().0 < landmarks.right_eye().center().0);
        }

        let mut pa = ProcrustesAnalyzer::new(reference_positions());

        let res = pa.analyze(landmarks.landmarks().positions().iter().map(|&[x, y, z]| {
            // Flip Y to bring us to canonical 3D coordinates (where Y points up).
            // Only rotation matters, so we don't have to correct for the added
            // translation.
            (x, -y, z)
        }));
        let (roll, pitch, yaw) = res.rotation().euler_angles();
        check_angle(0.0, roll);
        check_angle(0.0, pitch);
        check_angle(-expected_radians, yaw); // we've flipped the Y coord, so this flips too
    }

    #[test]
    fn estimates_landmarks_upright() {
        check_landmarks(test::sad_linus_cropped().as_view(), 0.0);
    }

    #[test]
    fn estimates_landmarks_rotated() {
        let image = test::sad_linus_cropped();
        check_landmarks(
            image.view(RotatedRect::new(image.rect(), 10.0f32.to_radians())),
            -10.0,
        );
    }

    #[test]
    fn estimates_landmarks_rotated2() {
        let image = test::sad_linus_cropped();
        check_landmarks(
            image.view(RotatedRect::new(image.rect(), -10.0f32.to_radians())),
            10.0,
        );
    }
}
