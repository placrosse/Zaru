//! A wrapper around MediaPipe's [Face Mesh] landmark predictor network.
//!
//! The [`reference_positions`] function returns the positions of the landmarks for a "reference
//! face" that faces the camera head-on.
//!
//! [Face Mesh]: https://google.github.io/mediapipe/solutions/face_mesh.html

// NOTE: MediaPipe also has a `face_landmarks_with_attention` network, which outputs more accurate
// eye and mouth landmarks. However, it uses custom ops, and so can't be converted to a
// non-TensorFlow format.

use include_blob::include_blob;
use itertools::Itertools;
use nalgebra::{Rotation2, Vector2};
use once_cell::sync::Lazy;

use crate::image::{draw, AsImageViewMut, Color, ImageViewMut};
use crate::landmark::{Confidence, Landmark};
use crate::nn::ColorMapper;
use crate::rect::RotatedRect;
use crate::{
    iter::zip_exact,
    num::{sigmoid, TotalF32},
};
use crate::{
    landmark::{self, Landmarks},
    nn::{Cnn, CnnInputShape, NeuralNetwork, Outputs},
    slice::SliceExt,
};

static MODEL_V1: Lazy<Cnn> = Lazy::new(|| {
    let model_data = include_blob!("../../3rdparty/onnx/face_landmark.onnx");
    Cnn::new(
        NeuralNetwork::from_onnx(model_data)
            .unwrap()
            .load()
            .unwrap(),
        CnnInputShape::NCHW,
        ColorMapper::linear(-1.0..=1.0),
    )
    .unwrap()
});

static MODEL_V2: Lazy<Cnn> = Lazy::new(|| {
    let model_data = include_blob!("../../3rdparty/onnx/face_landmarks_detector.onnx");
    Cnn::new(
        NeuralNetwork::from_onnx(model_data)
            .unwrap()
            .load()
            .unwrap(),
        CnnInputShape::NCHW,
        ColorMapper::linear(-1.0..=1.0),
    )
    .unwrap()
});

/// Estimates facial landmarks using the MediaPipe Face Mesh network.
///
/// The input image must be a cropped image of a face.
///
/// The image should depict a face that is mostly upright. Results will be poor if the face is
/// rotated too much. A [`LandmarkTracker`] can be used to automatically follow the rotation of a
/// face and pass an upright view to the network.
///
/// [`LandmarkTracker`]: crate::landmark::LandmarkTracker
pub struct FaceMeshV1;

impl landmark::Network for FaceMeshV1 {
    type Output = LandmarkResult;

    fn cnn(&self) -> &Cnn {
        &MODEL_V1
    }

    fn extract(&self, outputs: &Outputs, estimate: &mut Self::Output) {
        estimate.face_flag = sigmoid(outputs[1].index([0, 0, 0, 0]).as_singular());

        let landmark_coords = outputs[0].index([0, 0, 0]);
        for (&[x, y, z], out) in zip_exact(
            landmark_coords.as_slice().array_chunks_exact::<3>(),
            estimate.landmarks.positions_mut(),
        ) {
            out[0] = x;
            out[1] = y;
            out[2] = z;
        }
    }
}

/// Estimates facial and iris landmarks using the improved MediaPipe Face Mesh network.
///
/// This network estimates the same set of landmarks as [`FaceMeshV1`] (the reference positions can
/// be obtained via [`reference_positions`]), and augments these landmarks with 5 iris landmarks per
/// eye that can be used for eye tracking without requiring a separate iris landmark network.
/// Additionally, this network also computes a `tongueOut` blendshape that indicates whether the
/// pictured person is sticking their tongue out, which can be used for animations.
pub struct FaceMeshV2;

impl landmark::Network for FaceMeshV2 {
    type Output = LandmarkResultV2;

    fn cnn(&self) -> &Cnn {
        &MODEL_V2
    }

    fn extract(&self, outputs: &Outputs, estimate: &mut Self::Output) {
        estimate.face_flag = sigmoid(outputs[1].index([0, 0, 0, 0]).as_singular());

        // (sigmoid applied inside model)
        estimate.tongue_out = outputs[2].index([0, 0]).as_singular();

        let landmark_coords = outputs[0].index([0, 0, 0]);
        for (&[x, y, z], out) in zip_exact(
            landmark_coords.as_slice().array_chunks_exact::<3>(),
            estimate.landmarks.positions_mut(),
        ) {
            out[0] = x;
            out[1] = y;
            out[2] = z;
        }
    }
}

/// Landmark results estimated by [`FaceMeshV1`].
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

    #[inline]
    pub fn landmarks(&self) -> &Landmarks {
        &self.landmarks
    }

    #[inline]
    pub fn landmarks_mut(&mut self) -> &mut Landmarks {
        &mut self.landmarks
    }

    pub fn rotation_radians(&self) -> f32 {
        let left_eye = self
            .landmarks()
            .get(LandmarkIdx::LeftEyeOuterCorner as _)
            .position();
        let right_eye = self
            .landmarks()
            .get(LandmarkIdx::RightEyeOuterCorner as _)
            .position();
        let left_to_right_eye =
            Vector2::new(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]);
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
                let [x, y, ..] = self.landmarks().get(idx as usize).position();
                [x, y]
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
                let [x, y, ..] = self.landmarks().get(idx as usize).position();
                [x, y]
            }),
        )
        .unwrap()
    }

    /// Draws the landmark result onto an image.
    pub fn draw<I: AsImageViewMut>(&self, image: &mut I) {
        self.draw_impl(&mut image.as_view_mut());
    }

    fn draw_impl(&self, image: &mut ImageViewMut<'_>) {
        for &[x, y, ..] in self.landmarks().positions() {
            draw::marker(image, x, y).size(3);
        }

        let color = match self.confidence() {
            0.75.. => Color::GREEN,
            0.5..=0.75 => Color::YELLOW,
            _ => Color::RED,
        };
        let (x_min, x_max) = self
            .landmarks()
            .positions()
            .iter()
            .map(|[x, ..]| TotalF32(*x))
            .minmax()
            .into_option()
            .unwrap();
        let x = (x_min.0 + x_max.0) / 2.0;
        let y = self
            .landmarks()
            .positions()
            .iter()
            .map(|[_, y, ..]| TotalF32(*y))
            .min()
            .unwrap()
            .0;
        draw::text(
            image,
            x,
            y - 3.0,
            &format!("conf={:.01}", self.confidence()),
        )
        .align_bottom()
        .color(color);

        let left_eye = self
            .landmarks()
            .get(LandmarkIdx::LeftEyeOuterCorner as _)
            .position();
        let right_eye = self
            .landmarks()
            .get(LandmarkIdx::RightEyeOuterCorner as _)
            .position();
        draw::line(image, left_eye[0], left_eye[1], right_eye[0], right_eye[1]).color(Color::WHITE);
        draw::text(
            image,
            (left_eye[0] + right_eye[0]) / 2.0,
            (left_eye[1] + right_eye[1]) / 2.0,
            &format!("{:.1} deg", self.rotation_radians().to_degrees()),
        )
        .align_bottom()
        .color(Color::WHITE);
    }
}

impl landmark::Estimate for LandmarkResult {
    #[inline]
    fn landmarks_mut(&mut self) -> &mut Landmarks {
        &mut self.landmarks
    }

    #[inline]
    fn angle_radians(&self) -> Option<f32> {
        Some(self.rotation_radians())
    }
}

impl landmark::Confidence for LandmarkResult {
    #[inline]
    fn confidence(&self) -> f32 {
        self.face_flag
    }
}

#[derive(Clone)]
pub struct LandmarkResultV2 {
    landmarks: Landmarks,
    face_flag: f32,
    tongue_out: f32,
}

impl Default for LandmarkResultV2 {
    fn default() -> Self {
        Self {
            landmarks: Landmarks::new(Self::NUM_LANDMARKS),
            face_flag: 0.0,
            tongue_out: 0.0,
        }
    }
}

impl LandmarkResultV2 {
    pub const NUM_LANDMARKS: usize = 478;

    #[inline]
    pub fn landmarks(&self) -> &Landmarks {
        &self.landmarks
    }

    #[inline]
    pub fn landmarks_mut(&mut self) -> &mut Landmarks {
        &mut self.landmarks
    }

    /// Returns an iterator over the landmarks corresponding to the reference mesh (ie. without the
    /// iris landmarks, which are not part of the reference mesh).
    pub fn mesh_landmarks(&self) -> impl Iterator<Item = Landmark> + '_ {
        self.landmarks.iter().take(LandmarkResult::NUM_LANDMARKS)
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
                let [x, y, ..] = self.landmarks().get(idx as usize).position();
                [x, y]
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
                let [x, y, ..] = self.landmarks().get(idx as usize).position();
                [x, y]
            }),
        )
        .unwrap()
    }

    /// Returns the 5 landmarks marking the left iris (from the perspective of the camera).
    ///
    /// The first landmark is the center of the iris, the 4 other surround the iris on the left,
    /// right, top and bottom.
    pub fn left_iris(&self) -> impl Iterator<Item = Landmark> + '_ {
        self.landmarks
            .iter()
            .skip(LandmarkResult::NUM_LANDMARKS)
            .take(5)
    }

    /// Returns the 5 landmarks surrounding the right iris (from the perspective of the camera).
    ///
    /// The first landmark is the center of the iris, the 4 other surround the iris on the left,
    /// right, top and bottom.
    pub fn right_iris(&self) -> impl Iterator<Item = Landmark> + '_ {
        self.landmarks
            .iter()
            .skip(LandmarkResult::NUM_LANDMARKS + 5)
    }

    pub fn left_eye_contour(&self) -> [Landmark; 16] {
        #[rustfmt::skip]
        let indices = [
            /* top */ 33, 246, 161, 160, 159, 158, 157, 173, 133,
            /* bottom */ 7, 163, 144, 145, 153, 154, 155,
        ];
        indices.map(|i| self.landmarks.get(i))
    }

    pub fn right_eye_contour(&self) -> [Landmark; 16] {
        #[rustfmt::skip]
        let indices = [
            /* top */ 362, 398, 384, 385, 386, 387, 388, 466, 263,
            /* bottom */ 382, 381, 380, 374, 373, 390, 249,
        ];
        indices.map(|i| self.landmarks.get(i))
    }

    /// Returns the value of the *tongue out* blendshape (in range 0..=1).
    ///
    /// Note that this is not a *confidence* value, but a blendshape. Any value above ~0.1 typically
    /// means that the tongue is visibly out.
    #[inline]
    pub fn tongue_out(&self) -> f32 {
        self.tongue_out
    }

    pub fn rotation_radians(&self) -> f32 {
        let left_eye = self
            .landmarks()
            .get(LandmarkIdx::LeftEyeOuterCorner as _)
            .position();
        let right_eye = self
            .landmarks()
            .get(LandmarkIdx::RightEyeOuterCorner as _)
            .position();
        let left_to_right_eye =
            Vector2::new(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]);
        Rotation2::rotation_between(&Vector2::x(), &left_to_right_eye).angle()
    }

    /// Draws the landmark result onto an image.
    pub fn draw<I: AsImageViewMut>(&self, image: &mut I) {
        self.draw_impl(&mut image.as_view_mut());
    }

    fn draw_impl(&self, image: &mut ImageViewMut<'_>) {
        for lm in self.mesh_landmarks() {
            draw::marker(image, lm.x(), lm.y()).size(3);
        }
        for (i, lm) in self.left_iris().enumerate() {
            let size = if i == 0 { 3 } else { 1 };
            draw::marker(image, lm.x(), lm.y())
                .size(size)
                .color(Color::GREEN);
        }
        for (i, lm) in self.right_iris().enumerate() {
            let size = if i == 0 { 3 } else { 1 };
            draw::marker(image, lm.x(), lm.y())
                .size(size)
                .color(Color::BLUE);
        }

        let color = match self.confidence() {
            0.75.. => Color::GREEN,
            0.5..=0.75 => Color::YELLOW,
            _ => Color::RED,
        };
        let (x_min, x_max) = self
            .landmarks()
            .positions()
            .iter()
            .map(|[x, ..]| TotalF32(*x))
            .minmax()
            .into_option()
            .unwrap();
        let x = (x_min.0 + x_max.0) / 2.0;
        let y = self
            .landmarks()
            .positions()
            .iter()
            .map(|[_, y, ..]| TotalF32(*y))
            .min()
            .unwrap()
            .0;
        draw::text(
            image,
            x,
            y - 3.0,
            &format!(
                "conf={:.01}, tongue={:.01}",
                self.confidence(),
                self.tongue_out()
            ),
        )
        .align_bottom()
        .color(color);

        draw::text(
            image,
            x,
            y + 10.0,
            &format!("{:.1} deg", self.rotation_radians().to_degrees()),
        )
        .align_bottom()
        .color(color);
    }
}

impl landmark::Estimate for LandmarkResultV2 {
    #[inline]
    fn landmarks_mut(&mut self) -> &mut Landmarks {
        &mut self.landmarks
    }

    #[inline]
    fn angle_radians(&self) -> Option<f32> {
        Some(self.rotation_radians())
    }
}

impl landmark::Confidence for LandmarkResultV2 {
    #[inline]
    fn confidence(&self) -> f32 {
        self.face_flag
    }
}

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../3rdparty/3d/canonical_face_model.rs"
));

/// Returns an iterator over the vertices of the reference face mesh.
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
///
/// This enum is valid for the landmarks in both [`LandmarkResult`] and [`LandmarkResultV2`].
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

impl From<LandmarkIdx> for usize {
    #[inline]
    fn from(idx: LandmarkIdx) -> usize {
        idx as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{AsImageView, ImageView};
    use crate::landmark::Confidence;
    use crate::{landmark::Estimator, procrustes::ProcrustesAnalyzer, test};

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

        let mut lm = Estimator::new(FaceMeshV1);
        let landmarks = lm.estimate(&image);
        assert!(landmarks.confidence() > 0.9);
        check_angle(expected_radians, landmarks.rotation_radians());
        check_angle(expected_radians, landmarks.left_eye().rotation_radians());
        check_angle(expected_radians, landmarks.right_eye().rotation_radians());

        if degrees.abs() <= 45.0f32 {
            assert!(landmarks.left_eye().x_center() < landmarks.right_eye().x_center());
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
