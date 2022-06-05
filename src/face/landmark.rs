//! Facial landmark detection.
//!
//! Also known as *face alignment* or *registration*.
//!
//! This uses one of the neural networks also used in MediaPipe's [Face Mesh] pipeline.
//!
//! [Face Mesh]: https://google.github.io/mediapipe/solutions/face_mesh.html

use std::ops::Index;

use once_cell::sync::Lazy;

use crate::{
    image::{AsImageView, ImageView, Rect},
    iter::zip_exact,
    nn::{create_linear_color_mapper, unadjust_aspect_ratio, Cnn, CnnInputShape, NeuralNetwork},
    resolution::{AspectRatio, Resolution},
    timer::Timer,
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

/// A neural network based facial landmark predictor.
pub struct Landmarker {
    model: &'static Cnn,
    t_resize: Timer,
    t_infer: Timer,
    /// Large, so keep one around and return by ref.
    result_buffer: LandmarkResult,
}

impl Landmarker {
    /// Creates a new facial landmark calculator.
    pub fn new() -> Self {
        Self {
            t_resize: Timer::new("resize"),
            t_infer: Timer::new("infer"),
            result_buffer: LandmarkResult {
                landmarks: Landmarks {
                    positions: [(0.0, 0.0, 0.0); 468],
                },
                face_flag: 0.0,
                orig_res: Resolution::new(1, 1),
                orig_aspect: AspectRatio::SQUARE,
                input_res: MODEL.input_resolution(),
            },
            model: &MODEL,
        }
    }

    /// Returns the expected input resolution of the internal neural network.
    pub fn input_resolution(&self) -> Resolution {
        self.model.input_resolution()
    }

    /// Computes facial landmarks in `image`.
    ///
    /// `image` must be a cropped image of a face. When using [`Detector`], the
    /// rectangle returned by [`Detection::bounding_rect_loose`] produces good results.
    ///
    /// The image should depict a face that is mostly upright. Results will be poor if the face is
    /// rotated too much.
    ///
    /// [`Detector`]: super::detection::Detector
    /// [`Detection::bounding_rect_loose`]: super::detection::Detection::bounding_rect_loose
    pub fn compute<V: AsImageView>(&mut self, image: &V) -> &LandmarkResult {
        self.compute_impl(image.as_view())
    }

    fn compute_impl(&mut self, image: ImageView<'_>) -> &LandmarkResult {
        let input_res = self.model.input_resolution();
        let full_res = image.resolution();
        self.result_buffer.orig_res = full_res;
        self.result_buffer.orig_aspect = full_res.aspect_ratio();

        let mut image = image.reborrow();
        let resized;
        if image.resolution() != input_res {
            resized = self.t_resize.time(|| image.aspect_aware_resize(input_res));
            image = resized.as_view();
        }
        let result = self.t_infer.time(|| self.model.estimate(&image)).unwrap();
        log::trace!("inference result: {:?}", result);

        self.result_buffer.face_flag = result[1].index([0, 0, 0, 0]).as_singular();
        for (coords, out) in zip_exact(
            result[0].index([0, 0, 0]).as_slice().chunks(3),
            &mut self.result_buffer.landmarks.positions,
        ) {
            out.0 = coords[0];
            out.1 = coords[1];
            out.2 = coords[2];
        }

        &self.result_buffer
    }

    /// Returns profiling timers for image resizing and neural inference.
    pub fn timers(&self) -> impl IntoIterator<Item = &Timer> + '_ {
        [&self.t_resize, &self.t_infer]
    }
}

/// Landmark results returned by [`Landmarker::compute`].
#[derive(Clone)]
pub struct LandmarkResult {
    landmarks: Landmarks,
    face_flag: f32,
    orig_res: Resolution,
    orig_aspect: AspectRatio,
    input_res: Resolution,
}

impl LandmarkResult {
    /// Returns the 3D landmark positions in the input image's coordinate system.
    pub fn landmark_positions(&self) -> impl Iterator<Item = (f32, f32, f32)> + '_ {
        (0..self.landmark_count()).map(|index| self.landmark_position(index))
    }

    /// Returns a landmark's position in the input image's coordinate system.
    pub fn landmark_position(&self, index: usize) -> (f32, f32, f32) {
        let (x, y, z) = self.landmarks.positions[index];
        let (x, y) = unadjust_aspect_ratio(
            x / self.input_res.width() as f32,
            y / self.input_res.height() as f32,
            self.orig_aspect,
        );
        let (x, y) = (
            x * self.orig_res.width() as f32,
            y * self.orig_res.height() as f32,
        );
        (x, y, z)
    }

    #[inline]
    pub fn landmark_count(&self) -> usize {
        self.landmarks.positions.len()
    }

    /// Returns a reference to the raw landmarks, unadjusted for the input image resolution.
    #[inline]
    pub fn raw_landmarks(&self) -> &Landmarks {
        &self.landmarks
    }

    /// Returns the confidence that the input image contains a proper face.
    ///
    /// This can be used to estimate the fit quality, or to re-run face detection if that isn't done
    /// each frame. Typical values are >20.0 when a good landmark fit is produced, between 10 and 20
    /// when the face is rotated a bit too far, and <10 when the face is rotated much too far or
    /// there is no face in the input image.
    #[inline]
    pub fn face_confidence(&self) -> f32 {
        // TODO: sigmoid?
        self.face_flag
    }

    /// Returns a [`Rect`] containing the left eye.
    pub fn left_eye(&self) -> Rect {
        Rect::bounding(
            [
                LandmarkIdx::LeftEyeBottom,
                LandmarkIdx::LeftEyeLeftCorner,
                LandmarkIdx::LeftEyeRightCorner,
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

    /// Returns a [`Rect`] containing the right eye.
    pub fn right_eye(&self) -> Rect {
        Rect::bounding(
            [
                LandmarkIdx::RightEyeBottom,
                LandmarkIdx::RightEyeLeftCorner,
                LandmarkIdx::RightEyeRightCorner,
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
}

/// Raw face landmark positions.
#[derive(Clone)]
pub struct Landmarks {
    positions: [(f32, f32, f32); 468],
}

impl Landmarks {
    /// Returns an iterator over the positions of all landmarks.
    pub fn positions(&self) -> impl Iterator<Item = (f32, f32, f32)> + '_ {
        self.positions.iter().copied()
    }
}

impl Index<LandmarkIdx> for Landmarks {
    type Output = (f32, f32, f32);

    fn index(&self, index: LandmarkIdx) -> &Self::Output {
        &self.positions[index as usize]
    }
}

impl Index<usize> for Landmarks {
    type Output = (f32, f32, f32);

    fn index(&self, index: usize) -> &Self::Output {
        &self.positions[index]
    }
}

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/3rdparty/3d/canonical_face_model.rs"
));

/// Returns an iterator over the vertices of the reference face model.
///
/// Each point yielded by the returned iterator corresponds to the same point in the sequence
/// of landmarks output by [`Landmarker`], but the scale and coordinate system does not: The points
/// returned by this function have Y pointing up, and X and Y are in a smaller range around `(0,0)`,
/// while [`Landmarker`] yields points that have Y point down, and X and Y are in term of the input
/// image's coordinates.
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
    LeftEyeLeftCorner = 33,
    LeftEyeRightCorner = 133,
    LeftEyeTop = 159,
    LeftEyeBottom = 145,
    RightEyeLeftCorner = 362,
    RightEyeRightCorner = 263,
    RightEyeTop = 386,
    RightEyeBottom = 374,
    RightEyebrowLeftCorner = 295,
    LeftEyebrowRightCorner = 65,
}

impl Into<usize> for LandmarkIdx {
    #[inline]
    fn into(self) -> usize {
        self as usize
    }
}

/// Uses a [`Landmarker`] to track the position of a face across frames.
///
/// The last known position of the tracked face is remembered across frames, and used to locate it
/// in the next frame. To start tracking a face, the user has to set the [`TrackedFace`] by calling
/// [`LandmarkTracker::set_tracked_face`].
///
/// This can be used to avoid the jitter from running a [`Detector`] on every frame of a video feed,
/// in cases where the application would compute the landmarks anyways.
///
/// [`Detector`]: super::detection::Detector
pub struct LandmarkTracker {
    lm: Landmarker,
    face: Option<TrackedFace>,
    loss_thresh: f32,
    input_aspect: AspectRatio,
}

impl LandmarkTracker {
    const DEFAULT_TRACKING_LOSS_THRESHOLD: f32 = 10.0;

    /// Creates a new landmark tracker which initially does not track a face.
    pub fn new() -> Self {
        let lm = Landmarker::new();
        Self {
            input_aspect: lm.input_resolution().aspect_ratio(),
            lm,
            face: None,
            loss_thresh: Self::DEFAULT_TRACKING_LOSS_THRESHOLD,
        }
    }

    /// Returns the currently tracked face rectangle and rotation.
    pub fn tracked_face(&self) -> Option<&TrackedFace> {
        self.face.as_ref()
    }

    /// Sets face rectangle and rotation to track.
    pub fn set_tracked_face(&mut self, face: TrackedFace) {
        self.face = Some(face);
    }

    /// Sets the confidence threshold below which a tracked face is considered "lost".
    ///
    /// This uses the confidence value returned by [`LandmarkResult::face_confidence`].
    pub fn set_tracking_loss_threshold(&mut self, thresh: f32) {
        self.loss_thresh = thresh;
    }

    pub fn landmarker(&self) -> &Landmarker {
        &self.lm
    }

    /// Tracks the face across the next frame.
    ///
    /// If tracking succeeds, returns the updated [`TrackedFace`] as well as the computed facial
    /// landmarks.
    ///
    /// If no face is currently tracked, or the newly computed landmarks have too low of a
    /// confidence (as set by [`LandmarkTracker::set_tracking_loss_threshold`]), returns `None`.
    pub fn track<V: AsImageView>(&mut self, image: &V) -> Option<TrackingResult<'_>> {
        self.track_impl(image.as_view())
    }

    fn track_impl(&mut self, image: ImageView<'_>) -> Option<TrackingResult<'_>> {
        // 1. Extract rotated face
        // 2. Run landmarker
        // 3. Update tracked rect and rotation

        let mut face = self.face?;
        let view_rect = face.rect.grow_to_fit_aspect(self.input_aspect);
        let view = image.view(&view_rect); // TODO rotate
        let res = self.lm.compute(&view);
        if res.face_confidence() < self.loss_thresh {
            log::debug!(
                "LandmarkTracker: confidence {}, loss threshold {} -> LOST",
                res.face_confidence(),
                self.loss_thresh,
            );

            self.face = None;
            return None;
        }

        let extra_area = 0.3;
        let rect = Rect::bounding(res.landmark_positions().map(|(x, y, _z)| (x as _, y as _)))
            .unwrap()
            .grow_rel(extra_area, extra_area, extra_area, extra_area);
        face.rect = rect.move_by(view_rect.x(), view_rect.y());
        // TODO set rotation
        let face = self.face.insert(face);

        Some(TrackingResult {
            updated_face: face,
            landmarks: res,
            view_rect,
        })
    }
}

pub struct TrackingResult<'a> {
    updated_face: &'a TrackedFace,
    landmarks: &'a LandmarkResult,
    view_rect: Rect,
}

impl<'a> TrackingResult<'a> {
    pub fn updated_face(&self) -> &TrackedFace {
        self.updated_face
    }

    /// Returns the extracted landmarks, relative to `view_rect`.
    pub fn landmarks(&self) -> &LandmarkResult {
        self.landmarks
    }

    /// Returns the rectangle inside the input image that was used to compute the landmarks.
    pub fn view_rect(&self) -> Rect {
        self.view_rect
    }
}

/// Location info of a face tracked by a [`LandmarkTracker`].
#[derive(Clone, Copy)]
pub struct TrackedFace {
    /// The rectangle where we last recorded this face. This will be used as the image view in which
    /// we'll run the landmarker.
    rect: Rect,
    rotation: f32,
}

impl TrackedFace {
    pub fn new(rect: Rect, rotation: f32) -> Self {
        Self { rect, rotation }
    }

    pub fn rect(&self) -> &Rect {
        &self.rect
    }

    pub fn rotation(&self) -> f32 {
        self.rotation
    }
}
