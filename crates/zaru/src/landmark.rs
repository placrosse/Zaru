//! Common code for visual landmark estimation.

use std::iter;

use crate::image::{AsImageView, AspectRatio, ImageView, Resolution};
use crate::iter::zip_exact;

use crate::rect::RotatedRect;
use crate::{
    filter::Filter,
    nn::{Cnn, Outputs},
    timer::Timer,
};

type Position = [f32; 3];

#[derive(Clone)]
pub struct Landmarks {
    positions: Box<[Position]>,
    visibility: Option<Box<[f32]>>,
    presence: Option<Box<[f32]>>,
}

impl Landmarks {
    /// Creates a new [`Landmarks`] collection containing `len` preallocated landmarks.
    ///
    /// All landmarks will start with all coordinates at `0.0`.
    pub fn new(len: usize) -> Self {
        Self {
            positions: vec![[0.0, 0.0, 0.0]; len].into_boxed_slice(),
            visibility: None,
            presence: None,
        }
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = Landmark> + Clone + '_ {
        (0..self.positions.len()).map(|i| self.get(i))
    }

    pub fn get(&self, index: usize) -> Landmark {
        let mut lm = Landmark::new(self.positions[index]);
        if let Some(vis) = &self.visibility {
            lm = lm.with_visibility(vis[index]);
        }
        if let Some(pres) = &self.presence {
            lm = lm.with_presence(pres[index]);
        }
        lm
    }

    pub fn set(&mut self, index: usize, landmark: Landmark) {
        let len = self.positions.len();
        self.positions[index] = landmark.pos;
        if let Some(vis) = landmark.visibility {
            self.visibility.get_or_insert_with(|| vec![0.0; len].into())[index] = vis;
        }
        if let Some(pres) = landmark.presence {
            self.presence.get_or_insert_with(|| vec![0.0; len].into())[index] = pres;
        }
    }

    pub fn positions(&self) -> &[Position] {
        &self.positions
    }

    pub fn positions_mut(&mut self) -> &mut [Position] {
        &mut self.positions
    }

    pub fn average_position(&self) -> Position {
        let mut center = [0.0; 3];
        for pos in self.positions() {
            center[0] += pos[0] / self.positions().len() as f32;
            center[1] += pos[1] / self.positions().len() as f32;
            center[2] += pos[2] / self.positions().len() as f32;
        }
        center
    }

    pub fn map_positions(&mut self, mut f: impl FnMut(Position) -> Position) {
        for pos in self.positions_mut() {
            *pos = f(*pos);
        }
    }
}

/// A landmark in 3D space.
#[derive(Debug, PartialEq, PartialOrd, Clone, Copy)]
pub struct Landmark {
    pos: [f32; 3],
    visibility: Option<f32>,
    presence: Option<f32>,
}

impl Landmark {
    pub fn new(position: [f32; 3]) -> Self {
        Self {
            pos: position,
            visibility: None,
            presence: None,
        }
    }

    pub fn with_visibility(self, visibility: f32) -> Self {
        Self {
            visibility: Some(visibility),
            ..self
        }
    }

    pub fn with_presence(self, presence: f32) -> Self {
        Self {
            presence: Some(presence),
            ..self
        }
    }

    #[inline]
    pub fn position(&self) -> Position {
        self.pos
    }

    #[inline]
    pub fn x(&self) -> f32 {
        self.pos[0]
    }

    #[inline]
    pub fn y(&self) -> f32 {
        self.pos[1]
    }

    #[inline]
    pub fn z(&self) -> f32 {
        self.pos[2]
    }
}

/// Batch-filter for landmarks.
///
/// This should be applied to the unadjusted landmarks output by the neural network, otherwise the
/// filter parameters require tuning that depends on the input image size, which may vary across
/// invocations.
pub struct LandmarkFilter {
    filter: Box<dyn FnMut(&mut Landmarks) + Send>,
}

/// The default [`LandmarkFilter`] does not perform any filtering.
impl Default for LandmarkFilter {
    fn default() -> Self {
        Self {
            filter: Box::new(|_| ()),
        }
    }
}

impl LandmarkFilter {
    /// Creates a new landmark filter.
    ///
    /// # Parameters
    ///
    /// - `filter` is the set of filter parameters to use.
    /// - `num_landmarks` is the number of landmarks that will be filtered with this filter in each
    ///   batch.
    pub fn new<F: Filter<f32> + Send + 'static>(filter: F, num_landmarks: usize) -> Self
    where
        F::State: Send,
    {
        let mut states = iter::repeat_with(|| {
            [
                F::State::default(),
                F::State::default(),
                F::State::default(),
            ]
        })
        .take(num_landmarks)
        .collect::<Vec<_>>();

        Self {
            filter: Box::new(move |landmarks| {
                for (lm, state) in zip_exact(&mut *landmarks.positions, &mut states) {
                    for (coord, state) in zip_exact(lm, state) {
                        *coord = filter.filter(state, *coord);
                    }
                }
            }),
        }
    }

    /// Filters a list of landmarks in-place.
    ///
    /// # Panics
    ///
    /// This method panics if `landmarks` does not have exactly as many entries as were specified in
    /// the `num_landmarks` parameter in the call to [`LandmarkFilter::new`].
    pub fn filter(&mut self, landmarks: &mut Landmarks) {
        (self.filter)(landmarks);
    }
}

/// Trait for landmark estimation results returned by [`Estimator::estimate`].
pub trait Estimate: Send + Sync + 'static {
    /// Returns the predicted [`Landmarks`].
    fn landmarks_mut(&mut self) -> &mut Landmarks;

    /// Returns the estimated clockwise object rotation in radians.
    ///
    /// This will be used by [`LandmarkTracker`] to automatically rotate the object before passing
    /// it to the [`Estimator`]. This helps some estimators that expect the object to be in a
    /// certain orientation.
    ///
    /// If this returns [`None`], no angle estimate is available, and the RoI will not automatically
    /// follow the rotation of the tracked object.
    fn angle_radians(&self) -> Option<f32> {
        None
    }
}

/// Trait for network inference results that contain a confidence value.
///
/// The confidence value can be used to detect when the object becomes obscured or leaves the
/// camera's field of view. It is required by [`LandmarkTracker`] in order to decide when to stop
/// tracking a particular region of interest.
pub trait Confidence {
    /// Confidence value indicating whether the tracked object is in view.
    ///
    /// By convention, this is in range 0.0 to 1.0, with anything above 0.5 indicating that the
    /// tracked object is probably still in view. If a different range is used, the tracking loss
    /// threshold probably needs adjusting.
    fn confidence(&self) -> f32;

    // FIXME: should this be named `presence` instead? confidence is a very broad term
}

/// Trait implemented by wrapper types around neural networks that estimate landmarks.
pub trait Network: Send + Sync + 'static {
    /// Type representing the predicted landmarks.
    type Output: Estimate;

    /// Returns the [`Cnn`] to use for landmark estimation.
    fn cnn(&self) -> &Cnn;

    /// Extracts the network outputs and writes them to `estimate`.
    ///
    /// The landmark positions are expected to be in the coordinate system of the network's input.
    fn extract(&self, outputs: &Outputs, estimate: &mut Self::Output);
}

/// Neural-network based landmark estimator.
///
/// This estimator processes an input image and yields an [`Estimate`] of type `E`, containing the
/// derived [`Landmarks`] and other data (depending on the network).
pub struct Estimator<E: Estimate> {
    network: Box<dyn Network<Output = E>>,
    estimate: E,
    t_infer: Timer,
    t_extract: Timer,
    t_filter: Timer,
    filter: LandmarkFilter,
}

impl<E: Estimate + Default> Estimator<E> {
    pub fn new<N: Network<Output = E>>(network: N) -> Self {
        Self {
            network: Box::new(network),
            estimate: E::default(),
            t_infer: Timer::new("infer"),
            t_extract: Timer::new("extract"),
            t_filter: Timer::new("filter"),
            filter: LandmarkFilter::default(),
        }
    }
}

impl<E: Estimate> Estimator<E> {
    /// Returns the expected input resolution of the internal neural network.
    ///
    /// If an image is passed that has a different resolution, it will be sampled to match the input
    /// resolution. The [`Estimator`] will also automatically ensure that the aspect ratio matches
    /// by creating an oversized view of the input.
    pub fn input_resolution(&self) -> Resolution {
        self.network.cnn().input_resolution()
    }

    /// Returns profiling timers for this landmark estimator.
    pub fn timers(&self) -> impl Iterator<Item = &Timer> + '_ {
        [&self.t_infer, &self.t_extract, &self.t_filter].into_iter()
    }

    /// Sets the [`LandmarkFilter`] to apply to all landmark positions.
    ///
    /// The filter will be applied after inference, but before adjusting landmark coordinates back
    /// to the input image's coordinates.
    ///
    /// This should only be used if the estimator is fed subsequent frames from an animation or
    /// video feed.
    pub fn set_filter(&mut self, filter: LandmarkFilter) {
        self.filter = filter;
    }

    /// Performs landmark estimation on `image`, returning the [`Estimate`].
    ///
    /// If the aspect ratio of `image` does not match the aspect ratio of the network's input, an
    /// enlarged [`ImageView`] of the right aspect ratio is created first. If `image` is a view into
    /// a larger base image, this may include more pixels from the base image that aren't included
    /// in `image`. Otherwise, it adds black bars to pad the image to the right aspect ratio.
    pub fn estimate<V: AsImageView>(&mut self, image: &V) -> &mut E {
        self.estimate_impl(image.as_view())
    }

    fn estimate_impl(&mut self, image: ImageView<'_>) -> &mut E {
        let cnn = self.network.cnn();
        let input_res = cnn.input_resolution();

        // If the input image's aspect ratio doesn't match the CNN's input, create an oversized view
        // that does.
        let rect = image
            .rect()
            .grow_to_fit_aspect(input_res.aspect_ratio().unwrap());
        let view = image.view(rect);
        let outputs = self.t_infer.time(|| cnn.estimate(&view)).unwrap();
        log::trace!("inference result: {:?}", outputs);

        self.t_extract
            .time(|| self.network.extract(&outputs, &mut self.estimate));

        // Importantly, the filter uses the network's coordinates, which makes filter parameters
        // independent of the image's dimensions.
        self.t_filter
            .time(|| self.filter.filter(self.estimate.landmarks_mut()));

        // Map landmark coordinates back into the input image.
        let scale = rect.width() / input_res.width() as f32;
        for pos in self.estimate.landmarks_mut().positions_mut() {
            // Map all coordinates from the network's input coordinate system to `rect`'s system.
            *pos = pos.map(|t| t * scale);

            // Now remove the offset added by the oversized rectangle (this compensates for
            // "black bars" added to adjust the aspect ratio).
            pos[0] += rect.x();
            pos[1] += rect.y();
        }

        &mut self.estimate
    }
}

/// Tracks a region of interest (RoI) across subsequent frames by tracking the movement of estimated
/// landmarks.
///
/// Once seeded with a region of interest, the tracker will adjust its RoI based on the bounding
/// rectangle of the estimated landmarks.
///
/// This requires a landmark estimator that outputs a confidence value via the [`Estimator`] trait,
/// indicating whether the tracked object is still in view.
///
/// If the [`Estimator`] supports it, the tracker will also track the object's rotation.
pub struct LandmarkTracker<E: Estimate + Confidence> {
    aspect_ratio: AspectRatio,
    estimator: Estimator<E>,
    roi: Option<RotatedRect>,
    loss_thresh: f32,
    roi_padding: f32,
}

impl<E: Estimate + Confidence> LandmarkTracker<E> {
    pub const DEFAULT_LOSS_THRESHOLD: f32 = 0.5;

    pub const DEFAULT_ROI_PADDING: f32 = 0.3;

    /// Creates a new [`LandmarkTracker`].
    pub fn new(estimator: Estimator<E>) -> Self {
        Self {
            aspect_ratio: estimator.input_resolution().aspect_ratio().unwrap(),
            estimator,
            roi: None,
            loss_thresh: Self::DEFAULT_LOSS_THRESHOLD,
            roi_padding: Self::DEFAULT_ROI_PADDING,
        }
    }

    /// Returns a reference to the [`Estimator`] used to estimate landmarks.
    pub fn estimator(&self) -> &Estimator<E> {
        &self.estimator
    }

    /// Returns profiling timers of the internal [`Estimator`].
    pub fn timers(&self) -> impl Iterator<Item = &Timer> {
        self.estimator.timers()
    }

    /// Sets the tracking loss threshold.
    ///
    /// If the confidence value of the predicted landmarks falls below this value, tracking is
    /// considered lost: the RoI is cleared, [`LandmarkTracker::track`] will return [`None`], and
    /// tracking has to be re-seeded by calling [`LandmarkTracker::set_roi`].
    ///
    /// By default, [`LandmarkTracker::DEFAULT_LOSS_THRESHOLD`] is used.
    pub fn set_loss_threshold(&mut self, threshold: f32) {
        self.loss_thresh = threshold;
    }

    /// Sets the relative amount of RoI padding to apply.
    ///
    /// The RoI is updated based on the bounding rectangle of the estimated landmarks, and enlarged
    /// by the RoI padding value.
    ///
    /// The padding is relative to the bounding rectangle's width and height. It is added to all
    /// sides of the rectangle individually. For example, a padding value of `0.1` results in 10%
    /// of the rectangles height to be added to its top and bottom, and 10% of its width to be added
    /// to its left and right sides.
    ///
    /// By default, [`LandmarkTracker::DEFAULT_ROI_PADDING`] is used.
    ///
    /// # Panics
    ///
    /// This method panics when `padding` is less than 0.0 or when it is NaN.
    pub fn set_roi_padding(&mut self, padding: f32) {
        assert!(padding >= 0.0);
        self.roi_padding = padding;
    }

    /// Returns the current region of interest.
    ///
    /// If no region of interest is being tracked, or tracking was lost, returns [`None`].
    pub fn roi(&self) -> Option<&RotatedRect> {
        self.roi.as_ref()
    }

    /// Sets the region of interest.
    ///
    /// This can be passed either a [`Rect`][crate::rect::Rect] or a [`RotatedRect`].
    ///
    /// Note that this does not apply RoI padding. The rectangle is used as-is.
    pub fn set_roi(&mut self, roi: impl Into<RotatedRect>) {
        self.roi = Some(roi.into());
    }

    /// Performs landmark tracking on `full_image`.
    ///
    /// If no RoI is set, this returns `None`. [`LandmarkTracker::set_roi`] must be called in order
    /// to start tracking. It also has to be called to restart tracking when tracking is lost.
    ///
    /// If the estimator indicates that tracking is lost, the RoI is cleared and `None` is returned.
    /// Otherwise, the RoI is updated to the bounding rectangle of all landmarks, with some added
    /// padding that can be configured with [`LandmarkTracker::set_roi_padding`].
    ///
    /// The returned [`TrackingResult`] grants access to the estimated landmarks, using `full_image`
    /// coordinates.
    ///
    /// `track` always has to be called on images of the same size, otherwise the tracking window
    /// won't match between frames.
    pub fn track<V>(&mut self, full_image: &V) -> Option<TrackingResult<'_, E>>
    where
        V: AsImageView,
    {
        self.track_impl(full_image.as_view())
    }

    fn track_impl(&mut self, full_image: ImageView<'_>) -> Option<TrackingResult<'_, E>> {
        let roi = self.roi?;
        let view_rect = roi.map(|rect| rect.grow_to_fit_aspect(self.aspect_ratio));
        let view = full_image.view(view_rect);
        let estimate = self.estimator.estimate(&view);
        if estimate.confidence() < self.loss_thresh {
            log::trace!(
                "LandmarkTracker: confidence {}, loss threshold {} -> LOST",
                estimate.confidence(),
                self.loss_thresh,
            );

            self.roi = None;
            return None;
        }

        let angle = roi.rotation_radians() + estimate.angle_radians().unwrap_or(0.0);

        // Map all landmarks to the image coordinate system.
        for [x, y, _] in estimate.landmarks_mut().positions_mut() {
            [*x, *y] = view_rect.transform_out(*x, *y);
        }

        let updated_roi = RotatedRect::bounding(
            angle,
            estimate.landmarks_mut().iter().map(|lm| [lm.x(), lm.y()]),
        )
        .unwrap();

        self.roi = Some(updated_roi.map(|rect| rect.grow_rel(self.roi_padding)));

        Some(TrackingResult {
            view_rect,
            estimate,
            updated_roi,
        })
    }
}

/// The result returned by [`LandmarkTracker::track`].
pub struct TrackingResult<'a, E: Estimate> {
    view_rect: RotatedRect,
    estimate: &'a E,
    updated_roi: RotatedRect,
}

impl<'a, E: Estimate> TrackingResult<'a, E> {
    /// Returns the rectangle inside the full image passed to [`LandmarkTracker::track`] that was
    /// used to compute the landmarks.
    pub fn view_rect(&self) -> RotatedRect {
        self.view_rect
    }

    /// Returns the estimation result, including landmarks.
    ///
    /// Landmark coordinates are in terms of the full image passed to [`LandmarkTracker::track`].
    pub fn estimate(&self) -> &'a E {
        self.estimate
    }

    /// Returns the RoI that will be used in the next call to [`LandmarkTracker::track`].
    pub fn updated_roi(&self) -> RotatedRect {
        self.updated_roi
    }
}
