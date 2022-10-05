//! Common code for visual landmark estimation.

use std::iter;

use crate::{
    filter::Filter,
    image::{AsImageView, RotatedRect},
    iter::zip_exact,
    resolution::AspectRatio,
};

type Position = [f32; 3];

#[derive(Default, Clone)]
pub struct Landmarks {
    positions: Vec<Position>,
}

impl Landmarks {
    /// Creates a new [`Landmarks`] collection containing `len` preallocated landmarks.
    ///
    /// All landmarks will start with all coordinates at `0.0`.
    pub fn new(len: usize) -> Self {
        Self {
            positions: vec![[0.0, 0.0, 0.0]; len],
        }
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = Landmark> + Clone + '_ {
        self.positions.iter().map(|&pos| Landmark { pos })
    }

    pub fn landmark(&self, index: usize) -> Landmark {
        Landmark {
            pos: self.positions[index],
        }
    }

    pub fn positions(&self) -> &[Position] {
        &self.positions
    }

    pub fn positions_mut(&mut self) -> &mut [Position] {
        &mut self.positions
    }
}

/// A landmark in 3D space.
#[derive(Debug, PartialEq, PartialOrd, Clone, Copy)]
pub struct Landmark {
    pos: [f32; 3],
}

impl Landmark {
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
                for (lm, state) in zip_exact(&mut landmarks.positions, &mut states) {
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
pub trait Estimation {
    /// Confidence value indicating whether the tracked object is in view.
    ///
    /// By convention, this is in range 0.0 to 1.0, with anything above 0.5 indicating that the
    /// tracked object is probably still in view. If a different range is used, the tracking loss
    /// threshold probably needs adjusting.
    fn confidence(&self) -> f32;

    /// Returns the predicted [`Landmarks`].
    ///
    /// The landmark coordinates must be the the coordinate system of the image passed to
    /// [`Estimator::estimate`].
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

/// Trait for landmark estimators.
///
/// Implementing this trait allows using a landmark estimator with [`LandmarkTracker`].
pub trait Estimator {
    /// Type representing the predicted landmarks.
    type Estimation: Estimation;

    /// Performs landmark estimation on `image`.
    ///
    /// This method has to accept images of arbitrary resolution and aspect ratio.
    ///
    /// This trait requires that the estimation result is allocated within `self` and returned by
    /// reference to avoid unnecessary copies. Once Rust has GATs, the design should be changed to
    /// be more flexible with this.
    fn estimate<V: AsImageView>(&mut self, image: &V) -> &mut Self::Estimation;
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
/// If the [`Estimator`] supports it, the tracking will also track the object's rotation.
pub struct LandmarkTracker {
    roi: Option<RotatedRect>,
    loss_thresh: f32,
    roi_padding: f32,
    input_ratio: AspectRatio,
}

impl LandmarkTracker {
    pub const DEFAULT_LOSS_THRESHOLD: f32 = 0.5;

    pub const DEFAULT_ROI_PADDING: f32 = 0.3;

    /// Creates a new [`LandmarkTracker`].
    pub fn new(input_ratio: AspectRatio) -> Self {
        Self {
            roi: None,
            loss_thresh: Self::DEFAULT_LOSS_THRESHOLD,
            roi_padding: Self::DEFAULT_ROI_PADDING,
            input_ratio,
        }
    }

    /// Sets the tracking loss threshold.
    ///
    /// If the confidence value of the predicted landmarks falls below this value, tracking is
    /// considered lost: the RoI is cleared, [`LandmarkTracker::track`] will return `None`, and
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
    pub fn set_roi_padding(&mut self, padding: f32) {
        self.roi_padding = padding;
    }

    /// Returns the current region of interest.
    pub fn roi(&self) -> Option<&RotatedRect> {
        self.roi.as_ref()
    }

    /// Sets the region of interest.
    ///
    /// This can be passed either a [`Rect`][crate::image::Rect] or a [`RotatedRect`].
    ///
    /// Note that this does not apply RoI padding. The rectangle is used as-is
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
    /// won't match between frames. The same estimator should also be used to ensure that landmark
    /// and confidence value meanings stay the same across subsequent frames.
    pub fn track<'e, L: Estimator, V: AsImageView>(
        &mut self,
        estimator: &'e mut L,
        full_image: &V,
    ) -> Option<TrackingResult<'e, L>> {
        let roi = self.roi?;
        let view_rect = roi.map(|rect| rect.grow_to_fit_aspect(self.input_ratio));
        let full_image = full_image.as_view();
        let view = full_image.view(view_rect);
        let estimation = estimator.estimate(&view);
        if estimation.confidence() < self.loss_thresh {
            log::trace!(
                "LandmarkTracker: confidence {}, loss threshold {} -> LOST",
                estimation.confidence(),
                self.loss_thresh,
            );

            self.roi = None;
            return None;
        }

        let angle = roi.rotation_radians() + estimation.angle_radians().unwrap_or(0.0);

        // Map all landmarks to the image coordinate system.
        for [x, y, _] in estimation.landmarks_mut().positions_mut() {
            [*x, *y] = view_rect.transform_out_f32(*x, *y);
        }

        let updated_roi = RotatedRect::bounding(
            angle,
            estimation
                .landmarks_mut()
                .iter()
                .map(|lm| (lm.x().round() as i32, lm.y().round() as i32)),
        )
        .unwrap();

        self.roi = Some(updated_roi.map(|rect| rect.grow_rel(self.roi_padding)));

        Some(TrackingResult {
            view_rect,
            estimation,
            updated_roi,
        })
    }
}

/// The result returned by [`LandmarkTracker::track`].
pub struct TrackingResult<'a, L: Estimator> {
    view_rect: RotatedRect,
    estimation: &'a L::Estimation,
    updated_roi: RotatedRect,
}

impl<'a, L: Estimator> TrackingResult<'a, L> {
    /// Returns the rectangle inside the full image passed to [`LandmarkTracker::track`] that was
    /// used to compute the landmarks.
    pub fn view_rect(&self) -> RotatedRect {
        self.view_rect
    }

    /// Returns the estimation result, including landmarks.
    ///
    /// Landmark coordinates are in terms of the full image passed to [`LandmarkTracker::track`].
    pub fn estimation(&self) -> &'a L::Estimation {
        self.estimation
    }

    /// Returns the RoI that will be used in the next call to [`LandmarkTracker::track`].
    pub fn updated_roi(&self) -> RotatedRect {
        self.updated_roi
    }
}
