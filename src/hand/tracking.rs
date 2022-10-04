//! Detection and tracking of multiple hands.
//!
//! This is a higher-level module that provides a self-contained hand tracking solution that will
//! detect and track any number of hands, and compute landmarks for each one.

use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use crate::{
    image::{Image, RotatedRect},
    landmark::LandmarkTracker,
    worker::{promise, Promise, PromiseHandle, Worker},
};

use super::{
    detection::{self, Detection, PalmDetector},
    landmark::{self, LandmarkResult, Landmarker},
};

/// Self-contained hand detector, tracker, and landmarker.
pub struct HandTracker {
    hands: Vec<TrackedHand>,
    next_hand_id: HandId,
    detector: Worker<(Arc<Image>, Promise<Vec<Detection>>)>,
    detections_handle: Option<PromiseHandle<Vec<Detection>>>,
    next_det: Instant,
    det_interval: Duration,
    landmarker: Landmarker,
    iou_thresh: f32,
}

impl HandTracker {
    /// Default intersection-over-union threshold for deduplicating tracking regions.
    pub const DEFAULT_IOU_THRESH: f32 = 0.3;

    /// Default interval for palm detection, when at least one hand is in view.
    pub const DEFAULT_REDETECT_INTERVAL: Duration = Duration::from_millis(300);

    /// Creates a new [`HandTracker`] with the given palm detection and landmarking networks.
    pub fn new<D, L>(detector: D, landmarker: L) -> Self
    where
        D: detection::PalmDetectionNetwork,
        L: landmark::LandmarkNetwork,
    {
        let mut palm_detector = PalmDetector::new(detector);
        Self {
            hands: Vec::new(),
            next_hand_id: HandId(0),
            detector: Worker::spawn(
                "palm detector",
                move |(image, promise): (Arc<Image>, Promise<Vec<Detection>>)| {
                    let detections = palm_detector.detect(&*image);
                    promise.fulfill(detections.to_vec());
                },
            )
            .unwrap(),
            detections_handle: None,
            next_det: Instant::now(),
            det_interval: Self::DEFAULT_REDETECT_INTERVAL,
            landmarker: Landmarker::new(landmarker),
            iou_thresh: Self::DEFAULT_IOU_THRESH,
        }
    }

    /// Sets the redetection interval.
    ///
    /// Redetection works as follows:
    /// - if no hands are currently being tracked, `track` will always attempt to trigger a
    ///   detection
    /// - otherwise, if the last detection was triggered more than the redetect interval ago,
    ///   `track` will attempt to trigger another detection
    ///
    /// "Attempt" means that if a detection is already ongoing, nothing happens, and if not, one is
    /// started.
    ///
    /// This scheme ensures that newly appearing hands get picked up in a timely fashion, but
    /// without blocking (detection is expensive, so it runs in its own worker).
    ///
    /// By default, [`Self::DEFAULT_REDETECT_INTERVAL`] is used.
    pub fn set_redetect_interval(&mut self, interval: Duration) {
        self.det_interval = interval;
    }

    /// Sets the intersection-over-union threshold at which two tracking regions are considered to
    /// overlap.
    ///
    /// Since redetection will also detect all hands that are already being tracked, this threshold
    /// is used to ensure that each hand is only tracked once. No new tracking worker is spawned if
    /// a detection overlaps with an existing tracker's region of interest.
    ///
    /// By default, [`Self::DEFAULT_IOU_THRESH`] is used.
    pub fn set_iou_thresh(&mut self, thresh: f32) {
        self.iou_thresh = thresh;
    }

    /// Returns an iterator over the tracking data for each hand.
    pub fn hands(&self) -> impl Iterator<Item = HandData<'_>> {
        self.hands.iter().filter_map(|hand| {
            hand.lm.as_ref().map(|lm| HandData {
                id: hand.id,
                lm,
                view_rect: *hand.roi.lock().unwrap(),
            })
        })
    }

    /// Blocks until all tracking computations from the last call to `track` are finished, and
    /// restarts them on `image`.
    ///
    /// After this method returns, [`HandTracker::hands`] will return the state of all hands in the
    /// previous image passed to `track`.
    pub fn track(&mut self, image: Arc<Image>) {
        self.hands.retain_mut(|hand| {
            let (promise, ph) = promise();
            let old_ph = std::mem::replace(&mut hand.ph, ph);
            match old_ph.block().unwrap() {
                Some(lm) => {
                    hand.worker.send((image.clone(), promise));
                    hand.lm = Some(lm);
                    true
                }
                None => false,
            }
        });

        let mut detections = match &self.detections_handle {
            Some(handle) if handle.is_fulfilled() => {
                self.detections_handle.take().unwrap().block().unwrap()
            }
            _ => Vec::new(),
        };

        let grow_by = 1.5; // Palm -> Hand grow factor

        // Compute IoU with existing RoIs, discard detection if it overlaps with any, spawn tracker
        // when it doesn't.
        detections.retain(|det| {
            for hand in &self.hands {
                if hand
                    .roi
                    .lock()
                    .unwrap()
                    .rect()
                    .iou(&det.bounding_rect().grow_rel(grow_by))
                    >= self.iou_thresh
                {
                    // Overlap
                    return false;
                }
            }

            true
        });

        self.hands.extend(detections.iter().map(|det| {
            let roi = RotatedRect::new(
                det.bounding_rect().grow_rel(grow_by),
                det.rotation_radians(),
            );
            let mut landmarker = self.landmarker.clone();
            let mut tracker =
                LandmarkTracker::new(landmarker.input_resolution().aspect_ratio().unwrap());
            tracker.set_roi(roi);
            let roi_arc = Arc::new(Mutex::new(roi));
            let roi_arc2 = roi_arc.clone();
            let mut worker = Worker::spawn(
                "hand tracker",
                move |(image, promise): (Arc<Image>, Promise<_>)| match tracker
                    .track(&mut landmarker, &*image)
                {
                    Some(res) => {
                        *roi_arc2.lock().unwrap() = res.updated_roi();

                        let lm = res.estimation().clone();
                        promise.fulfill(Some(lm));
                    }
                    None => {
                        log::trace!("tracking lost");
                        promise.fulfill(None);
                    }
                },
            )
            .unwrap();

            let id = self.next_hand_id;
            self.next_hand_id.0 += 1;
            let (promise, ph) = promise();
            worker.send((image.clone(), promise));
            TrackedHand {
                id,
                roi: roi_arc.clone(),
                worker,
                ph,
                lm: None,
            }
        }));

        // Check if any of the tracked regions started to overlap, and remove one of them.
        for i in (0..self.hands.len()).rev() {
            let roi = *self.hands[i].roi.lock().unwrap();

            for j in 0..i {
                let other_roi = *self.hands[j].roi.lock().unwrap();
                // FIXME: IoU computation ignores rotation because hard
                if roi.rect().iou(&other_roi.rect()) >= self.iou_thresh {
                    self.hands.swap_remove(i);
                    break;
                }
            }
        }

        if (self.hands.is_empty() || Instant::now() >= self.next_det)
            && self.detections_handle.is_none()
        {
            // We want to start a detection, and none is currently running, so start one.
            let (promise, handle) = promise();
            self.detector.send((image.clone(), promise));
            self.detections_handle = Some(handle);
            self.next_det += self.det_interval;
        }
    }
}

struct TrackedHand {
    id: HandId,
    roi: Arc<Mutex<RotatedRect>>,
    worker: Worker<(Arc<Image>, Promise<Option<LandmarkResult>>)>,
    ph: PromiseHandle<Option<LandmarkResult>>,
    lm: Option<LandmarkResult>,
}

/// ID of a tracked hand.
///
/// The assigned [`HandId`]s are unique per [`HandTracker`] assigning them. They are reused between
/// frames for as long as the hand is tracked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HandId(u64);

/// Tracking data returned for a hand in the input image.
pub struct HandData<'a> {
    id: HandId,
    lm: &'a LandmarkResult,
    view_rect: RotatedRect,
}

impl<'a> HandData<'a> {
    /// Returns the unique ID of this hand.
    #[inline]
    pub fn id(&self) -> HandId {
        self.id
    }

    /// Returns the hand landmarks, in global image coordinates.
    #[inline]
    pub fn landmark_result(&self) -> &LandmarkResult {
        self.lm
    }

    /// Returns the hand's bounding rectangle in the original image.
    #[inline]
    pub fn view_rect(&self) -> RotatedRect {
        self.view_rect
    }
}
