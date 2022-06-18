use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use crate::{
    image::{Image, Rect},
    landmark::LandmarkTracker,
    pipeline::{promise, Promise, PromiseHandle, Receiver, Worker},
};

use super::{
    detection::{self, Detection, PalmDetector},
    landmark::{self, LandmarkResult, Landmarker},
};

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
    pub const DEFAULT_IOU_THRESH: f32 = 0.3;

    pub const DEFAULT_REDETECT_INTERVAL: Duration = Duration::from_millis(300);

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
                move |recv: Receiver<(Arc<Image>, Promise<Vec<Detection>>)>| {
                    for (image, promise) in recv {
                        let detections = palm_detector.detect(&*image);
                        promise.fulfill(detections.to_vec());
                    }
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
    pub fn set_redetect_interval(&mut self, interval: Duration) {
        self.det_interval = interval;
    }

    /// Sets the intersection-over-union threshold at which two tracking regions are considered to
    /// overlap.
    ///
    /// Since redetection will also detect all hands that are already being tracked, this threshold
    /// is used to ensure that each hand is only tracked once. No new tracking worker is spawned if
    /// a detection overlaps with an existing tracker's region of interest.
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

    /// Blocks until all tracking from the last call to `track` are finished, and restarts them on
    /// `image`.
    ///
    /// When this method returns, [`HandTracker::hands`] will return the state of all hands in the
    /// image passed to the last call to `track`.
    pub fn track(&mut self, image: Arc<Image>) {
        self.hands.retain_mut(|hand| {
            let (promise, ph) = promise();
            let old_ph = std::mem::replace(&mut hand.ph, ph);
            match old_ph.block().unwrap() {
                Some(lm) => {
                    hand.worker.send((image.clone(), promise)).unwrap();
                    hand.lm = Some(lm);
                    true
                }
                None => {
                    promise.fulfill(None); // defuse promise
                    false
                }
            }
        });

        let mut detections = match &self.detections_handle {
            Some(handle) if handle.is_fulfilled() => {
                self.detections_handle.take().unwrap().block().unwrap()
            }
            _ => Vec::new(),
        };

        let grow_by = 1.5; // Palm -> Hand grow factor

        // Compute IoU with existing RoIs, discard detection if it overlaps, spawn tracker when
        // it doesn't.
        detections.retain(|det| {
            for hand in &self.hands {
                if hand
                    .roi
                    .lock()
                    .unwrap()
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
            let roi = det.bounding_rect().grow_rel(grow_by);
            let mut landmarker = self.landmarker.clone();
            let mut tracker =
                LandmarkTracker::new(landmarker.input_resolution().aspect_ratio().unwrap());
            tracker.set_roi(roi);
            let roi_arc = Arc::new(Mutex::new(roi));
            let roi_arc2 = roi_arc.clone();
            let mut worker = Worker::spawn(
                "hand tracker",
                move |recv: Receiver<(Arc<Image>, Promise<_>)>| {
                    for (image, promise) in recv {
                        match tracker.track(&mut landmarker, &*image) {
                            Some(res) => {
                                *roi_arc2.lock().unwrap() = res.updated_roi();

                                let mut lm = res.estimation().clone();
                                lm.move_by(
                                    res.view_rect().x() as f32,
                                    res.view_rect().y() as f32,
                                    0.0,
                                );
                                promise.fulfill(Some(lm));
                            }
                            None => {
                                log::trace!("tracking lost, exiting worker thread");
                                promise.fulfill(None);
                                break;
                            }
                        }
                    }
                },
            )
            .unwrap();

            let id = self.next_hand_id;
            self.next_hand_id.0 += 1;
            let (promise, ph) = promise();
            worker.send((image.clone(), promise)).unwrap();
            TrackedHand {
                id,
                roi: roi_arc.clone(),
                worker,
                ph,
                lm: None,
            }
        }));

        // TODO: might need a pass that removes trackers whose RoIs have started to overlap since
        // they were spawned.

        if (self.hands.is_empty() || Instant::now() >= self.next_det)
            && self.detections_handle.is_none()
        {
            // If the detector is waiting for a message, start a detection.
            let (promise, handle) = promise();
            if let Ok(()) = self.detector.send((image.clone(), promise)) {
                self.detections_handle = Some(handle);
                self.next_det += self.det_interval;
            }
        }
    }
}

struct TrackedHand {
    id: HandId,
    roi: Arc<Mutex<Rect>>,
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
    view_rect: Rect,
}

impl<'a> HandData<'a> {
    pub fn id(&self) -> HandId {
        self.id
    }

    /// Hand landmarks, in global coordinates.
    pub fn landmark_result(&self) -> &LandmarkResult {
        self.lm
    }

    pub fn view_rect(&self) -> Rect {
        self.view_rect
    }
}
