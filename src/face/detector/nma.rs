//! Non-Maximum Averaging computes a confidence-weighted average of overlapping detections.
//!
//! Sadly, the filtered detections are still extremely jittery.

use crate::num::TotalF32;

use super::{BoundingBox, Landmark, RawDetection};

pub struct NonMaxAvg {
    seed_thresh: f32,
    iou_thresh: f32,
    avg_buf: Vec<RawDetection>,
    out_buf: Vec<RawDetection>,
}

impl NonMaxAvg {
    /// Creates a new non-maximum averager.
    ///
    /// # Parameters
    ///
    /// - `seed_thresh`: required detection confidence to "seed" an NMA round with a detection.
    /// - `contrib_thresh`: required detection confidence for a detection to participate in
    ///   confidence-weighted averaging.
    /// - `iou_thresh`: required intersection-over-union to consider two detections as overlapping.
    pub(super) fn new(seed_thresh: f32, iou_thresh: f32) -> Self {
        Self {
            seed_thresh,
            iou_thresh,
            avg_buf: Vec::new(),
            out_buf: Vec::new(),
        }
    }

    pub(super) fn average(&mut self, detections: &mut Vec<RawDetection>) -> &[RawDetection] {
        self.out_buf.clear();

        // Sort by ascending confidence.
        detections.sort_unstable_by_key(|det| TotalF32(det.confidence));

        while let Some(seed) = detections.pop() {
            if seed.confidence < self.seed_thresh {
                // no more significant detections left
                break;
            }

            self.avg_buf.clear();
            self.avg_buf.push(seed);
            detections.retain(|other| {
                let iou = seed.bounding_box.iou(&other.bounding_box);
                if iou >= self.iou_thresh {
                    self.avg_buf.push(*other);
                    false // remove from detection list
                } else {
                    true
                }
            });

            // compute confidence-weighted average of the overlapping detections
            let mut acc = RawDetection {
                bounding_box: BoundingBox {
                    xc: 0.0,
                    yc: 0.0,
                    w: 0.0,
                    h: 0.0,
                },
                landmarks: [Landmark { x: 0.0, y: 0.0 }; 6],
                confidence: 0.0,
            };
            let mut divisor = 0.0;
            for det in &self.avg_buf {
                let factor = det.confidence;
                divisor += factor;
                for (acc, lm) in acc.landmarks.iter_mut().zip(&det.landmarks) {
                    acc.x += lm.x * factor;
                    acc.y += lm.y * factor;
                }
                acc.bounding_box.xc += det.bounding_box.xc * factor;
                acc.bounding_box.yc += det.bounding_box.yc * factor;
                acc.bounding_box.w += det.bounding_box.w * factor;
                acc.bounding_box.h += det.bounding_box.h * factor;

                acc.confidence = f32::max(acc.confidence, det.confidence);
            }

            for lm in &mut acc.landmarks {
                lm.x /= divisor;
                lm.y /= divisor;
            }
            acc.bounding_box.xc /= divisor;
            acc.bounding_box.yc /= divisor;
            acc.bounding_box.w /= divisor;
            acc.bounding_box.h /= divisor;

            self.out_buf.push(acc);
        }

        &self.out_buf
    }
}
