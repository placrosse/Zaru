use crate::filter::Filter;

use super::{BoundingBox, Detection, RawDetection};

/// A [`Filter`] that operates on [`Detection`]s.
///
/// This applies the same (configurable) type of filter to all detection coordinates.
pub struct DetectionFilter<F> {
    xc: F,
    yc: F,
    w: F,
    h: F,
    landmarks: [(F, F); 6],
}

impl<F: Filter<f32> + Clone> DetectionFilter<F> {
    /// Creates a new detection filter that uses a clone of `filter` to filter every coordinate.
    pub fn new(filter: F) -> Self {
        Self {
            xc: filter.clone(),
            yc: filter.clone(),
            w: filter.clone(),
            h: filter.clone(),
            landmarks: [
                (filter.clone(), filter.clone()),
                (filter.clone(), filter.clone()),
                (filter.clone(), filter.clone()),
                (filter.clone(), filter.clone()),
                (filter.clone(), filter.clone()),
                (filter.clone(), filter.clone()),
            ],
        }
    }
}

impl<F: Filter<f32>> Filter<Detection> for DetectionFilter<F> {
    fn push(&mut self, det: Detection) -> Detection {
        let mut landmarks = det.raw.landmarks;
        for (i, lm) in landmarks.iter_mut().enumerate() {
            lm.x = self.landmarks[i].0.push(lm.x);
            lm.y = self.landmarks[i].1.push(lm.y);
        }

        Detection {
            full_res: det.full_res,
            raw: RawDetection {
                bounding_box: BoundingBox {
                    xc: self.xc.push(det.raw.bounding_box.xc),
                    yc: self.yc.push(det.raw.bounding_box.yc),
                    w: self.w.push(det.raw.bounding_box.w),
                    h: self.h.push(det.raw.bounding_box.h),
                },
                landmarks,
                confidence: det.raw.confidence,
            },
        }
    }

    fn reset(&mut self) {
        self.xc.reset();
        self.yc.reset();
        self.w.reset();
        self.h.reset();
        for (x, y) in &mut self.landmarks {
            x.reset();
            y.reset();
        }
    }
}
