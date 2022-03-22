use crate::filter::Filter;

use super::{BoundingBox, RawDetection};

/// A [`Filter`] that operates on [`RawDetection`]s.
///
/// This applies the same (configurable) type of filter to all detection coordinates.
pub struct DetectionFilter<A> {
    xc: A,
    yc: A,
    w: A,
    h: A,
    landmarks: [(A, A); 6],
}

impl<A: Filter<f32> + Clone> DetectionFilter<A> {
    /// Creates a new detection filter that uses a clone of `filter` for every coordinate.
    pub fn new(filter: A) -> Self {
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

impl<A: Filter<f32>> Filter<RawDetection> for DetectionFilter<A> {
    fn push(&mut self, det: RawDetection) -> RawDetection {
        let mut landmarks = det.landmarks;
        for (i, lm) in landmarks.iter_mut().enumerate() {
            lm.x = self.landmarks[i].0.push(lm.x);
            lm.y = self.landmarks[i].1.push(lm.y);
        }

        RawDetection {
            bounding_box: BoundingBox {
                xc: self.xc.push(det.bounding_box.xc),
                yc: self.yc.push(det.bounding_box.yc),
                w: self.w.push(det.bounding_box.w),
                h: self.h.push(det.bounding_box.h),
            },
            landmarks,
            confidence: det.confidence,
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
