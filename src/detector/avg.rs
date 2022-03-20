use crate::avg::Averager;

use super::{BoundingBox, RawDetection};

/// An [`Averager`] that operates on [`RawDetection`]s.
pub struct DetectionAvg<A> {
    xc: A,
    yc: A,
    w: A,
    h: A,
    landmarks: [(A, A); 6],
}

impl<A: Averager<f32> + Clone> DetectionAvg<A> {
    /// Creates a new detection averager that uses clones of `averager` for every coordinate.
    pub fn new(averager: A) -> Self {
        Self {
            xc: averager.clone(),
            yc: averager.clone(),
            w: averager.clone(),
            h: averager.clone(),
            landmarks: [
                (averager.clone(), averager.clone()),
                (averager.clone(), averager.clone()),
                (averager.clone(), averager.clone()),
                (averager.clone(), averager.clone()),
                (averager.clone(), averager.clone()),
                (averager.clone(), averager.clone()),
            ],
        }
    }
}

impl<A: Averager<f32>> Averager<RawDetection> for DetectionAvg<A> {
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
