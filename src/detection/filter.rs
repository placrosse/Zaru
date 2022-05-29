use crate::filter::Filter;

use super::RawDetection;

/// A [`Filter`] that operates on [`RawDetection`]s.
///
/// This applies the same (configurable) type of filter to all detection coordinates.
pub struct DetectionFilter<F> {
    xc: F,
    yc: F,
    w: F,
    h: F,
    conf: F,
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
            conf: filter.clone(),
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

impl<F: Filter<f32>> Filter<RawDetection> for DetectionFilter<F> {
    fn push(&mut self, mut det: RawDetection) -> RawDetection {
        let landmarks = det.landmarks_mut();
        for (i, lm) in landmarks.iter_mut().enumerate() {
            lm.x = self.landmarks[i].0.push(lm.x);
            lm.y = self.landmarks[i].1.push(lm.y);
        }

        let conf = self.conf.push(det.confidence());
        det.set_confidence(conf);

        let mut rect = det.bounding_rect();
        rect.xc = self.xc.push(rect.xc);
        rect.yc = self.xc.push(rect.yc);
        rect.w = self.xc.push(rect.w);
        rect.h = self.xc.push(rect.h);
        det.set_bounding_rect(rect);

        det
    }

    fn reset(&mut self) {
        self.xc.reset();
        self.yc.reset();
        self.w.reset();
        self.h.reset();
        self.conf.reset();
        for (x, y) in &mut self.landmarks {
            x.reset();
            y.reset();
        }
    }
}
