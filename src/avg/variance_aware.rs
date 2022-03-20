const INIT_STDDEV: f32 = 0.1;

// TODO find existing terminology for this, it must exist

use std::collections::VecDeque;

use super::Averager;

/// An averaging algorithm that quickly adapts to changes with high variance.
///
/// A problem with the face detector output is that even after averaging the detection boxes, there
/// is a very high amount of jitter in between frames. The obvious fix is to average the detections
/// across frames, but doing enough filtering to fully remove the jitter means that if I move my
/// head, the detection lags behind for quite a few frames. That's why I came up with this
/// algorithm.
///
/// The idea is that noisy data has a certain variance, and we can distinguish noise from real
/// changes in the data by computing how many standard deviations fit between a new data point and
/// the current moving average.
///
/// If a data point's weight is increased exponentially as it moves more standard deviations away
/// from the current average, the average should catch up more quickly with real changes to the
/// data, while still smoothing out noise.
///
/// One issue with the current implementation is that (I think) the standard deviation of the data
/// changes quickly enough after a data point outside the noise range is added, that subsequent data
/// points that move further in the same direction get weighted less, because the variance of the
/// data is now considered higher, so it still needs quite a few new data points to catch up.
///
/// NOTE: this is not currently used, because an exponentially-weighted moving average has mostly
/// acceptable performance for filtering the detections, and the current state is probably broken,
/// but the idea is worth exploring further.
#[derive(Clone)]
pub struct VarianceAwareAvg {
    history: VecDeque<Entry>,
    /// Max. number of values to keep in the history.
    history_size: usize,
    /// Weighted mean of all values in `history` weighted by the entry's associated weight.
    mean: f32,
    /// Sum of all value weights in the history buffer.
    total_weight: f32,
    stddev: f32,
}

impl VarianceAwareAvg {
    pub fn new(history_size: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(history_size),
            mean: 0.0,
            stddev: 0.0,
            total_weight: 0.0,
            history_size,
        }
    }
}

impl Averager<f32> for VarianceAwareAvg {
    fn push(&mut self, value: f32) -> f32 {
        if self.history.is_empty() {
            // This is the first value, so we need to bootstrap.
            self.history.push_back(Entry { value, weight: 1.0 });
            self.total_weight = 1.0;
            self.mean = value;
            // The standard deviation of a single value is 0. However, that means that the next
            // value not equal to this `value` will get infinite weight, because it is infinite
            // standard deviations away from the mean.
            // So we initialize `stddev` with a small factor of `value` to avoid thatt.
            self.stddev = INIT_STDDEV * value;
            return value;
        }

        // Number of standard deviations that the new value is away from the current mean.
        let factor = compute_weight_factor((value - self.mean).abs(), self.stddev);
        // Weight is the unweighted mean weight times the factor.
        let weight = self.total_weight / self.history.len() as f32 * factor;

        // TODO since `factor >= 1.0`, this causes weight to grow towards infinity over time.
        // Either find a fix, or manually reduce it when it gets too large.
        self.history.push_back(Entry { value, weight });
        if self.history.len() > self.history_size {
            self.history.pop_front();
        }

        // Recompute metrics. TODO: this could be done incrementally.
        self.total_weight = 0.0;
        self.mean = 0.0;
        let mut stddev = 0.0;
        for Entry { value, weight } in &self.history {
            self.total_weight += weight;
            self.mean += value * weight;
        }
        self.mean /= self.total_weight;
        for Entry { value, weight: _ } in &self.history {
            // XXX: stddev calculation is not weighted. scholars are unclear on what exactly that means.
            stddev += (value - self.mean).powi(2);
        }
        stddev = (stddev / self.history.len() as f32).sqrt();
        if stddev == 0.0 {
            // After inserting the same value multiple times, the stddev will eventually reach 0.
            // In that case, just reuse the previous value to avoid breaking everything (NaNs).
            stddev = self.stddev;
        }
        self.stddev = stddev;

        // total weight is getting too big, scale all weights down a bit
        // TODO: seems like a really ugly hack
        if self.total_weight > self.history.len() as f32 * 100.0 {
            self.total_weight /= 100.0;
            for Entry { weight, .. } in &mut self.history {
                *weight /= 100.0;
            }
        }

        assert!(
            !self.mean.is_nan(),
            "{} {} {} {:?}",
            self.mean,
            self.stddev,
            value,
            self.history
        );

        self.mean
    }

    fn reset(&mut self) {
        self.history.clear();
    }
}

/// Compute a weight for a value that is `abs_deviation` away from mean, where the current
/// standard deviation is `stddev`.
///
/// Returned factor is 1.0 for "usual" values close to the mean, but larger for outliers.
fn compute_weight_factor(abs_deviation: f32, stddev: f32) -> f32 {
    // 2 stddevs covers 95% of expected values, so at 2 stddevs we should give a larger, but not
    // outrageous boost.
    // Since the returned value must be >=1, we use an exponential function with a base chosen so
    // that the values "looks reasonable".
    let stddevs = abs_deviation / stddev;
    let weight = 1.5f32.powf(stddevs);
    if stddevs > 4.0 {
        dbg!(stddevs, abs_deviation, weight);
    }
    weight
}

#[derive(Clone, Copy, Debug)]
struct Entry {
    value: f32,
    weight: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! assert_in {
        ($lo:expr, $val:expr, $hi:expr) => {
            assert!($lo <= $val, "value below lower bound: {}", $val);
            assert!($val <= $hi, "value above upper bound: {}", $val);
        };
    }

    #[test]
    fn test_singular() {
        let mut avg = VarianceAwareAvg::new(1);
        assert_eq!(avg.push(1.0), 1.0);
        assert_eq!(avg.push(2.0), 2.0);
        assert_eq!(avg.push(0.0), 0.0);
    }

    #[test]
    fn test_same_value() {
        let mut avg = VarianceAwareAvg::new(4);
        assert_eq!(avg.push(1.0), 1.0);
        assert_in!(1.0, avg.push(2.0), 2.0);
        assert_in!(1.0, avg.push(2.0), 2.0);
        assert_in!(1.0, avg.push(2.0), 2.0);
        assert_in!(1.0, avg.push(2.0), 2.0);
        assert_in!(1.0, avg.push(1.0), 2.0);
    }
}
