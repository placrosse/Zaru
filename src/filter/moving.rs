//! A collection of simple moving average variants.

use std::collections::VecDeque;

use super::Filter;

/// Moving Average over a fixed history of values (FIR filter).
///
/// All values are weighted equally.
#[derive(Clone)]
pub struct MovingAvg {
    history: VecDeque<f32>,
    /// Max. number of values to keep in the history.
    history_size: usize,
}

impl MovingAvg {
    /// Creates a new moving average calculator that averages the last `history_size` values.
    pub fn new(history_size: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(history_size),
            history_size,
        }
    }
}

impl Filter<f32> for MovingAvg {
    fn push(&mut self, value: f32) -> f32 {
        self.history.push_back(value);

        if self.history.len() > self.history_size {
            self.history.pop_front();
        }

        // TODO make incremental
        let factor = 1.0 / self.history.len() as f32;
        self.history.iter().fold(0.0, |acc, v| acc + v * factor)
    }

    fn reset(&mut self) {
        self.history.clear();
    }
}

/// Exponential Moving Average – a weighted moving average whose weight decreases exponentially.
///
/// This is a tunable IIR filter.
#[derive(Clone)]
pub struct Ema {
    alpha: f32,
    last: Option<f32>,
}

impl Ema {
    /// Creates a new Exponential Moving Average calculator.
    ///
    /// The `alpha` parameter must be between 0.0 and 1.0 and defines how quickly the weight of
    /// older values should decay. Values close to 1.0 very strongly favor recent values over older
    /// values, while values closer to 0.0 favor more recent values less strongly.
    pub fn new(alpha: f32) -> Self {
        assert!(alpha >= 0.0 && alpha <= 1.0);
        Self { alpha, last: None }
    }
}

impl Filter<f32> for Ema {
    fn push(&mut self, value: f32) -> f32 {
        match self.last {
            Some(last) => {
                let avg = self.alpha * value + (1.0 - self.alpha) * last;
                self.last = Some(avg);
                avg
            }
            None => {
                self.last = Some(value);
                value
            }
        }
    }

    fn reset(&mut self) {
        self.last = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moving_avg() {
        let mut moving_avg = MovingAvg::new(2);
        assert_eq!(moving_avg.push(1.0), 1.0);
        assert_eq!(moving_avg.push(1.0), 1.0);
        assert_eq!(moving_avg.push(0.0), 0.5);
        assert_eq!(moving_avg.push(0.0), 0.0);
    }
}
