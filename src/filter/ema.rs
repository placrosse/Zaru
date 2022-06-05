//! Exponential Moving Average.

use super::Filter;

/// An Exponential Moving Average (EMA) filter.
#[derive(Debug, Clone, Copy)]
pub struct Ema {
    alpha: f32,
}

impl Ema {
    /// Creates a new Exponential Moving Average filter.
    ///
    /// The `alpha` parameter must be between 0.0 and 1.0 and defines how quickly the weight of
    /// older values should decay. Values closer to 1.0 favor recent values over older values, while
    /// values closer to 0.0 favor more recent values less strongly.
    ///
    /// # Panics
    ///
    /// This method will panic if `alpha` is not in between 0.0 and 1.0.
    pub fn new(alpha: f32) -> Self {
        assert!(alpha >= 0.0 && alpha <= 1.0);
        Self { alpha }
    }
}

/// Filter state for [`Ema`] filters.
#[derive(Debug, Default)]
pub struct State {
    last: Option<f32>,
}

impl Filter<f32> for Ema {
    type State = State;

    fn filter(&self, state: &mut Self::State, value: f32) -> f32 {
        match state.last {
            Some(last) => {
                let avg = self.alpha * value + (1.0 - self.alpha) * last;
                state.last = Some(avg);
                avg
            }
            None => {
                state.last = Some(value);
                value
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::filter::SimpleFilter;

    use super::*;

    #[test]
    fn test_ema() {
        let mut filter = SimpleFilter::new(Ema::new(0.5));
        assert_eq!(filter.filter(1.0), 1.0);
        assert_eq!(filter.filter(2.0), 1.5);
        assert_eq!(filter.filter(2.0), 1.75);
    }
}
