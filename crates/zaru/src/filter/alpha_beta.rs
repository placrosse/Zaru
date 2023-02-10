//! [Alpha beta filter] implementation.
//!
//! [Alpha beta filter]: https://en.wikipedia.org/wiki/Alpha_beta_filter

use super::{FilterBase, TimeBasedFilter};

/// An [alpha beta filter] that predicts a variable using its previous value and estimated rate of
/// change.
///
/// This type of filter is an extension of the exponentially-weighted moving average
/// [`Ema`][super::ema::Ema].
///
/// Alpha beta filters perform well when the measured value has an expected rate of change that is
/// constant over short periods (ie. it isn't subject to large accelerations).
///
/// [alpha beta filter]: https://en.wikipedia.org/wiki/Alpha_beta_filter
#[derive(Debug, Clone, Copy)]
pub struct AlphaBetaFilter {
    alpha: f32,
    beta: f32,
}

/// State of an [`AlphaBetaFilter`].
#[derive(Debug, Default)]
pub struct AlphaBetaState {
    /// Last filter prediction. Initially `None`.
    x: Option<f32>,
    /// Predicted change in `x` per second.
    v: f32,
}

impl AlphaBetaFilter {
    pub fn new(alpha: f32, beta: f32) -> Self {
        assert!(0.0 <= alpha && alpha <= 1.0);
        assert!(0.0 <= beta && beta <= 1.0);
        Self { alpha, beta }
    }
}

impl FilterBase<f32> for AlphaBetaFilter {
    type State = AlphaBetaState;
}

impl TimeBasedFilter<f32> for AlphaBetaFilter {
    fn filter(&self, state: &mut Self::State, value: f32, elapsed: f32) -> f32 {
        match &mut state.x {
            None => {
                state.x = Some(value);
                value
            }
            Some(x) => {
                let prediction = *x + state.v * elapsed;
                let residual = value - prediction;

                *x = prediction + self.alpha * residual;
                state.v += self.beta * residual / elapsed;

                *x
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpha_beta_filter() {
        let filter = AlphaBetaFilter::new(0.5, 0.1);
        let state = &mut Default::default();

        assert_eq!(filter.filter(state, 10.0, 0.2), 10.0);
        assert_eq!(filter.filter(state, 10.0, 0.2), 10.0);
        assert_eq!(filter.filter(state, 10.0, 0.2), 10.0);
        assert_eq!(filter.filter(state, 10.0, 0.2), 10.0);

        assert_eq!(filter.filter(state, -10.0, 0.2), 0.0);
        assert_eq!(filter.filter(state, -10.0, 0.2), -6.0);
        assert_eq!(filter.filter(state, -10.0, 0.2), -9.4);
    }
}
