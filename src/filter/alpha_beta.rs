use super::TimeBasedFilter;

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
    x: f32,
    /// Predicted change in `x` per call to `push`.
    v: f32,
}

impl AlphaBetaFilter {
    pub fn new(alpha: f32, beta: f32) -> Self {
        assert!(0.0 <= alpha && alpha <= 1.0);
        assert!(0.0 <= beta && beta <= 1.0);
        Self { alpha, beta }
    }
}

impl TimeBasedFilter<f32> for AlphaBetaFilter {
    type State = AlphaBetaState;

    fn filter(&self, state: &mut Self::State, value: f32, elapsed: f32) -> f32 {
        let prediction = state.x + state.v * elapsed;
        let residual = value - prediction;

        state.x = prediction + self.alpha * residual;
        state.v += self.beta * residual / elapsed;

        state.x
    }
}
