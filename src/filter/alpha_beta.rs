use super::Filter;

/// An [alpha beta filter] that predicts a variable using its previous value and estimated rate of
/// change.
///
/// This type of filter is related to the Kalman filter, but much simpler. It is also a sort of
/// extension of moving averages and specifically the exponentially-weighted moving average
/// [`Ema`][super::Ema].
///
/// Alpha beta filters perform well when the measured value has an expected rate of change that is
/// constant over short periods (ie. it isn't subject to large accelerations).
///
/// FIXME: this implementation does not allow adjusting for uneven measurement intervals, so it
/// expects that `push` is called roughly in constant intervals. This also means that using a
/// different interval might require a change to the `alpha` or `beta` parameters to perform well.
/// This should be fixed by introducing the concept of time-dependent filters.
///
/// [alpha beta filter]: https://en.wikipedia.org/wiki/Alpha_beta_filter
#[derive(Clone, Debug)]
pub struct AlphaBetaFilter {
    alpha: f32,
    beta: f32,
    x: f32,
    /// Predicted change in `x` per call to `push`.
    v: f32,
}

impl AlphaBetaFilter {
    pub fn new(alpha: f32, beta: f32) -> Self {
        assert!(0.0 <= alpha && alpha <= 1.0);
        assert!(0.0 <= beta && beta <= 1.0);
        Self {
            alpha,
            beta,
            x: 0.0,
            v: 0.0,
        }
    }
}

impl Filter<f32> for AlphaBetaFilter {
    fn push(&mut self, value: f32) -> f32 {
        let prediction = self.x + self.v;
        let residual = value - prediction;

        self.x = prediction + self.alpha * residual;
        self.v = self.v + self.beta * residual;

        self.x
    }

    fn reset(&mut self) {
        self.x = 0.0;
        self.v = 0.0;
    }
}
