//! An implementation of the [1€ Filter].
//!
//! [1€ Filter]: https://gery.casiez.net/1euro/

use std::f32::consts::PI;

use super::{FilterBase, TimeBasedFilter};

/// [1€ Filter] parameters.
///
/// [1€ Filter]: https://gery.casiez.net/1euro/
#[derive(Debug, Clone, Copy)]
pub struct OneEuroFilter {
    beta: f32,
    min_cutoff: f32,
    d_cutoff: f32,
}

impl OneEuroFilter {
    /// Creates a new set of 1€ Filter parameters.
    ///
    /// # Parameters
    ///
    /// - `min_cutoff` or *fcmin* is the minimum cutoff frequency. Lowering this value reduces
    ///   jitter but increases lag.
    /// - `beta` is the speed coefficient. Increasing this value reduces lag.
    ///
    /// # Panics
    ///
    /// `min_cutoff` must be greater than 0.0, and `beta` must be 0.0 or greater, otherwise this
    /// function will panic.
    pub fn new(min_cutoff: f32, beta: f32) -> Self {
        assert!(min_cutoff > 0.0);
        assert!(beta >= 0.0);
        Self {
            beta,
            min_cutoff,
            d_cutoff: 1.0,
        }
    }

    /// Returns a copy of `self` with a different derivative frequency cutoff value.
    ///
    /// This value defaults to 1.0 and typically does not need to be adjusted.
    pub fn with_d_cutoff(self, d_cutoff: f32) -> Self {
        Self { d_cutoff, ..self }
    }
}

/// Filter state for the [`OneEuroFilter`].
#[derive(Debug, Default)]
pub struct OneEuroFilterState {
    prev: Option<Prev>,
}

#[derive(Debug)]
struct Prev {
    x: f32,
    dx: f32,
}

impl FilterBase<f32> for OneEuroFilter {
    type State = OneEuroFilterState;
}

impl TimeBasedFilter<f32> for OneEuroFilter {
    fn filter(&self, state: &mut Self::State, x: f32, elapsed: f32) -> f32 {
        match &mut state.prev {
            None => {
                state.prev = Some(Prev { x, dx: 0.0 });
                x
            }
            Some(prev) => {
                let a_d = smoothing_factor(elapsed, self.d_cutoff);
                let dx = (x - prev.x) / elapsed;
                let dx_hat = exponential_smoothing(a_d, dx, prev.dx);

                let cutoff = self.min_cutoff + self.beta * dx_hat.abs();
                let a = smoothing_factor(elapsed, cutoff);
                let x_hat = exponential_smoothing(a, x, prev.x);

                prev.x = x_hat;
                prev.dx = dx_hat;

                x_hat
            }
        }
    }
}

fn smoothing_factor(t_e: f32, cutoff: f32) -> f32 {
    let r = 2.0 * PI * cutoff * t_e;
    r / (r + 1.0)
}

fn exponential_smoothing(a: f32, x: f32, x_prev: f32) -> f32 {
    a * x + (1.0 - a) * x_prev
}
