//! Data filtering, averaging and smoothing.

mod alpha_beta;
mod moving;

pub use alpha_beta::AlphaBetaFilter;
pub use moving::{Ema, MovingAvg};

/// A filter for values of type `V`.
pub trait Filter<V> {
    /// Adds a new value to the filter, returning the filtered value.
    fn push(&mut self, value: V) -> V;

    /// Resets the accumulated history and state of the filter to be identical to the state just
    /// after construction.
    fn reset(&mut self);
}

impl<V> Filter<V> for Box<dyn Filter<V>> {
    fn push(&mut self, value: V) -> V {
        (**self).push(value)
    }

    fn reset(&mut self) {
        (**self).reset();
    }
}
