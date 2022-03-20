//! Data filtering, averaging and smoothing.

mod moving;
mod variance_aware;

pub use moving::{Ema, MovingAvg};
pub use variance_aware::VarianceAwareAvg;

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
