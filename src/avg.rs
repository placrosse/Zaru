//! Data averaging and smoothing.

mod moving;
mod variance_aware;

pub use moving::{Ema, MovingAvg};
pub use variance_aware::VarianceAwareAvg;

/// Trait for types that compute an average over values of type `V`.
pub trait Averager<V> {
    /// Adds a new value to the averager, returning the resulting average.
    fn push(&mut self, value: V) -> V;

    /// Resets the accumulated history and state of the averager to be identical to the state just
    /// after construction.
    fn reset(&mut self);
}

impl<V> Averager<V> for Box<dyn Averager<V>> {
    fn push(&mut self, value: V) -> V {
        (**self).push(value)
    }

    fn reset(&mut self) {
        (**self).reset();
    }
}
