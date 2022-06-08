//! Data filtering, averaging and smoothing.
//!
//! This module provides a generic interface for data filtering algorithms, as well as several
//! filter implementations.
//!
//! # Filter Parameters and State
//!
//! The filter interface splits filters into *parameters* and *state*. Filter *parameters* tune the
//! filter's behavior, are independent of the filtered data, are typically constant over the
//! lifetime of a filter, and can be applied to any number of filtered variables. Filter *state*
//! changes based on the filtered data, is specific to a filtered variable, and has a non-tunable
//! default value.
//!
//! This split allows using a single set of parameters to filter a large number of variables,
//! without duplicating the parameters for each one. For convenience, [`SimpleFilter`] can be used
//! to manage filter state when only a single variable is being filtered (or when parameter
//! duplication just doesn't matter much).
//!
//! # Time-based filtering
//!
//! Some filtering algorithms are explicitly designed to account for the time between measurements,
//! while some aren't. Algorithms that aren't will implement [`Filter`], while those that are will
//! implement [`TimeBasedFilter`].
//!
//! The interface of [`TimeBasedFilter`] takes the time since the previous measurement as an
//! argument. This allows users to pass custom time deltas for unit testing, data replay and other
//! use cases. In the common case where the real world time should be used, [`TimedFilterAdapter`]
//! can be used to adapt a [`TimeBasedFilter`] to the [`Filter`] trait.

use std::time::Instant;

pub mod alpha_beta;
pub mod ema;

/// Base trait for filtering algorithms that defines the per-variable state of the filter.
///
/// The `V` type parameter is the type of value that this filter operates on (typically `f32` for
/// basic filter algorithms).
pub trait FilterBase<V> {
    /// Per-variable filter state.
    type State: Default;
}

/// Trait implemented for filter algorithms operating on data of type `V`.
///
/// The implementing type is expected to carry all filter parameters with it, while any per-variable
/// state that needs updating is passed as an argument of type [`FilterBase::State`].
pub trait Filter<V>: FilterBase<V> {
    /// Filters `value` according to the filter parameters stored in `self` and the current filter
    /// state in `state`.
    ///
    /// `state` is updated according to the filter's equation to reflect the addition of `value`.
    ///
    /// The filtered value is returned.
    fn filter(&self, state: &mut Self::State, value: V) -> V;
}

/// Trait for filter algorithms that take the time difference between measurements into account.
///
/// This trait mostly works just like [`Filter`], except that the [`TimeBasedFilter::filter`] method
/// takes the time delta since the last measurement as an additional argument.
///
/// To just use a time-based filter with real-world time stamps, the [`TimedFilterAdapter`] type can
/// be used.
pub trait TimeBasedFilter<V>: FilterBase<V> {
    /// Filters `value` according to the filter parameters stored in `self` and the current filter
    /// state in `state`.
    ///
    /// `state` is updated according to the filter's equation to reflect the addition of `value`.
    ///
    /// `elapsed` is the time in seconds since the previous value was fed into the filter. For the
    /// first value, an `elapsed` time delta of 0.0 should be passed.
    ///
    /// The filtered value is returned.
    fn filter(&self, state: &mut Self::State, value: V, elapsed: f32) -> V;

    /// Wraps `self` in a [`TimedFilterAdapter`], returning a [`Filter`] implementation that uses
    /// the elapsed real-world time.
    fn real_time(self) -> TimedFilterAdapter<Self>
    where
        Self: Sized,
    {
        TimedFilterAdapter::new(self)
    }
}

/// Adapts a [`TimeBasedFilter`] to the [`Filter`] trait by supplying time deltas derived from the
/// current system time.
pub struct TimedFilterAdapter<F> {
    filter: F,
    last: Instant,
}

impl<F> TimedFilterAdapter<F> {
    pub fn new(filter: F) -> Self {
        Self {
            filter,
            last: Instant::now(),
        }
    }
}

impl<F: FilterBase<V>, V> FilterBase<V> for TimedFilterAdapter<F> {
    type State = F::State;
}

impl<F: TimeBasedFilter<V>, V> Filter<V> for TimedFilterAdapter<F> {
    fn filter(&self, state: &mut Self::State, value: V) -> V {
        let elapsed = self.last.elapsed();
        self.filter.filter(state, value, elapsed.as_secs_f32())
    }
}

/// A [`Filter`] wrapper for filtering a single variable.
#[derive(Debug)]
pub struct SimpleFilter<A: Filter<V>, V> {
    params: A,
    state: A::State,
}

impl<A: Filter<V>, V> SimpleFilter<A, V> {
    /// Creates a new filter from a set of filter parameters.
    pub fn new(params: A) -> Self {
        Self {
            params,
            state: Default::default(),
        }
    }

    /// Passes `value` through the filter, updates its internal state, and returns the filtered
    /// value.
    pub fn filter(&mut self, value: V) -> V {
        self.params.filter(&mut self.state, value)
    }

    /// Sets the filter parameters to use.
    ///
    /// Note that this does not reset the filter state, which might be advisable after changing the
    /// parameters.
    pub fn set_params(&mut self, params: A) {
        self.params = params;
    }

    /// Resets the filter state to its default, without affecting the filter parameters.
    pub fn reset_state(&mut self) {
        self.state = Default::default();
    }
}

/// A filter that passes filtered values through as-is.
#[derive(Default, Clone, Copy)]
pub struct NoopFilter {
    _p: (),
}

impl NoopFilter {
    pub fn new() -> Self {
        Self { _p: () }
    }
}

/// State associated with a [`NoopFilter`] instance.
#[derive(Default, Clone, Copy)]
pub struct NoopFilterState {
    _p: (),
}

impl<V> FilterBase<V> for NoopFilter {
    type State = NoopFilterState;
}

impl<V> Filter<V> for NoopFilter {
    fn filter(&self, _state: &mut Self::State, value: V) -> V {
        value
    }
}

impl<V> TimeBasedFilter<V> for NoopFilter {
    fn filter(&self, _state: &mut Self::State, value: V, _elapsed: f32) -> V {
        value
    }
}
