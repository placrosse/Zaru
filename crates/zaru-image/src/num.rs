//! Utilities for numerics.

use std::cmp::Ordering;

/// An `f32` that implements [`Ord`] according to the IEEE 754 totalOrder predicate.
#[derive(Clone, Copy)]
pub struct TotalF32(pub f32);

impl PartialEq for TotalF32 {
    fn eq(&self, other: &Self) -> bool {
        f32::total_cmp(&self.0, &other.0) == Ordering::Equal
    }
}

impl Eq for TotalF32 {}

impl PartialOrd for TotalF32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for TotalF32 {
    fn cmp(&self, other: &Self) -> Ordering {
        f32::total_cmp(&self.0, &other.0)
    }
}

/// Applies the standard sigmoid/logistic function to the input.
pub fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}
