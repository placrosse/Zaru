//! Utilities for numerics.

pub use zaru_image::num::TotalF32;

/// Applies the standard sigmoid/logistic function to the input.
pub fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}
