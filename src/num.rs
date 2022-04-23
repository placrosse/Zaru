//! Utilities for numerics.

use std::cmp::Ordering;

/// An `f32` that implements [`Ord`] according to the IEEE 754 totalOrder predicate.
#[derive(Clone, Copy)]
pub struct TotalF32(pub f32);

impl PartialEq for TotalF32 {
    fn eq(&self, other: &Self) -> bool {
        f32_total_cmp(self.0, other.0) == Ordering::Equal
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
        f32_total_cmp(self.0, other.0)
    }
}

// FIXME unstable, copied from stdlib
fn f32_total_cmp(a: f32, b: f32) -> Ordering {
    let mut left = a.to_bits() as i32;
    let mut right = b.to_bits() as i32;

    // In case of negatives, flip all the bits except the sign
    // to achieve a similar layout as two's complement integers
    //
    // Why does this work? IEEE 754 floats consist of three fields:
    // Sign bit, exponent and mantissa. The set of exponent and mantissa
    // fields as a whole have the property that their bitwise order is
    // equal to the numeric magnitude where the magnitude is defined.
    // The magnitude is not normally defined on NaN values, but
    // IEEE 754 totalOrder defines the NaN values also to follow the
    // bitwise order. This leads to order explained in the doc comment.
    // However, the representation of magnitude is the same for negative
    // and positive numbers – only the sign bit is different.
    // To easily compare the floats as signed integers, we need to
    // flip the exponent and mantissa bits in case of negative numbers.
    // We effectively convert the numbers to "two's complement" form.
    //
    // To do the flipping, we construct a mask and XOR against it.
    // We branchlessly calculate an "all-ones except for the sign bit"
    // mask from negative-signed values: right shifting sign-extends
    // the integer, so we "fill" the mask with sign bits, and then
    // convert to unsigned to push one more zero bit.
    // On positive values, the mask is all zeros, so it's a no-op.
    left ^= (((left >> 31) as u32) >> 1) as i32;
    right ^= (((right >> 31) as u32) >> 1) as i32;

    left.cmp(&right)
}
