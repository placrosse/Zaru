//! Iterator extension methods.

use std::iter::Zip;

/// A variant of [`Iterator::zip`] that panics if the iterators have different lengths.
///
/// Since [`Iterator::zip`] stops yielding items when either of the two iterators is exhausted, it
/// can lead to bugs when using it to copy or merge data and the iterators are created incorrectly.
/// This function can be used when the iterators are expected to always have equal lengths to avoid
/// bugs like that.
#[track_caller]
pub fn zip_exact<A, B>(a: A, b: B) -> Zip<A::IntoIter, B::IntoIter>
where
    A: IntoIterator,
    B: IntoIterator,
    A::IntoIter: ExactSizeIterator,
    B::IntoIter: ExactSizeIterator,
{
    let a = a.into_iter();
    let b = b.into_iter();
    assert_eq!(
        a.len(),
        b.len(),
        "`zip_exact` called on iterators with different lengths"
    );

    a.zip(b)
}
