//! Slice extension methods.

/// Extensions for immutable slices.
pub trait SliceExt<'a, T> {
    /// Returns an iterator that yields array chunks of `N` elements.
    fn array_chunks_exact<const N: usize>(self) -> ArrayChunksExact<'a, N, T>;
}

impl<'a, T> SliceExt<'a, T> for &'a [T] {
    fn array_chunks_exact<const N: usize>(self) -> ArrayChunksExact<'a, N, T> {
        assert_ne!(N, 0);
        assert_eq!(self.len() % N, 0);
        ArrayChunksExact { remainder: self }
    }
}

pub struct ArrayChunksExact<'a, const N: usize, T> {
    remainder: &'a [T],
}

impl<'a, const N: usize, T> Iterator for ArrayChunksExact<'a, N, T> {
    type Item = &'a [T; N];

    fn next(&mut self) -> Option<Self::Item> {
        if self.remainder.is_empty() {
            None
        } else {
            let item = self.remainder[..N].try_into().unwrap();
            self.remainder = &self.remainder[N..];
            Some(item)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let chunks = self.remainder.len() / N;
        (chunks, Some(chunks))
    }
}

impl<'a, const N: usize, T> ExactSizeIterator for ArrayChunksExact<'a, N, T> {}
