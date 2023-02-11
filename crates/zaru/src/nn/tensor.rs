//! Tensor API.
//!
//! Tensors are the inputs and outputs of neural networks. They can encode image data, text, or
//! other inputs and outputs, though Zaru deals mainly with images at the moment.
//!
//! A Tensor is nothing more than an N-dimensional array. The term *Tensor* is mathematical and
//! describes a generalization of scalars, vectors, and matrices with an arbitrary number of
//! dimensions.

use std::fmt;

use crate::iter::zip_exact;
use tinyvec::TinyVec;

#[derive(Clone)]
struct Layout(TinyVec<[usize; 8]>);

impl Layout {
    fn from_shape(shape: &[usize]) -> Self {
        let mut vec = TinyVec::from(shape);
        vec.extend(shape.iter().map(|_| 0));

        // compute strides
        let mut stride = 1;
        for (out, size) in zip_exact(
            vec[shape.len()..].iter_mut().rev(),
            shape.iter().copied().rev(),
        ) {
            *out = stride;
            stride *= size;
        }

        Self(vec)
    }

    fn shape(&self) -> &[usize] {
        &self.0[..self.0.len() / 2]
    }

    fn elements(&self) -> usize {
        self.shape().iter().product()
    }

    fn strides(&self) -> &[usize] {
        &self.0[self.0.len() / 2..]
    }

    fn shape_and_strides(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.shape()
            .iter()
            .copied()
            .zip(self.strides().iter().copied())
    }

    fn remove_prefix(&self, num: usize) -> Layout {
        assert!(num <= self.shape().len());

        let mut vec = TinyVec::with_capacity(self.shape().len() - num);
        for &size in &self.shape()[num..] {
            vec.push(size);
        }
        for &stride in &self.strides()[num..] {
            vec.push(stride);
        }
        Layout(vec)
    }
}

impl fmt::Debug for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(&self.0).finish()
    }
}

struct ShapeIndices<S: AsRef<[usize]>, B: AsMut<[usize]>> {
    shape: S,
    last: B,
    first: bool,
}

impl<S: AsRef<[usize]>, B: AsMut<[usize]>> ShapeIndices<S, B> {
    fn new(shape: S, buf: B) -> Self {
        Self {
            shape,
            last: buf,
            first: true,
        }
    }
    fn next(&mut self) -> Option<&B> {
        if self.first {
            self.first = false;
            if self.shape.as_ref().iter().any(|&x| x == 0) {
                return None;
            } else {
                return Some(&self.last);
            }
        }

        let mut has_next = false;
        for (next, shape) in zip_exact(self.last.as_mut(), self.shape.as_ref()).rev() {
            if *next == *shape - 1 {
                *next = 0;
            } else {
                *next += 1;
                has_next = true;
                break;
            }
        }

        if has_next {
            Some(&self.last)
        } else {
            None
        }
    }
    fn fold<V, F>(mut self, init: V, mut f: F) -> V
    where
        F: FnMut(V, &B) -> V,
    {
        let mut val = init;
        while let Some(indices) = self.next() {
            val = f(val, indices);
        }
        val
    }
}

/// A dynamically sized tensor.
///
/// Currently, Zaru's tensors always use `f32` as the element type, since that is what the neural
/// networks I'm interested in use.
///
/// # Construction
///
/// A tensor can either be created via the provided `From` impls (from singular values and
/// 1-dimensional arrays and slices), or by calling one of the `from_*` constructor methods.
///
/// # Data Access
///
/// Through [`Tensor::index`] it is possible to create a [`TensorView`] that refers to a specific
/// suffix of a tensor. Additionally, the data in one-dimensional tensors and views can be accessed
/// by calling [`Tensor::as_slice`] or [`TensorView::as_slice`], respectively, and the data in
/// zero-dimentional tensors and views can be accessed by calling [`Tensor::as_singular`] and
/// [`TensorView::as_singular`], respectively.
///
/// There is also a [`Tensor::iter`] / [`TensorView::iter`] method for iterating over the sub-views
/// you get by indexing into the outermost dimension.
///
/// Together, these mechanisms allow you to access any data in a tensor.
#[derive(Clone)]
pub struct Tensor {
    layout: Layout,
    data: Box<[f32]>,
}

/// A borrowed view into a [`Tensor`].
#[derive(Clone)]
pub struct TensorView<'a> {
    layout: Layout,
    data: &'a [f32],
}

impl Tensor {
    /// Creates an `N`-dimensional tensor of the given shape by calling `f` for each element.
    ///
    /// This will invoke `f` with successive indices to fill, starting with `[0, ..., 0, 0]`, then
    /// `[0, ..., 0, 1]` and so on. `f` can choose to use or ignore the index vector.
    pub fn from_array_shape_fn<const N: usize, F: FnMut([usize; N]) -> f32>(
        shape: [usize; N],
        mut f: F,
    ) -> Self {
        let indices = ShapeIndices::new(shape, [0; N]);
        let mut data = Vec::with_capacity(shape.iter().product());
        indices.fold((), |(), indices| data.push(f(*indices)));
        Self {
            layout: Layout::from_shape(&shape),
            data: data.into_boxed_slice(),
        }
    }

    /// Creates a tensor with a dynamic number of dimensions.
    pub fn from_dyn_shape_fn<F: FnMut(&[usize]) -> f32>(shape: &[usize], mut f: F) -> Self {
        let layout = Layout::from_shape(shape);
        let buf = vec![0; shape.len()];
        let indices = ShapeIndices::new(shape, buf);
        let mut data = Vec::with_capacity(layout.elements());
        indices.fold((), |(), indices| data.push(f(indices)));
        Self {
            layout,
            data: data.into_boxed_slice(),
        }
    }

    /// Creates a tensor of the given shape by pulling elements from an iterator.
    ///
    /// # Panics
    ///
    /// `iter` must yield exactly as many elements as specified by `shape` (by multiplying all of
    /// its entries), otherwise this method will panic.
    pub fn from_iter<I: IntoIterator<Item = f32>>(shape: &[usize], iter: I) -> Self {
        let layout = Layout::from_shape(shape);
        let data: Box<_> = iter.into_iter().collect();
        assert_eq!(data.len(), layout.elements());
        Self { layout, data }
    }

    pub(super) fn from_tract(tract: &tract_onnx::prelude::Tensor) -> Self {
        let mut iter = tract.as_slice::<f32>().unwrap().iter();
        Self::from_dyn_shape_fn(tract.shape(), |_| *iter.next().unwrap())
    }

    pub(super) fn to_tract(&self) -> tract_onnx::prelude::Tensor {
        tract_onnx::prelude::Tensor::from_shape(self.shape(), &self.data).unwrap()
    }

    pub(super) fn as_raw_data(&self) -> &[f32] {
        &self.data
    }

    /// Returns the shape of this tensor.
    ///
    /// A tensor's shape is the number of entries in each dimension.
    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    /// Returns the number of dimensions of this tensor.
    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    /// Indexes a prefix of the tensor's dimensions with `indices`.
    ///
    /// For an example, consider a tensor of shape `[2, 3, 4, 5]`. Indexing it with 2 indices
    /// `[a, b]` will return a view of shape `[4, 5]`, while indexing it with 4 indices
    /// `[a, b, c, d]` will return a view of shape `[]` (aka a single value).
    ///
    /// Indexing a tensor with zero indices (`[]`) is also permitted and will return a view of the
    /// same shape as the tensor.
    ///
    /// # Panics
    ///
    /// This method will panic if `indices` has more entries than `self` has dimensions, or if any
    /// index is out of bounds.
    #[track_caller]
    pub fn index<const N: usize>(&self, indices: [usize; N]) -> TensorView<'_> {
        assert!(
            N <= self.rank(),
            "attempted to index tensor of shape {:?} with {:?}",
            self.shape(),
            indices
        );

        let mut data = &*self.data;
        for ((length, stride), index) in self.layout.shape_and_strides().zip(indices) {
            assert!(
                index < length,
                "attempted to index tensor of shape {:?} with {:?}",
                self.shape(),
                indices
            );
            data = &data[index * stride..(index + 1) * stride];
        }
        TensorView {
            layout: self.layout.remove_prefix(indices.len()),
            data,
        }
    }

    /// Iterates over the outermost dimension of this tensor.
    ///
    /// For example, iterating over the outermost dimension of a tensor with shape `[3, 4, 5]` will
    /// yield 3 [`TensorView`]s of shape `[4, 5]`.
    ///
    /// # Panics
    ///
    /// `self` must have at least one dimension, otherwise this method will panic.
    #[track_caller]
    pub fn iter(&self) -> impl Iterator<Item = TensorView<'_>> {
        assert!(
            self.rank() > 0,
            "attempted to iterate over 0-dimensional tensor"
        );
        (0..self.shape()[0]).map(|index| self.index([index]))
    }

    /// Returns the values stored in a 1-dimensional tensor as a slice.
    ///
    /// # Panics
    ///
    /// `self` must have exactly 1 dimension, otherwise this method panics.
    #[track_caller]
    pub fn as_slice(&self) -> &[f32] {
        assert_eq!(
            self.rank(),
            1,
            "attempted to access tensor of shape {:?} as slice",
            self.shape()
        );
        &self.data
    }

    /// Returns the value stored in a 0-dimensional tensor.
    ///
    /// # Panics
    ///
    /// `self` must have exactly 0 dimensions, otherwise this method will panic.
    #[track_caller]
    pub fn as_singular(&self) -> f32 {
        assert_eq!(
            self.rank(),
            0,
            "attempted to access tensor of shape {:?} as singular element",
            self.shape(),
        );
        self.data[0]
    }
}

impl From<f32> for Tensor {
    fn from(value: f32) -> Self {
        Tensor::from_array_shape_fn([], |[]| value)
    }
}

impl<'a> From<&'a [f32]> for Tensor {
    fn from(slice: &'a [f32]) -> Self {
        Tensor::from_array_shape_fn([slice.len()], |[i]| slice[i])
    }
}

impl<const N: usize> From<[f32; N]> for Tensor {
    fn from(arr: [f32; N]) -> Self {
        Tensor::from_array_shape_fn([N], |[i]| arr[i])
    }
}

impl<'a, const N: usize> From<&'a [f32; N]> for Tensor {
    fn from(arr: &'a [f32; N]) -> Self {
        Tensor::from_array_shape_fn([N], |[i]| arr[i])
    }
}

impl<'d> TensorView<'d> {
    /// Returns the shape of this tensor view.
    ///
    /// The shape is the number of entries in each dimension.
    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    /// Returns the number of dimensions of this tensor.
    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    /// Indexes a prefix of the tensor view's dimensions with `indices`.
    ///
    /// For an example, consider a tensor view of shape `[2, 3, 4, 5]`. Indexing it with 2 indices
    /// `[a, b]` will return a view of shape `[4, 5]`, while indexing it with 4 indices
    /// `[a, b, c, d]` will return a view of shape `[]` (aka a single value).
    ///
    /// Indexing a tensor view with zero indices (`[]`) is also permitted and will return a copy of
    /// the view.
    ///
    /// # Panics
    ///
    /// This method will panic if `indices` has more entries than `self` has dimensions, or if any
    /// index is out of bounds.
    #[track_caller]
    pub fn index<const N: usize>(&self, indices: [usize; N]) -> TensorView<'d> {
        assert!(
            N <= self.rank(),
            "attempted to index tensor view of shape {:?} with {:?}",
            self.shape(),
            indices
        );
        let mut data = self.data;
        for (&stride, &index) in self.layout.strides().iter().zip(&indices) {
            data = &data[index * stride..(index + 1) * stride];
        }
        TensorView {
            layout: self.layout.remove_prefix(indices.len()),
            data,
        }
    }

    /// Iterates over the outermost dimension of this tensor view.
    ///
    /// For example, iterating over the outermost dimension of a tensor view with shape `[3, 4, 5]`
    /// will yield 3 [`TensorView`]s of shape `[4, 5]`.
    ///
    /// # Panics
    ///
    /// `self` must have at least one dimension, otherwise this method will panic.
    #[track_caller]
    pub fn iter(&self) -> impl Iterator<Item = TensorView<'_>> {
        assert!(
            self.rank() > 0,
            "attempted to iterate over 0-dimensional tensor view"
        );
        (0..self.shape()[0]).map(|index| self.index([index]))
    }

    /// Returns the values stored in a 1-dimensional view as a slice.
    ///
    /// # Panics
    ///
    /// `self` must have exactly 1 dimension, otherwise this method panics.
    #[track_caller]
    pub fn as_slice(&self) -> &[f32] {
        assert_eq!(
            self.rank(),
            1,
            "attempted to access tensor view of shape {:?} as slice",
            self.shape()
        );
        self.data
    }

    /// Returns the value stored in a 0-dimensional tensor.
    ///
    /// # Panics
    ///
    /// `self` must have exactly 0 dimensions, otherwise this method will panic.
    #[track_caller]
    pub fn as_singular(&self) -> f32 {
        assert_eq!(
            self.rank(),
            0,
            "attempted to access view of shape {:?} as singular element",
            self.shape(),
        );
        self.data[0]
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // TODO improve
        f.debug_struct("Tensor")
            .field("shape", &self.shape())
            .finish()
    }
}

impl fmt::Debug for TensorView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // TODO improve
        f.debug_struct("TensorView")
            .field("shape", &self.shape())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `as &[usize]` casts courtesy of using `serde_json` as a transitive dependency and the fact
    // that Rust has some utterly devastating design flaws that are nothing but embarrassing.

    #[test]
    fn from_shape_fn() {
        let indices = [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 2],
        ];

        let mut iter = indices.into_iter();
        let tensor = Tensor::from_array_shape_fn([1, 2, 3], |index| {
            assert_eq!(iter.next(), Some(index));
            0.0
        });
        assert_eq!(tensor.rank(), 3);
        assert_eq!(tensor.shape(), &[1, 2, 3]);

        let mut iter = indices.into_iter();
        let tensor = Tensor::from_dyn_shape_fn(&[1, 2, 3], |index| {
            assert_eq!(iter.next().as_ref().map(|arr| &arr[..]), Some(index));
            0.0
        });
        assert_eq!(tensor.rank(), 3);
        assert_eq!(tensor.shape(), &[1, 2, 3]);
    }

    #[test]
    fn empty() {
        let tensor = Tensor::from_array_shape_fn([1, 2, 0, 3], |idx| unreachable!("{idx:?}"));
        assert_eq!(tensor.shape(), &[1, 2, 0, 3]);
        assert_eq!(tensor.iter().count(), 1);

        let view = tensor.index([0, 1]);
        assert_eq!(view.shape(), &[0, 3]);
        assert_eq!(view.iter().count(), 0);
        view.index([]);
    }

    #[test]
    fn singular() {
        // "What if f32, but obtuse"?

        let mut hits = 0;
        let tensor = Tensor::from_array_shape_fn([], |[]| {
            hits += 1;
            1.0
        });
        assert_eq!(hits, 1);
        assert_eq!(tensor.shape(), &[] as &[usize]);
        assert_eq!(tensor.rank(), 0);
        assert_eq!(tensor.as_singular(), 1.0);

        let view = tensor.index([]);
        assert_eq!(view.shape(), &[] as &[usize]);
        assert_eq!(view.rank(), 0);
        assert_eq!(view.as_singular(), 1.0);
    }

    #[test]
    fn index_shapes() {
        let tensor = Tensor::from_array_shape_fn([1, 1, 1], |_| 1.0);
        assert_eq!(tensor.shape(), &[1, 1, 1]);
        assert_eq!(tensor.rank(), 3);

        let view2d = tensor.index([0]);
        assert_eq!(view2d.shape(), &[1, 1]);
        assert_eq!(view2d.rank(), 2);

        let view1d = view2d.index([0]);
        assert_eq!(view1d.shape(), &[1]);
        assert_eq!(view1d.rank(), 1);
        assert_eq!(view1d.as_slice(), &[1.0]);

        let view0d = view1d.index([0]);
        assert_eq!(view0d.shape(), &[] as &[usize]);
        assert_eq!(view0d.rank(), 0);
        assert_eq!(view0d.as_singular(), 1.0);

        // same view but from the tensor directly
        let view1d = tensor.index([0, 0]);
        assert_eq!(view1d.shape(), &[1]);
        assert_eq!(view1d.rank(), 1);
        assert_eq!(view1d.as_slice(), &[1.0]);

        let view0d = view1d.index([0]);
        assert_eq!(view0d.shape(), &[] as &[usize]);
        assert_eq!(view0d.rank(), 0);
        assert_eq!(view0d.as_singular(), 1.0);
    }

    #[test]
    fn index_1d_elems() {
        let array = Tensor::from([0.0, 1.0, 2.0]);
        assert_eq!(array.shape(), &[3]);
        assert_eq!(array.as_slice(), &[0.0, 1.0, 2.0]);
        let same_array = array.index([]);
        assert_eq!(same_array.shape(), &[3]);
        assert_eq!(same_array.as_slice(), &[0.0, 1.0, 2.0]);

        let first = array.index([0]);
        assert_eq!(first.shape(), &[] as &[usize]);
        assert_eq!(first.as_singular(), 0.0);
        let second = array.index([1]);
        assert_eq!(second.shape(), &[] as &[usize]);
        assert_eq!(second.as_singular(), 1.0);
        let third = array.index([2]);
        assert_eq!(third.shape(), &[] as &[usize]);
        assert_eq!(third.as_singular(), 2.0);
    }

    #[test]
    fn index_2d_elems() {
        let iter = [[0.0, 1.0], [2.0, 3.0]].into_iter().flatten();
        let tensor = Tensor::from_iter(&[2, 2], iter);
        assert_eq!(tensor.shape(), [2, 2]);

        let row0 = tensor.index([0]);
        assert_eq!(row0.shape(), [2]);
        assert_eq!(row0.as_slice(), [0.0, 1.0]);

        let row1 = tensor.index([1]);
        assert_eq!(row1.shape(), [2]);
        assert_eq!(row1.as_slice(), [2.0, 3.0]);

        assert_eq!(tensor.index([1, 1]).as_singular(), 3.0);
        assert_eq!(tensor.index([1, 0]).as_singular(), 2.0);
    }
}
