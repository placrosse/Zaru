use std::{
    array, fmt,
    mem::{self, ManuallyDrop, MaybeUninit},
};

use crate::{Number, One, Trig, Vector, Zero};

mod ops;

/// A 1x1 matrix.
pub type Mat1<T> = Matrix<T, 1, 1>;
/// A 1x1 matrix with [`f32`] elements.
pub type Mat1f = Mat1<f32>;
/// A 2x2 matrix.
pub type Mat2<T> = Matrix<T, 2, 2>;
/// A 2x2 matrix with [`f32`] elements.
pub type Mat2f = Mat2<f32>;
/// A 3x3 matrix.
pub type Mat3<T> = Matrix<T, 3, 3>;
/// A 3x3 matrix with [`f32`] elements.
pub type Mat3f = Mat3<f32>;
/// A 4x4 matrix.
pub type Mat4<T> = Matrix<T, 4, 4>;
/// A 4x4 matrix with [`f32`] elements.
pub type Mat4f = Mat4<f32>;

/// A matrix with 2 rows and 3 columns.
pub type Mat2x3<T> = Matrix<T, 2, 3>;
/// A matrix with 2 rows and 4 columns.
pub type Mat2x4<T> = Matrix<T, 2, 4>;
/// A matrix with 3 rows and 2 columns.
pub type Mat3x2<T> = Matrix<T, 3, 2>;
/// A matrix with 3 rows and 4 columns.
pub type Mat3x4<T> = Matrix<T, 3, 4>;
/// A matrix with 4 rows and 2 columns.
pub type Mat4x2<T> = Matrix<T, 4, 2>;
/// A matrix with 4 rows and 3 columns.
pub type Mat4x3<T> = Matrix<T, 4, 3>;

/// A column-major matrix with `R` rows and `C` columns, and element type `T`.
///
/// # Construction
///
/// There are several ways to create a [`Matrix`]:
///
/// - [`Matrix::from_rows`] and [`Matrix::from_columns`] allow filling a matrix with raw elements,
///   as well as creating them from an array of row or column vectors.
/// - [`Matrix::from_fn`] will create each element by invoking a closure with its row and column.
/// - For square matrices (where `R` equals `C`), [`Matrix::from_diagonal`] can be used to create a
///   matrix with a specified diagonal and zero outside of its diagonal.
/// - [`Matrix::rotation_clockwise`] and [`Matrix::rotation_counterclockwise`] allow creating 2D
///   rotation matrices from a rotation angle.
///
/// Additionally, some associated constants for commonly used matrices are defined:
///
/// - [`Matrix::ZERO`] is a matrix with every element set to 0.
/// - [`Matrix::IDENTITY`] is a square matrix with 1 on its diagonal and 0 everywhere else.
///
/// # Element Access
///
/// [`Matrix`] implements the [`Index`] and [`IndexMut`] traits for tuples of `(usize, usize)`. The
/// first element of the tuple is the *row* (Y coordinate), the second is the *column* (X
/// coordinate), matching common mathematical notation. Indices are 0-based.
///
/// ```
/// # use zaru_linalg::*;
/// let mut mat = Matrix::from_rows([
///     [0, 1]
/// ]);
/// mat[(0, 0)] = 4;
/// assert_eq!(mat[(0, 0)], 4);
/// assert_eq!(mat[(0, 1)], 1);
/// ```
///
/// Indexing out of bounds will result in a panic, just like it does for slices. [`Matrix::get`] and
/// [`Matrix::get_mut`] return [`Option`]s instead and can be used for checked indexing:
///
/// ```
/// # use zaru_linalg::*;
/// let mut mat = Matrix::from_rows([
///     [0, 1]
/// ]);
/// assert_eq!(mat.get(0, 0), Some(&0));
/// assert_eq!(mat.get(0, 1), Some(&1));
/// assert_eq!(mat.get(0, 2), None);
/// ```
///
/// [`Index`]: std::ops::Index
/// [`IndexMut`]: std::ops::IndexMut
#[derive(Clone, Copy, Hash)]
pub struct Matrix<T, const R: usize, const C: usize>([[T; R]; C]);

#[rustfmt::skip]
unsafe impl<T: bytemuck::Zeroable, const R: usize, const C: usize> bytemuck::Zeroable for Matrix<T, R, C> {}
unsafe impl<T: bytemuck::Pod, const R: usize, const C: usize> bytemuck::Pod for Matrix<T, R, C> {}

impl<T, const R: usize, const C: usize> Matrix<T, R, C> {
    /// The smallest dimension of the matrix (`R` or `C`).
    const MIN_DIMENSION: usize = if R > C { C } else { R };

    /// Creates a new [`Matrix`] in which the elements are wrapped in [`MaybeUninit`].
    const fn new_uninit() -> Matrix<MaybeUninit<T>, R, C> {
        // FIXME: make `pub` once libstd settles on how to do these
        // Safety: `uninit` is a valid value for the `MaybeUninit<T>` elements
        unsafe { MaybeUninit::<Matrix<MaybeUninit<T>, R, C>>::uninit().assume_init() }
    }

    /// Creates a [`Matrix`] from an array of row vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// let rows = Matrix::from_rows([
    ///     [0, 1],
    ///     [2, 3],
    /// ]);
    /// let columns = Matrix::from_columns([
    ///     [0, 2],
    ///     [1, 3],
    /// ]);
    /// assert_eq!(rows, columns);
    /// ```
    pub fn from_rows<U: Into<Vector<T, C>>>(rows: [U; R]) -> Self {
        Matrix::from_columns(rows).transpose()
    }

    /// Creates a [`Matrix`] from an array of column vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// let rows = Matrix::from_rows([
    ///     [0, 1],
    ///     [2, 3],
    /// ]);
    /// let columns = Matrix::from_columns([
    ///     [0, 2],
    ///     [1, 3],
    /// ]);
    /// assert_eq!(rows, columns);
    /// ```
    pub fn from_columns<U: Into<Vector<T, R>>>(columns: [U; C]) -> Self {
        Self(columns.map(|col| col.into().into_array()))
    }

    /// Creates a [`Matrix`] by invoking a closure with the position (row and column) of each element.
    ///
    /// This mirrors [`array::from_fn`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// let mat = Matrix::from_fn(|row, col| row * 10 + col);
    /// assert_eq!(mat, Matrix::from_rows([
    ///     [ 0,  1,  2],
    ///     [10, 11, 12],
    /// ]));
    /// ```
    pub fn from_fn<F>(mut cb: F) -> Self
    where
        F: FnMut(usize, usize) -> T,
    {
        Self(array::from_fn(|col| array::from_fn(|row| cb(row, col))))
    }

    /// Applies a closure to each element, returning a new matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// let mat = Matrix::from_rows([
    ///     [0, 1, 2],
    ///     [3, 4, 5],
    /// ]);
    /// let mat = mat.map(|i| i * 2);
    /// assert_eq!(mat, Matrix::from_rows([
    ///     [ 0,  2,  4],
    ///     [ 6,  8, 10],
    /// ]));
    /// ```
    pub fn map<F, U>(self, mut f: F) -> Matrix<U, R, C>
    where
        F: FnMut(T) -> U,
    {
        Matrix(self.0.map(|column| column.map(|v| f(v))))
    }

    /// Swaps the rows and columns of this matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// let mat = Matrix::from_rows([
    ///     [0, 1, 2],
    ///     [3, 4, 5],
    /// ]).transpose();
    /// assert_eq!(mat, Matrix::from_rows([
    ///     [0, 3],
    ///     [1, 4],
    ///     [2, 5],
    /// ]));
    /// ```
    pub fn transpose(self) -> Matrix<T, C, R> {
        let mut out = Matrix::<T, C, R>::new_uninit();
        for (c, column) in self.0.into_iter().enumerate() {
            for (r, elem) in column.into_iter().enumerate() {
                out.0[r][c] = MaybeUninit::new(elem);
            }
        }
        // Safety: the loop above writes to each element.
        unsafe { out.assume_init() }
    }

    /// Returns a reference to the element at `(row, col)`, or [`None`] if out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// let mat = Matrix::from_rows([
    ///     [0, 1, 2],
    ///     [3, 4, 5],
    /// ]);
    /// assert_eq!(mat.get(0, 0), Some(&0));
    /// assert_eq!(mat.get(1, 0), Some(&3));
    /// assert_eq!(mat.get(2, 0), None);
    /// ```
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.0.get(col).and_then(|col| col.get(row))
    }

    /// Returns a mutable reference to the element at `(row, col)`, or [`None`] if out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// let mut mat = Matrix::from_rows([
    ///     [0, 1, 2],
    ///     [3, 4, 5],
    /// ]);
    /// if let Some(elem) = mat.get_mut(1, 0) {
    ///     *elem = 999;
    /// }
    /// if let Some(elem) = mat.get_mut(2, 0) {
    ///     *elem = 777;
    /// }
    /// assert_eq!(mat, Matrix::from_rows([
    ///     [0, 1, 2],
    ///     [999, 4, 5],
    /// ]));
    /// ```
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        self.0.get_mut(col).and_then(|col| col.get_mut(row))
    }

    /// Returns a matrix with the contents of `self`, but a potentially different size.
    ///
    /// Elements not present in `self` will be initialized with [`T::ZERO`][`Zero::ZERO`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// let mat = Matrix::from_rows([
    ///     [1, 2, 3],
    /// ]);
    /// let resized = mat.resize::<2, 2>();
    /// assert_eq!(resized, Matrix::from_rows([
    ///     [1, 2],
    ///     [0, 0],
    /// ]));
    /// ```
    pub fn resize<const R2: usize, const C2: usize>(mut self) -> Matrix<T, R2, C2>
    where
        T: Zero, // FIXME: `T: Zero` or `T: Default`?
    {
        Matrix::from_fn(|row, col| {
            if col < C && row < R {
                mem::replace(&mut self[(row, col)], T::ZERO)
            } else {
                T::ZERO
            }
        })
    }

    /// Returns `self`, but with the element at `(row, col)` replaced with `elem`, without dropping
    /// the old element at that position.
    const fn with_leaky_elem(self, row: usize, col: usize, elem: T) -> Self {
        unsafe {
            // Leaks whatever was at `(col,row)` before.
            union UnWrapper<T, const R: usize, const C: usize> {
                wrapped: ManuallyDrop<Matrix<ManuallyDrop<T>, R, C>>,
                unwrapped: ManuallyDrop<Matrix<T, R, C>>,
            }

            let mut wrapped = ManuallyDrop::into_inner(
                UnWrapper {
                    unwrapped: ManuallyDrop::new(self),
                }
                .wrapped,
            );
            wrapped.0[col][row] = ManuallyDrop::new(elem);

            ManuallyDrop::into_inner(
                UnWrapper {
                    wrapped: ManuallyDrop::new(wrapped),
                }
                .unwrapped,
            )
        }
    }
}

impl<T: fmt::Debug, const R: usize, const C: usize> fmt::Debug for Matrix<T, R, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct FormatRow<'a, T: fmt::Debug, const R: usize, const C: usize>(
            &'a Matrix<T, R, C>,
            usize,
        );
        impl<'a, T: fmt::Debug, const R: usize, const C: usize> fmt::Debug for FormatRow<'a, T, R, C> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "[")?;
                for col in 0..C {
                    if col != 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:?}", self.0[(self.1, col)])?;
                }
                write!(f, "]")?;
                Ok(())
            }
        }

        let mut list = f.debug_list();
        for row in 0..R {
            list.entry(&FormatRow(self, row));
        }
        list.finish()
    }
}

impl<T: Zero, const R: usize, const C: usize> Matrix<T, R, C> {
    /// A matrix with every element set to 0.
    pub const ZERO: Self = unsafe {
        // Because `[T::ZERO; N]` requires `T` to be `Copy`, we use this gross hack to duplicate
        // `T::ZERO` without that `Copy` bound.
        let mut mat = Self::new_uninit();
        let mut col = 0;
        while col < C {
            let mut row = 0;
            while row < R {
                mat.0[col][row] = MaybeUninit::new(T::ZERO);
                row += 1;
            }
            col += 1;
        }

        // Safety: the loop above has initialized every element.
        mat.assume_init()
    };
}

impl<T, const R: usize, const C: usize> Matrix<MaybeUninit<T>, R, C> {
    /// Removes the [`MaybeUninit`] wrapper from each matrix element.
    ///
    /// See [`MaybeUninit::assume_init`] for details about the safety invariant the caller needs to
    /// uphold.
    const unsafe fn assume_init(self) -> Matrix<T, R, C> {
        // FIXME: make `pub` after libstd figures out how to do these types of functions

        // Safety: `MaybeUninit<T>` and `T` have the same layout.
        union UnWrapper<T, const R: usize, const C: usize> {
            uninit: ManuallyDrop<Matrix<MaybeUninit<T>, R, C>>,
            init: ManuallyDrop<Matrix<T, R, C>>,
        }

        ManuallyDrop::into_inner(
            UnWrapper {
                uninit: ManuallyDrop::new(self),
            }
            .init,
        )
    }
}

impl<T: Zero + One, const R: usize, const C: usize> Matrix<T, R, C> {
    /// The identity matrix.
    ///
    /// The matrix has the value 1 on its diagonal and 0 everywhere else.
    ///
    /// Multiplying any vector with this matrix returns the vector unchanged.
    pub const IDENTITY: Self = {
        let mut this = Self::ZERO;
        let mut i = 0;
        while i < Self::MIN_DIMENSION {
            this = this.with_leaky_elem(i, i, T::ONE);
            i += 1;
        }
        this
    };
}

impl<T, const N: usize> Matrix<T, N, N> {
    /// Returns a [`Vector`] holding the diagonal elements of this square matrix.
    ///
    /// *Note*: This method is restricted to square matrices due to limitations in Rust's const
    /// generics.
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// let mat = Matrix::from_rows([
    ///     [1, 2],
    ///     [3, 4],
    /// ]);
    /// assert_eq!(mat.into_diagonal(), [1, 4]);
    /// ```
    pub fn into_diagonal(self) -> Vector<T, N>
    where
        T: Copy,
    {
        array::from_fn(|i| self[(i, i)]).into()
    }

    /// Creates a square matrix from its diagonal.
    ///
    /// Elements outside the diagonal will be initialized with zero.
    ///
    /// *Note*: This method is intentionally restricted to square matrices to allow type inference
    /// of the created [`Matrix`]. To create a non-square matrix from its diagonal, use
    /// [`Matrix::from_fn`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// let diag = Matrix::from_diagonal([1, 2, 3]);
    /// assert_eq!(diag, Matrix::from_rows([
    ///     [1, 0, 0],
    ///     [0, 2, 0],
    ///     [0, 0, 3],
    /// ]));
    /// ```
    pub fn from_diagonal<D: Into<Vector<T, N>>>(diag: D) -> Self
    where
        T: Zero,
    {
        let mut iter = diag.into().into_array().into_iter();
        let mut this = Self::ZERO;
        for i in 0..N {
            this[(i, i)] = iter.next().unwrap();
        }
        this
    }

    /// Returns the *trace* of the matrix (the sum of all elements on the diagonal).
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// let diag = Matrix::from_diagonal([1, 2, 3]);
    /// assert_eq!(diag.trace(), 1 + 2 + 3);
    ///
    /// assert_eq!(Mat3f::IDENTITY.trace(), 3.0);
    /// ```
    pub fn trace(&self) -> T
    where
        T: Number,
    {
        (0..N).fold(T::ZERO, |acc, i| acc + self[(i, i)])
    }
}

// Determinant limited to 3x3 for now; keep bounds in sync!
impl<T: Number> Matrix<T, 1, 1> {
    /// Returns the [determinant] of the matrix.
    ///
    /// [determinant]: https://en.wikipedia.org/wiki/Determinant
    #[inline]
    pub fn determinant(&self) -> T {
        self[(0, 0)]
    }

    /// Inverts this 1x1 matrix.
    ///
    /// # Panics
    ///
    /// This method will panic if `self` is not invertible (ie. if its [`determinant()`] is zero).
    ///
    /// [`determinant()`]: Self::determinant
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// assert_eq!(Mat1::<i32>::IDENTITY.invert(), Mat1::<i32>::IDENTITY);
    /// assert_eq!(Mat1f::IDENTITY.invert(), Mat1f::IDENTITY);
    /// ```
    pub fn invert(&self) -> Self {
        let det = self.determinant();
        if det == T::ZERO {
            panic!("attempt to invert a non-invertible matrix");
        }

        Matrix::from_columns([[T::ONE / self[(0, 0)]]])
    }
}

impl<T: Number> Matrix<T, 2, 2> {
    /// Returns the [determinant] of the matrix.
    ///
    /// [determinant]: https://en.wikipedia.org/wiki/Determinant
    #[inline]
    pub fn determinant(&self) -> T {
        self[(0, 0)] * self[(1, 1)] - self[(0, 1)] * self[(1, 0)]
    }

    /// Inverts this 2x2 matrix.
    ///
    /// # Panics
    ///
    /// This method will panic if `self` is not invertible (ie. if its [`determinant()`] is zero).
    ///
    /// [`determinant()`]: Self::determinant
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// assert_eq!(Mat2::<i32>::IDENTITY.invert(), Mat2::<i32>::IDENTITY);
    /// assert_eq!(Mat2f::IDENTITY.invert(), Mat2f::IDENTITY);
    /// ```
    pub fn invert(&self) -> Self {
        let det = self.determinant();
        if det == T::ZERO {
            panic!("attempt to invert a non-invertible matrix");
        }

        let [[a, c], [b, d]] = self.0;
        Matrix::from_columns([[d, -c], [-b, a]]) * (T::ONE / det)
    }

    /// Creates a 2x2 rotation matrix for a clockwise rotation in the XY plane.
    pub fn rotation_clockwise(radians: T) -> Self
    where
        T: Trig,
    {
        Self::rotation_counterclockwise(-radians)
    }

    /// Creates a 2x2 rotation matrix for a counterclockwise rotation in the XY plane.
    pub fn rotation_counterclockwise(radians: T) -> Self
    where
        T: Trig,
    {
        Self::from_columns([
            [radians.cos(), radians.sin()],
            [-radians.sin(), radians.cos()],
        ])
    }
}

impl<T: Number> Matrix<T, 3, 3> {
    /// Returns the [determinant] of the matrix.
    ///
    /// [determinant]: https://en.wikipedia.org/wiki/Determinant
    pub fn determinant(&self) -> T {
        let [[a, d, g], [b, e, h], [c, f, i]] = self.0;
        a * e * i + b * f * g + c * d * h - c * e * g - b * d * i - a * f * h
    }
}

impl<T, const R: usize, const C: usize> Default for Matrix<T, R, C>
where
    T: Default,
{
    fn default() -> Self {
        Self::from_fn(|_, _| T::default())
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use crate::{assert_approx_eq, vec2};

    use super::*;

    #[test]
    fn from_rows_columns() {
        assert_eq!(
            Mat2x3::from_rows([[1, 2, 3], [4, 5, 6]]),
            Mat2x3::from_columns([[1, 4], [2, 5], [3, 6]]),
        );
    }

    #[test]
    fn diagonal() {
        let mat = Matrix::from_diagonal([1, 2]);

        #[rustfmt::skip]
        assert_eq!(mat, Matrix::from_rows([
            [1, 0],
            [0, 2],
        ]));

        assert_eq!(mat.into_diagonal(), [1, 2]);
    }

    #[test]
    fn fmt() {
        let mat = Matrix::from_rows([[0, 1], [2, 3]]);

        // Natural writing order (row-wise) for debug output.
        assert_eq!(format!("{:?}", mat), "[[0, 1], [2, 3]]");

        // `#` modifier prints each row in its own line, but not each individual element.
        assert_eq!(
            format!("{:#?}", mat),
            "
[
    [0, 1],
    [2, 3],
]
"
            .trim()
        );
    }

    #[test]
    fn constants() {
        assert_eq!(format!("{:?}", Mat2f::ZERO), "[[0.0, 0.0], [0.0, 0.0]]");
        assert_eq!(format!("{:?}", Mat2f::IDENTITY), "[[1.0, 0.0], [0.0, 1.0]]");
    }

    #[rustfmt::skip]
    #[test]
    fn resize() {
        let mat = Matrix::from_rows([
            [1, 2],
            [3, 4],
        ]);

        let larger = mat.resize::<3, 3>();
        assert_eq!(larger, Matrix::from_rows([
            [1, 2, 0],
            [3, 4, 0],
            [0, 0, 0],
        ]));

        let smaller = mat.resize::<1, 2>();
        assert_eq!(smaller, Matrix::from_rows([
            [1, 2]
        ]));
    }

    #[test]
    fn mat_vec_mul() {
        let mat = Matrix::from_rows([[0, 1], [2, 3]]);
        let vec = vec2(4, 5);
        let out = mat * vec;
        assert_eq!(out, [4 * 0 + 5 * 1, 4 * 2 + 5 * 3]);
    }

    #[test]
    fn mat_mat_mul() {
        #[rustfmt::skip]
        let a = Matrix::from_rows([
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
        ]);
        #[rustfmt::skip]
        let b = Matrix::from_rows([
            [9, 10, 11],
            [12, 13, 14],
        ]);
        let c = a * b;
        assert_eq!(c[(0, 1)], a[(0, 0)] * b[(0, 1)] + a[(0, 1)] * b[(1, 1)]);
        assert_eq!(c[(2, 2)], a[(2, 0)] * b[(0, 2)] + a[(2, 1)] * b[(1, 2)]);
    }

    #[test]
    fn determinant() {
        assert_eq!(Mat1f::ZERO.determinant(), 0.0);
        assert_eq!(Mat2f::ZERO.determinant(), 0.0);
        assert_eq!(Mat3f::ZERO.determinant(), 0.0);
        assert_eq!(Mat1f::IDENTITY.determinant(), 1.0);
        assert_eq!(Mat2f::IDENTITY.determinant(), 1.0);
        assert_eq!(Mat3f::IDENTITY.determinant(), 1.0);

        #[rustfmt::skip]
        let testmat = Matrix::from_rows([
            [-2, -1,  2],
            [ 2,  1,  4],
            [-3,  3, -1],
        ]);
        assert_eq!(testmat.determinant(), 54);
        assert_eq!(testmat.transpose().determinant(), 54);
    }

    #[test]
    fn rotation() {
        let cw = Mat2f::rotation_clockwise(0.0);
        assert_eq!(cw, cw.invert());

        let ccw = Mat2f::rotation_counterclockwise(0.0);
        assert_eq!(ccw, ccw.invert());

        assert_eq!(ccw, cw);

        let cw = Mat2f::rotation_clockwise(PI);
        assert_approx_eq!(cw, cw.invert()).abs(1e-6);
    }
}
