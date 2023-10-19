use std::{
    array, fmt, mem,
    ops::{Deref, DerefMut},
};

use crate::{
    traits::{Number, Sqrt},
    Mat2, MinMax, One, Trig, Zero,
};

mod ops;

/// A 1-dimensional vector.
pub type Vec1<T> = Vector<T, 1>;
/// A 1-dimensional vector with [`f32`] elements.
pub type Vec1f = Vec1<f32>;
/// A 2-dimensional vector.
pub type Vec2<T> = Vector<T, 2>;
/// A 2-dimensional vector with [`f32`] elements.
pub type Vec2f = Vec2<f32>;
/// A 3-dimensional vector.
pub type Vec3<T> = Vector<T, 3>;
/// A 3-dimensional vector with [`f32`] elements.
pub type Vec3f = Vec3<f32>;
/// A 4-dimensional vector.
pub type Vec4<T> = Vector<T, 4>;
/// A 4-dimensional vector with [`f32`] elements.
pub type Vec4f = Vec4<f32>;

/// An `N`-element column vector storing elements of type `T`.
///
/// # Construction
///
/// There is a variety of ways to create a [`Vector`]:
///
/// - The freestanding [`vec2`], [`vec3`] and [`vec4`] functions directly create vectors from
///   provided values.
/// - [`Vector::splat`] creates a vector by copying the given value into each element.
/// - [`Vector::from_fn`] creates a vector by invoking a closure with the index of each element.
/// - Vectors can be created from arrays using their [`From`] implementation.
/// - The [`Default`] implementation of [`Vector`] initializes each element with its default value.
/// - [`Vector::ZERO`] is a vector containing all-zeroes.
/// - For vectors with up to 4 dimensions, `Vector::X`, `Vector::Y`, `Vector::Z` and `Vector::W` can
///   be used to obtain unit vectors pointing in the given direction.
///
/// # Element Access
///
/// Vector elements can be accessed and inspected in a few different ways:
///
/// - For vectors with up to 4 dimensions, elements can be accessed as fields (eg. `myvec.x` or
///   `myvec.w`).
/// - The [`Index`] and [`IndexMut`] impls can be used just like on arrays.
/// - The [`AsRef`] and [`AsMut`] impls can be used to access the underlying elements as a slice or
///   array.
/// - A [`From`] impl allows conversion from a [`Vector`] to an array of the same length.
/// - [`Vector::as_array`], [`Vector::as_slice`], and [`Vector::into_array`] allow the same
///   operations without requiring type annotations.
/// - [`bytemuck::Zeroable`] and [`bytemuck::Pod`] are implemented to allow safe transmutation when
///   the element type `T` also allows this.
///
/// [`Index`]: std::ops::Index
/// [`IndexMut`]: std::ops::IndexMut
#[derive(Clone, Copy, Hash)]
#[repr(transparent)]
pub struct Vector<T, const N: usize>([T; N]);

unsafe impl<T: bytemuck::Zeroable, const N: usize> bytemuck::Zeroable for Vector<T, N> {}
unsafe impl<T: bytemuck::Pod, const N: usize> bytemuck::Pod for Vector<T, N> {}
// FIXME: do we even want these? `.as_slice()` lets you do the same thing.

impl<T, const N: usize> Vector<T, N> {
    /// Creates a vector with each element initialized to `elem`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// let v = Vector::splat(2);
    /// assert_eq!(v, vec3(2, 2, 2));
    /// ```
    pub fn splat(elem: T) -> Self
    where
        T: Copy,
    {
        Self(array::from_fn(|_| elem))
    }

    /// Creates a vector where each element is initialized by invoking a closure with its index.
    ///
    /// Analogous to [`array::from_fn`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// let v = Vector::from_fn(|i| i + 100);
    /// assert_eq!(v, vec3(100, 101, 102));
    /// ```
    pub fn from_fn<F>(cb: F) -> Self
    where
        F: FnMut(usize) -> T,
    {
        Self(array::from_fn(cb))
    }

    /// Applies a closure to each element, returning a new vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// let v = vec3(1, 2, 3).map(|i| i * 10);
    /// assert_eq!(v, vec3(10, 20, 30));
    /// ```
    pub fn map<F, U>(self, f: F) -> Vector<U, N>
    where
        F: FnMut(T) -> U,
    {
        Vector(self.0.map(f))
    }

    /// Merges two [`Vector`]s into one that contains tuples of the original elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// let a = vec3(1, 2, 3);
    /// let b = vec3("1", "2", "3");
    /// let v = a.zip(b);
    /// assert_eq!(v, vec3((1, "1"), (2, "2"), (3, "3")));
    /// ```
    pub fn zip<U>(self, other: Vector<U, N>) -> Vector<(T, U), N> {
        let mut iter = self.0.into_iter().zip(other.0);
        Vector::from_fn(|_| iter.next().unwrap())
    }
    // TODO: replace `self` param with `left: Self`

    /// Returns a reference to the underlying elements as an array of length `N`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// assert_eq!(vec3(1, 2, 3).as_array(), &[1, 2, 3]);
    /// ```
    #[inline]
    pub const fn as_array(&self) -> &[T; N] {
        &self.0
    }

    /// Returns a mutable reference to the underlying elements as an array of length `N`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// let mut v = vec3(1, 2, 3);
    /// v.as_mut_array()[1] = 777;
    /// assert_eq!(v, [1, 777, 3]);
    /// ```
    #[inline]
    pub fn as_mut_array(&mut self) -> &mut [T; N] {
        &mut self.0
    }

    /// Returns a reference to the underlying elements as a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// assert_eq!(vec3(1, 2, 3).as_slice(), &[1, 2, 3]);
    /// ```
    #[inline]
    pub const fn as_slice(&self) -> &[T] {
        &self.0
    }

    /// Returns a mutable reference to the underlying elements as a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// let mut v = vec3(1, 2, 3);
    /// v.as_mut_slice()[1] = 777;
    /// assert_eq!(v, [1, 777, 3]);
    /// ```
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.0
    }

    /// Returns a [`Vector`] that borrows each element of `self`.
    ///
    /// *Note*: [`Vector`] also implements [`AsRef`]. This method will typically be preferred over
    /// those impls. Use fully-qualified syntax to invoke the trait methods if needed.
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// assert_eq!(vec3(1, 2, 3).as_ref(), vec3(&1, &2, &3));
    /// ```
    pub fn as_ref(&self) -> Vector<&T, N> {
        Vector::from_fn(|i| &self[i])
    }

    /// Converts this [`Vector`] into an `N`-element array.
    ///
    /// There is an equivalent [`From`] impl that can also be used, but this method is often shorter
    /// and requires no type annotation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// assert_eq!(vec3(1, 2, 3).into_array(), [1, 2, 3]);
    /// ```
    #[inline]
    pub fn into_array(self) -> [T; N] {
        self.0
    }

    /// Returns the squared length of this [`Vector`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// assert_eq!(vec2(4, 0).length2(), 16);
    /// ```
    pub fn length2(self) -> T
    where
        T: Number,
    {
        self.0
            .into_iter()
            .fold(T::ZERO, |prev, elem| prev + elem * elem)
    }

    /// Returns the length of this [`Vector`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// let z = Vec3f::Z;
    /// assert_eq!(z.length(), 1.0);
    /// ```
    pub fn length(self) -> T
    where
        T: Number + Sqrt,
    {
        self.length2().sqrt()
    }

    /// Divides this vector by its length, resulting in a unit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::{Vec3f, vec3};
    /// let z = vec3(0.0, 0.0, 4.0).normalize();
    /// assert_eq!(z, vec3(0.0, 0.0, 1.0));
    /// ```
    pub fn normalize(self) -> Self
    where
        T: Number + Sqrt,
    {
        self / self.length()
    }

    /// Element-wise minimum between `self` and `other`.
    ///
    /// # Example
    ///
    /// ```
    /// # use zaru_linalg::{Vec3f, vec3};
    /// let a = vec3(-1.0, 2.0, f32::NAN);
    /// let b = vec3(3.0, f32::NEG_INFINITY, 0.0);
    /// assert_eq!(a.min(b), b.min(a));
    /// assert_eq!(a.min(b), vec3(-1.0, f32::NEG_INFINITY, 0.0));
    /// ```
    pub fn min(self, other: Self) -> Self
    where
        T: MinMax + Copy,
    {
        Self::from_fn(|i| self[i].min(other[i]))
    }

    /// Element-wise maximum between `self` and `other`.
    ///
    /// # Example
    ///
    /// ```
    /// # use zaru_linalg::{Vec3f, vec3};
    /// let a = vec3(-1.0, 2.0, f32::NAN);
    /// let b = vec3(3.0, f32::NEG_INFINITY, 0.0);
    /// assert_eq!(a.max(b), b.max(a));
    /// assert_eq!(a.max(b), vec3(3.0, 2.0, 0.0));
    /// ```
    pub fn max(self, other: Self) -> Self
    where
        T: MinMax + Copy,
    {
        Self::from_fn(|i| self[i].max(other[i]))
    }
}

impl<T: Zero, const N: usize> Vector<T, N> {
    /// A vector with each element initialized to 0.
    ///
    /// This uses [`T::ZERO`][Zero::ZERO] as the value for all elements.
    pub const ZERO: Self = Self([T::ZERO; N]);
}

impl<T: Zero + One> Vector<T, 2> {
    /// A unit vector pointing in the X direction.
    pub const X: Self = Self([T::ONE, T::ZERO]);
    /// A unit vector pointing in the Y direction.
    pub const Y: Self = Self([T::ZERO, T::ONE]);
}

impl<T: Zero + One> Vector<T, 3> {
    /// A unit vector pointing in the X direction.
    pub const X: Self = Self([T::ONE, T::ZERO, T::ZERO]);
    /// A unit vector pointing in the Y direction.
    pub const Y: Self = Self([T::ZERO, T::ONE, T::ZERO]);
    /// A unit vector pointing in the Z direction.
    pub const Z: Self = Self([T::ZERO, T::ZERO, T::ONE]);
}

impl<T: Zero + One> Vector<T, 4> {
    /// A unit vector pointing in the X direction.
    pub const X: Self = Self([T::ONE, T::ZERO, T::ZERO, T::ZERO]);
    /// A unit vector pointing in the Y direction.
    pub const Y: Self = Self([T::ZERO, T::ONE, T::ZERO, T::ZERO]);
    /// A unit vector pointing in the Z direction.
    pub const Z: Self = Self([T::ZERO, T::ZERO, T::ONE, T::ZERO]);
    /// A unit vector pointing in the W direction.
    pub const W: Self = Self([T::ZERO, T::ZERO, T::ZERO, T::ONE]);
}

impl<T> Vector<T, 2> {
    /// Rotates `self` clockwise in the 2D plane.
    pub fn rotate_clockwise(self, radians: T) -> Self
    where
        T: Number + Trig,
    {
        Mat2::rotation_clockwise(radians) * self
    }

    /// Rotates `self` counterclockwise in the 2D plane.
    pub fn rotate_counterclockwise(self, radians: T) -> Self
    where
        T: Number + Trig,
    {
        Mat2::rotation_counterclockwise(radians) * self
    }
}

impl<T, const N: usize> Default for Vector<T, N>
where
    T: Default,
{
    fn default() -> Self {
        Self::from_fn(|_| T::default())
    }
}

impl<T, const N: usize> From<[T; N]> for Vector<T, N> {
    #[inline]
    fn from(value: [T; N]) -> Self {
        Self(value)
    }
}

impl<T, const N: usize> From<Vector<T, N>> for [T; N] {
    #[inline]
    fn from(value: Vector<T, N>) -> Self {
        value.0
    }
}

impl<T, const N: usize> fmt::Debug for Vector<T, N>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut tup = f.debug_tuple("");
        for elem in &self.0 {
            tup.field(elem);
        }
        tup.finish()
    }
}

impl<T, const N: usize> fmt::Display for Vector<T, N>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct DebugViaDisplay<D>(D);
        impl<D: fmt::Display> fmt::Debug for DebugViaDisplay<D> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }

        let mut tup = f.debug_tuple("");
        for elem in &self.0 {
            tup.field(&DebugViaDisplay(elem));
        }
        tup.finish()
    }
}

impl<T, const N: usize> AsRef<[T]> for Vector<T, N> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        &self.0
    }
}

impl<T, const N: usize> AsRef<[T; N]> for Vector<T, N> {
    #[inline]
    fn as_ref(&self) -> &[T; N] {
        &self.0
    }
}

impl<T, const N: usize> AsMut<[T]> for Vector<T, N> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}

impl<T, const N: usize> AsMut<[T; N]> for Vector<T, N> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T; N] {
        &mut self.0
    }
}

/// Constructs a [`Vec1`] from its single element.
pub const fn vec1<T>(x: T) -> Vec1<T> {
    Vector([x])
}

/// Constructs a [`Vec2`] from its two elements.
pub const fn vec2<T>(x: T, y: T) -> Vec2<T> {
    Vector([x, y])
}

/// Constructs a [`Vec3`] from its three elements.
pub const fn vec3<T>(x: T, y: T, z: T) -> Vec3<T> {
    Vector([x, y, z])
}

/// Constructs a [`Vec4`] from its four elements.
pub const fn vec4<T>(x: T, y: T, z: T, w: T) -> Vec4<T> {
    Vector([x, y, z, w])
}

mod view {
    #[repr(C)]
    pub struct X<T> {
        pub x: T,
    }

    #[repr(C)]
    pub struct XY<T> {
        pub x: T,
        pub y: T,
    }

    #[repr(C)]
    pub struct XYZ<T> {
        pub x: T,
        pub y: T,
        pub z: T,
    }

    #[repr(C)]
    pub struct XYZW<T> {
        pub x: T,
        pub y: T,
        pub z: T,
        pub w: T,
    }
}
use view::*;

impl<T> Deref for Vector<T, 1> {
    type Target = X<T>;

    fn deref(&self) -> &Self::Target {
        unsafe { mem::transmute(self) }
    }
}

impl<T> DerefMut for Vector<T, 1> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { mem::transmute(self) }
    }
}

impl<T> Deref for Vector<T, 2> {
    type Target = XY<T>;

    fn deref(&self) -> &Self::Target {
        unsafe { mem::transmute(self) }
    }
}

impl<T> DerefMut for Vector<T, 2> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { mem::transmute(self) }
    }
}

impl<T> Deref for Vector<T, 3> {
    type Target = XYZ<T>;

    fn deref(&self) -> &Self::Target {
        unsafe { mem::transmute(self) }
    }
}

impl<T> DerefMut for Vector<T, 3> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { mem::transmute(self) }
    }
}

impl<T> Deref for Vector<T, 4> {
    type Target = XYZW<T>;

    fn deref(&self) -> &Self::Target {
        unsafe { mem::transmute(self) }
    }
}

impl<T> DerefMut for Vector<T, 4> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { mem::transmute(self) }
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::TAU;

    use super::*;

    #[test]
    fn access() {
        assert_eq!(Vec3f::X.x, 1.0);
        assert_eq!(Vec3f::X[0], 1.0);
        assert_eq!(Vec3f::X[1], 0.0);
        assert_eq!(Vec3f::X.y, 0.0);
        assert_eq!(Vec3f::Y.y, 1.0);
        assert_eq!(Vec3f::Y.z, 0.0);
        assert_eq!(Vec4f::W.w, 1.0);
    }

    #[test]
    fn fmt() {
        assert_eq!(format!("{}", Vec4f::W), "(0, 0, 0, 1)");
        assert_eq!(format!("{:?}", Vec4f::W), "(0.0, 0.0, 0.0, 1.0)");
    }

    #[test]
    fn rotate() {
        assert!((Vec2f::Y.rotate_clockwise(TAU / 4.0) - Vec2f::X).length() < 0.0001);
        assert!((Vec2f::Y.rotate_clockwise(TAU / 2.0) - -Vec2f::Y).length() < 0.0001);
        assert!((Vec2f::X.rotate_clockwise(TAU / 2.0) - -Vec2f::X).length() < 0.0001);
        assert!((Vec2f::X.rotate_counterclockwise(TAU / 4.0) - Vec2f::Y).length() < 0.0001);
    }
}
