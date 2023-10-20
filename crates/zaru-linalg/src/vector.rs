use std::{array, fmt};

use crate::{
    traits::{Number, Sqrt},
    Mat2, MinMax, One, Trig, Zero,
};

mod ops;
mod view;

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
/// - For vectors with up to 4 dimensions, elements can be accessed as fields `x`, `y`, `z`, or `w`.
///   - Aliases `r`, `g`, `b`, and `a` are also provided, as well as aliases `w` and `h` for
///     2-dimensional vectors.
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

impl<T: Zero, const N: usize> Vector<T, N> {
    /// A vector with each element initialized to 0.
    ///
    /// This uses [`T::ZERO`][Zero::ZERO] as the value for all elements.
    pub const ZERO: Self = Self([T::ZERO; N]);
}

impl<T: Zero + One> Vector<T, 1> {
    /// A unit vector pointing in the X direction.
    pub const X: Self = Self([T::ONE]);
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
    #[inline]
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
    #[inline]
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
    pub fn length2(&self) -> T
    where
        T: Number,
    {
        self.dot(*self)
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
    pub fn length(&self) -> T
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
    /// # use zaru_linalg::*;
    /// let z = vec3(0.0, 0.0, 4.0).normalize();
    /// assert_eq!(z, vec3(0.0, 0.0, 1.0));
    /// ```
    pub fn normalize(self) -> Self
    where
        T: Number + Sqrt,
    {
        self / self.length()
    }

    /// Computes the dot product between `self` and `other`.
    ///
    /// Geometrically, the dot product provides information about the relative
    /// angle of the two vectors:
    /// - If the dot product is greater than zero, the angle between the vectors
    ///   is less than 90°.
    /// - If the dot product is equal to zero, their angle is exactly 90°.
    /// - If the dot product is negative, the angle is greater than 90°.
    ///
    /// Also see [`Vector::abs_angle_to`] for computing the exact angle between them.
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// let a = vec3(1, 3, -5);
    /// let b = vec3(4, -2, -1);
    /// assert_eq!(a.dot(b), 3);
    /// ```
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// assert_approx_eq!(Vec2f::Y.dot(Vec2f::X), 0.0);
    /// assert_approx_eq!(Vec2f::Y.dot(Vec2f::Y), 1.0);
    /// assert_approx_eq!(Vec2f::Y.dot(-Vec2f::Y), -1.0);
    /// ```
    pub fn dot(self, other: Self) -> T
    where
        T: Number,
    {
        self.into_array()
            .into_iter()
            .zip(other.into_array())
            .fold(T::ZERO, |acc, (a, b)| acc + a * b)
    }

    /// Computes the smallest positive angle between `self` and `other`, in radians.
    ///
    /// Both `self` and `other` must have non-zero length for the result to be meaningful.
    ///
    /// Also see [`Vector::signed_angle_to`] for getting a signed result depending on the relative
    /// orientation of the vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// use std::f32::consts::TAU;
    ///
    /// let a = Vec3f::Y;
    /// let b = Vec3f::X;
    /// assert_approx_eq!(a.abs_angle_to(b), TAU / 4.0);  // quarter turn
    /// assert_approx_eq!(b.abs_angle_to(a), TAU / 4.0);  // quarter turn
    /// assert_approx_eq!(a.abs_angle_to(-a), TAU / 2.0); // half a turn
    /// ```
    pub fn abs_angle_to(self, other: Self) -> T
    where
        T: Number + Trig + Sqrt,
    {
        let dot = self.dot(other);
        let angle = (dot / (self.length() * other.length())).acos();
        angle
    }

    /// Element-wise minimum between `self` and `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
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
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
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

    /// Element-wise range clamp of the elements in `self` between `min` and `max`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// let a = vec3(-1.0, 2.0, f32::NAN);
    /// let b = vec3(3.0, f32::NEG_INFINITY, 0.0);
    /// assert_eq!(a.max(b), b.max(a));
    /// assert_eq!(a.max(b), vec3(3.0, 2.0, 0.0));
    /// ```
    pub fn clamp(self, min: Self, max: Self) -> Self
    where
        T: MinMax + Copy,
    {
        Self::from_fn(|i| self[i].clamp(min[i], max[i]))
    }
}

impl<T> Vector<T, 1> {
    /// Removes the last element of this vector, yielding a vector with zero elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// let v = vec1(-1.0).truncate();
    /// assert_eq!(v, []);
    /// ```
    pub fn truncate(self) -> Vector<T, 0> {
        [].into()
    }

    /// Appends another value to the vector, yielding a vector with 2 dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// let v = vec1(-1.0).extend(5.0);
    /// assert_eq!(v, vec2(-1.0, 5.0));
    /// ```
    pub fn extend(self, value: T) -> Vector<T, 2> {
        let [x] = self.into_array();
        [x, value].into()
    }
}

impl<T> Vector<T, 2> {
    /// Removes the last element of this vector, yielding a vector with a single element.
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// let v = vec2(-1.0, 2.0).truncate();
    /// assert_eq!(v, vec1(-1.0));
    /// ```
    pub fn truncate(self) -> Vector<T, 1> {
        let [x, ..] = self.into_array();
        [x].into()
    }

    /// Appends another value to the vector, yielding a vector with 3 dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// let v = vec2(-1.0, 2.0).extend(5.0);
    /// assert_eq!(v, vec3(-1.0, 2.0, 5.0));
    /// ```
    pub fn extend(self, value: T) -> Vector<T, 3> {
        let [x, y] = self.into_array();
        [x, y, value].into()
    }

    /// Rotates `self` clockwise in the 2D plane.
    ///
    /// This operation assumes that the Y axis points up, and the X axis points to the right.
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// use std::f32::consts::TAU;
    ///
    /// assert_approx_eq!(Vec2f::Y.rotate_clockwise(TAU / 4.0), Vec2f::X);
    /// assert_approx_eq!(Vec2f::Y.rotate_clockwise(TAU / 2.0), -Vec2f::Y);
    /// ```
    pub fn rotate_clockwise(self, radians: T) -> Self
    where
        T: Number + Trig,
    {
        Mat2::rotation_clockwise(radians) * self
    }

    /// Rotates `self` counterclockwise in the 2D plane.
    ///
    /// This operation assumes that the Y axis points up, and the X axis points to the right.
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// use std::f32::consts::TAU;
    ///
    /// assert_approx_eq!(Vec2f::Y.rotate_counterclockwise(TAU / 4.0), -Vec2f::X);
    /// assert_approx_eq!(Vec2f::X.rotate_counterclockwise(TAU / 4.0), Vec2f::Y);
    /// assert_approx_eq!(Vec2f::Y.rotate_counterclockwise(TAU / 2.0), -Vec2f::Y);
    /// ```
    pub fn rotate_counterclockwise(self, radians: T) -> Self
    where
        T: Number + Trig,
    {
        Mat2::rotation_counterclockwise(radians) * self
    }

    /// Computes the (signed) clockwise rotation in radians needed to align `self` with `other`.
    ///
    /// This operation assumes that the Y axis points up, and the X axis points to the right. If the
    /// Y axis points *down*, swap the arguments to make the method work correctly.
    ///
    /// Also see [`Vector::abs_angle_to`] for a more general way of getting the unsigned angle
    /// between vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// use std::f32::consts::TAU;
    ///
    /// // The Y axis can be aligned with the X axis by rotating it clockwise by a quarter turn.
    /// assert_approx_eq!(Vec2f::Y.signed_angle_to(Vec2f::X), TAU / 4.0);
    ///
    /// // The X axis can be aligned with the Y axis by rotating it counterclockwise by a quarter turn.
    /// assert_approx_eq!(Vec2f::X.signed_angle_to(Vec2f::Y), -TAU / 4.0);
    ///
    /// // The angle of a vector to itself is, of course, 0.
    /// assert_approx_eq!(Vec2f::Y.signed_angle_to(Vec2f::Y), 0.0);
    ///
    /// // The result for opposing vectors is ambiguous: it could be either `TAU / 2` or `-TAU / 2`.
    /// assert_approx_eq!(Vec2f::Y.signed_angle_to(-Vec2f::Y), -TAU / 2.0);
    /// ```
    pub fn signed_angle_to(self, other: Self) -> T
    where
        T: Number + Trig,
    {
        -self.perp_dot(other).atan2(self.dot(other))
    }

    /// Computes the [perpendicular dot product] of `self` and `other`.
    ///
    /// This is equivalent to the Z coordinate of the cross product of `self` and `other`
    /// (extended with Z=0 in the third dimension). Since the Z coordinates of both inputs are 0,
    /// the Z coordinate is the only non-zero coordinate of the cross product.
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// let x = Vec2f::X;
    /// let y = Vec2f::Y;
    /// assert_eq!(x.perp_dot(y), 1.0);
    /// assert_eq!(y.perp_dot(x), -1.0);
    /// ```
    ///
    /// [perpendicular dot product]: https://mathworld.wolfram.com/PerpDotProduct.html
    pub fn perp_dot(self, other: Self) -> T
    where
        T: Number,
    {
        self.extend(T::ZERO).cross(other.extend(T::ZERO)).z
    }
}

impl<T> Vector<T, 3> {
    /// Removes the last element of this vector, yielding a vector with 2 elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// let v = vec3(-1.0, 2.0, 3.5).truncate();
    /// assert_eq!(v, vec2(-1.0, 2.0));
    /// ```
    pub fn truncate(self) -> Vector<T, 2> {
        let [x, y, ..] = self.into_array();
        [x, y].into()
    }

    /// Appends another value to the vector, yielding a vector with 4 dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// let v = vec3(-1.0, 2.0, 3.5).extend(99.0);
    /// assert_eq!(v, vec4(-1.0, 2.0, 3.5, 99.0));
    /// ```
    pub fn extend(self, value: T) -> Vector<T, 4> {
        let [x, y, z] = self.into_array();
        [x, y, z, value].into()
    }

    /// Computes the cross product of `self` and `other`.
    ///
    /// The result is a vector that is perpendicular to both `self` and `other`. Its direction
    /// depends on the order of the arguments: swapping them will invert the direction of the
    /// resulting vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # use zaru_linalg::*;
    /// let x = Vec3f::X;
    /// let y = Vec3f::Y;
    /// let z = Vec3f::Z;
    /// assert_eq!(x.cross(y), z);
    /// assert_eq!(y.cross(x), -z);
    /// ```
    pub fn cross(self, other: Self) -> Self
    where
        T: Number,
    {
        let [a1, a2, a3] = self.into_array();
        let [b1, b2, b3] = other.into_array();

        #[rustfmt::skip]
        let cross = vec3(
            a2 * b3 - a3 * b2,
            a3 * b1 - a1 * b3,
            a1 * b2 - a2 * b1,
        );
        cross
    }
}

impl<T, const N: usize> Default for Vector<T, N>
where
    T: Default,
{
    #[inline]
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
#[inline]
pub const fn vec1<T>(x: T) -> Vec1<T> {
    Vector([x])
}

/// Constructs a [`Vec2`] from its two elements.
#[inline]
pub const fn vec2<T>(x: T, y: T) -> Vec2<T> {
    Vector([x, y])
}

/// Constructs a [`Vec3`] from its three elements.
#[inline]
pub const fn vec3<T>(x: T, y: T, z: T) -> Vec3<T> {
    Vector([x, y, z])
}

/// Constructs a [`Vec4`] from its four elements.
#[inline]
pub const fn vec4<T>(x: T, y: T, z: T, w: T) -> Vec4<T> {
    Vector([x, y, z, w])
}

#[cfg(test)]
mod tests {
    use std::f32::consts::TAU;

    use crate::assert_approx_eq;

    use super::*;

    #[test]
    fn access() {
        assert_eq!(Vec3f::X.x, 1.0);
        assert_eq!(Vec3f::X[0], 1.0);
        assert_eq!(Vec3f::X[1], 0.0);
        assert_eq!(Vec3f::X[2], 0.0);
        assert_eq!(Vec3f::X.y, 0.0);
        assert_eq!(Vec3f::Y.y, 1.0);
        assert_eq!(Vec3f::Y.z, 0.0);
        assert_eq!(Vec4f::W.w, 1.0);

        let mut v = vec2(0, 1);
        assert_eq!(v.x, 0);
        assert_eq!(v.y, 1);
        assert_eq!(v.r, 0);
        assert_eq!(v.g, 1);
        assert_eq!(v.w, 0);
        assert_eq!(v.h, 1);
        assert_eq!(v[0], 0);
        assert_eq!(v[1], 1);

        v.r = 777;
        assert_eq!(v.x, 777);
        assert_eq!(v.y, 1);
        assert_eq!(v.r, 777);
        assert_eq!(v.g, 1);
        assert_eq!(v.w, 777);
        assert_eq!(v.h, 1);
        assert_eq!(v[0], 777);
        assert_eq!(v[1], 1);
        v.h = 9;
        assert_eq!(v.x, 777);
        assert_eq!(v.y, 9);
        assert_eq!(v.r, 777);
        assert_eq!(v.g, 9);
        assert_eq!(v.w, 777);
        assert_eq!(v.h, 9);
        assert_eq!(v[0], 777);
        assert_eq!(v[1], 9);

        // :3
    }

    #[test]
    fn fmt() {
        assert_eq!(format!("{}", Vec4f::W), "(0, 0, 0, 1)");
        assert_eq!(format!("{:?}", Vec4f::W), "(0.0, 0.0, 0.0, 1.0)");
    }

    #[test]
    fn rotate() {
        assert_approx_eq!(Vec2f::Y.rotate_clockwise(TAU / 4.0), Vec2f::X);
        assert_approx_eq!(Vec2f::Y.rotate_clockwise(TAU / 2.0), -Vec2f::Y);
        assert_approx_eq!(Vec2f::X.rotate_clockwise(TAU / 2.0), -Vec2f::X);
        assert_approx_eq!(Vec2f::X.rotate_counterclockwise(TAU / 4.0), Vec2f::Y);
    }

    #[test]
    fn dot() {
        assert_eq!(vec3(1, 3, -5).dot(vec3(4, -2, -1)), 3);
        assert_eq!(vec3(1, 3, -5).dot(vec3(1, 3, -5)), 35);

        assert_eq!(Vec2f::X.dot(Vec2f::X), 1.0);
        assert_eq!(Vec2f::Y.dot(Vec2f::Y), 1.0);
        assert_eq!(Vec2f::X.dot(Vec2f::Y), 0.0);
        assert_eq!(Vec2f::Y.dot(Vec2f::X), 0.0);
    }

    #[test]
    fn abs_angle() {
        assert_approx_eq!(Vec3f::Y.abs_angle_to(Vec3f::X), TAU / 4.0);
        assert_approx_eq!(Vec3f::X.abs_angle_to(Vec3f::Y), TAU / 4.0);

        assert_approx_eq!(Vec3f::Y.abs_angle_to(Vec3f::Y), 0.0);
        assert_approx_eq!(Vec3f::Y.abs_angle_to(-Vec3f::Y), TAU / 2.0);
        assert_approx_eq!(Vec3f::Y.abs_angle_to(-Vec3f::X), TAU / 4.0);

        assert_approx_eq!(Vec2f::Y.abs_angle_to(Vec2f::X), TAU / 4.0);
        assert_approx_eq!(Vec2f::Y.abs_angle_to(-Vec2f::Y), TAU / 2.0);

        assert_approx_eq!(vec2(0.0, 2.0).abs_angle_to(Vec2f::X), TAU / 4.0);
        assert_approx_eq!(vec2(0.0, 2.0).abs_angle_to(vec2(-3.0, 0.0)), TAU / 4.0);

        assert_approx_eq!(vec2(1.0, 1.0).abs_angle_to(vec2(1.0, -1.0)), TAU / 4.0);
    }

    #[test]
    fn signed_angle() {
        assert_approx_eq!(Vec2f::Y.signed_angle_to(Vec2f::X), TAU / 4.0);
        assert_approx_eq!(Vec2f::X.signed_angle_to(Vec2f::Y), -TAU / 4.0);
        assert_approx_eq!(Vec2f::Y.signed_angle_to(Vec2f::Y), 0.0);

        assert_approx_eq!(
            Vec2f::Y
                .rotate_counterclockwise(100.0f32.to_radians())
                .signed_angle_to(Vec2f::Y),
            100.0f32.to_radians()
        );

        assert_approx_eq!(Vec2f::Y.signed_angle_to(-Vec2f::Y), -TAU / 2.0);
    }
}
