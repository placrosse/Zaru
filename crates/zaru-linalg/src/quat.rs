mod ops;
mod view;

use crate::{vec4, Number, One, Sqrt, Trig, Vector, Zero};

/// A quaternion consisting of 3 imaginary numbers and a real number.
///
/// Unit-length quaternions ("*versors*") are commonly used to represent rotations in 3D space.
///
/// Quaternions are represented similar to a 4-dimensional vector, with an `x`, `y`, `z` and `w`
/// component.
#[derive(Clone, Copy, Hash)]
pub struct Quat<T> {
    vec: Vector<T, 4>,
}

impl<T: Zero + One> Quat<T> {
    /// The multiplicative identity.
    ///
    /// This is a unit quaternion that will not change a vector it is multiplied with.
    pub const IDENTITY: Self = Self {
        vec: vec4(T::ZERO, T::ZERO, T::ZERO, T::ONE),
    };
}

impl<T> Quat<T> {
    /// Creates a quaternion from a 4-dimensional [`Vector`].
    ///
    /// The `x`, `y`, and `z` coordinates correspond to the `i`, `j`, and `k` imaginary parts, while
    /// the `w` component corresponds to the real number part of the quaternion.
    pub fn from_vec(vec: Vector<T, 4>) -> Self {
        Self { vec }
    }

    pub fn from_components(x: T, y: T, z: T, w: T) -> Self {
        Self {
            vec: [x, y, z, w].into(),
        }
    }

    fn one_half() -> T
    where
        T: Number,
    {
        T::ONE / (T::ONE + T::ONE)
    }

    pub fn from_rotation_x(radians: T) -> Self
    where
        T: Trig + Number,
    {
        let (sin, cos) = (radians * Self::one_half()).sin_cos();
        Self::from_components(sin, T::ZERO, T::ZERO, cos)
    }

    pub fn from_rotation_y(radians: T) -> Self
    where
        T: Trig + Number,
    {
        let (sin, cos) = (radians * Self::one_half()).sin_cos();
        Self::from_components(sin, T::ZERO, T::ZERO, cos)
    }

    pub fn from_rotation_z(radians: T) -> Self
    where
        T: Trig + Number,
    {
        let (sin, cos) = (radians * Self::one_half()).sin_cos();
        Self::from_components(T::ZERO, T::ZERO, sin, cos)
    }

    /// Creates a quaternion representing a rotation around the X, Y, and Z axis, in sequence.
    #[doc(alias = "euler")]
    pub fn from_rotation_xyz(x: T, y: T, z: T) -> Self
    where
        T: Number + Trig,
    {
        Self::from_rotation_x(x) * Self::from_rotation_y(y) * Self::from_rotation_z(z)
    }

    /// Returns the squared length of this quaternion.
    ///
    /// If the squared length is not equal to one, multiplying a vector with this quaternion will
    /// scale the vector in addition to rotating it. When using quaternions to model rotations, it
    /// is advisable to ensure that quaternions are always of length one.
    pub fn length2(&self) -> T
    where
        T: Number,
    {
        self.vec.length2()
    }

    /// Returns the length of this quaternion.
    ///
    /// If the length is not equal to one, multiplying a vector with this quaternion will scale the
    /// vector in addition to rotating it. When using quaternions to model rotations, it is
    /// advisable to ensure that quaternions are always of length one.
    #[doc(alias = "norm", alias = "magnitude")]
    pub fn length(&self) -> T
    where
        T: Number + Sqrt,
    {
        self.vec.length()
    }

    /// Returns a normalized copy of this quaternion (whose length equals one).
    pub fn normalize(self) -> Self
    where
        T: Number + Sqrt,
    {
        Self {
            vec: self.vec.normalize(),
        }
    }
}
