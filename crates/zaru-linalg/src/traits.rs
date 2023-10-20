// FIXME: renaming to `ConstZero`/`ConstOne` and introducing non-const `Zero`/`One` traits might be useful

use std::ops;

/// Types that support the trigonometric functions.
pub trait Trig {
    /// Computes the sine of the angle `self` (in radians).
    fn sin(self) -> Self;
    /// Computes the cosine of the angle `self` (in radians).
    fn cos(self) -> Self;
    /// Computes the tangent of the angle `self` (in radians).
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn atan2(self, other: Self) -> Self;
}

/// Types that support computing their square root.
pub trait Sqrt {
    fn sqrt(self) -> Self;
}

/// Types that support a `min` and `max` operation.
///
/// [`f32`] and [`f64`] implement this trait in terms of the [`f32::min`] and [`f32::max`] functions
/// ([`f64::min`] and [`f64::max`] respectively). Built-in integer types implement it in terms of
/// [`Ord::min`] and [`Ord::max`].
pub trait MinMax: Sized {
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn clamp(self, min: Self, max: Self) -> Self {
        self.max(min).min(max)
    }
}
macro_rules! ord_min_max {
    ($($types:ty),+) => {
        $(
            impl MinMax for $types {
                fn min(self, other: Self) -> Self {
                    Ord::min(self, other)
                }

                fn max(self, other: Self) -> Self {
                    Ord::max(self, other)
                }
            }
        )+
    };
}
ord_min_max!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128);
impl MinMax for f32 {
    fn min(self, other: Self) -> Self {
        self.min(other)
    }

    fn max(self, other: Self) -> Self {
        self.max(other)
    }
}
impl MinMax for f64 {
    fn min(self, other: Self) -> Self {
        self.min(other)
    }

    fn max(self, other: Self) -> Self {
        self.max(other)
    }
}

/// Types that have a "zero" value (an additive identity).
pub trait Zero {
    /// The *0* value of this type.
    const ZERO: Self;
}

/// Types that have a "one" value (a multiplicative identity).
pub trait One {
    /// The *1* value of this type.
    const ONE: Self;
}

/// A trait for numeric types that support basic arithmetic operations.
pub trait Number:
    Zero
    + One
    + ops::Neg<Output = Self>
    + ops::Add<Output = Self>
    + ops::Sub<Output = Self>
    + ops::Mul<Output = Self>
    + ops::Div<Output = Self>
    + PartialEq
    + Copy
{
}
impl<T> Number for T where
    T: Zero
        + One
        + ops::Neg<Output = Self>
        + ops::Add<Output = Self>
        + ops::Sub<Output = Self>
        + ops::Mul<Output = Self>
        + ops::Div<Output = Self>
        + PartialEq
        + Copy
{
}

impl Zero for f32 {
    const ZERO: Self = 0.0;
}
impl Zero for f64 {
    const ZERO: Self = 0.0;
}
impl Zero for u8 {
    const ZERO: Self = 0;
}
impl Zero for u16 {
    const ZERO: Self = 0;
}
impl Zero for u32 {
    const ZERO: Self = 0;
}
impl Zero for u64 {
    const ZERO: Self = 0;
}
impl Zero for u128 {
    const ZERO: Self = 0;
}
impl Zero for i8 {
    const ZERO: Self = 0;
}
impl Zero for i16 {
    const ZERO: Self = 0;
}
impl Zero for i32 {
    const ZERO: Self = 0;
}
impl Zero for i64 {
    const ZERO: Self = 0;
}
impl Zero for i128 {
    const ZERO: Self = 0;
}

impl One for f32 {
    const ONE: Self = 1.0;
}
impl One for f64 {
    const ONE: Self = 1.0;
}
impl One for u8 {
    const ONE: Self = 1;
}
impl One for u16 {
    const ONE: Self = 1;
}
impl One for u32 {
    const ONE: Self = 1;
}
impl One for u64 {
    const ONE: Self = 1;
}
impl One for u128 {
    const ONE: Self = 1;
}
impl One for i8 {
    const ONE: Self = 1;
}
impl One for i16 {
    const ONE: Self = 1;
}
impl One for i32 {
    const ONE: Self = 1;
}
impl One for i64 {
    const ONE: Self = 1;
}
impl One for i128 {
    const ONE: Self = 1;
}

impl Trig for f32 {
    fn sin(self) -> Self {
        self.sin()
    }

    fn cos(self) -> Self {
        self.cos()
    }

    fn tan(self) -> Self {
        self.tan()
    }

    fn asin(self) -> Self {
        self.asin()
    }

    fn acos(self) -> Self {
        self.acos()
    }

    fn atan(self) -> Self {
        self.atan()
    }

    fn atan2(self, other: Self) -> Self {
        self.atan2(other)
    }
}

impl Trig for f64 {
    fn sin(self) -> Self {
        self.sin()
    }

    fn cos(self) -> Self {
        self.cos()
    }

    fn tan(self) -> Self {
        self.tan()
    }

    fn asin(self) -> Self {
        self.asin()
    }

    fn acos(self) -> Self {
        self.acos()
    }

    fn atan(self) -> Self {
        self.atan()
    }

    fn atan2(self, other: Self) -> Self {
        self.atan2(other)
    }
}

impl Sqrt for f32 {
    fn sqrt(self) -> Self {
        self.sqrt()
    }
}
impl Sqrt for f64 {
    fn sqrt(self) -> Self {
        self.sqrt()
    }
}
