// FIXME: renaming to `ConstZero`/`ConstOne` and introducing non-const `Zero`/`One` traits might be useful

use std::ops;

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

/// Trait for types that support the basic trigonometric functions (`sin`, `cos` and `tan`).
pub trait Trig {
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
}

/// Types that support computing their square root.
pub trait Sqrt {
    fn sqrt(self) -> Self;
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
