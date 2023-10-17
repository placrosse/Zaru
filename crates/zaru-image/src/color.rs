use std::{fmt, ops::Index};

/// An 8-bit RGBA color.
///
/// Colors are always in the non-linear sRGB color space and use non-premultiplied alpha.
#[derive(PartialEq, Eq, Clone, Copy)]
pub struct Color(pub(crate) [u8; 4]);

impl Color {
    /// Fully transparent black (all components are 0).
    pub const NONE: Self = Self([0, 0, 0, 0]);
    pub const BLACK: Self = Self([0, 0, 0, 255]);
    pub const WHITE: Self = Self([255, 255, 255, 255]);
    pub const RED: Self = Self([255, 0, 0, 255]);
    pub const GREEN: Self = Self([0, 255, 0, 255]);
    pub const BLUE: Self = Self([0, 0, 255, 255]);
    pub const YELLOW: Self = Self([255, 255, 0, 255]);
    pub const MAGENTA: Self = Self([255, 0, 255, 255]);
    pub const CYAN: Self = Self([0, 255, 255, 255]);

    #[inline]
    pub const fn from_rgb8(r: u8, g: u8, b: u8) -> Self {
        Self([r, g, b, 255])
    }

    #[inline]
    pub const fn from_rgba8(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self([r, g, b, a])
    }

    #[inline]
    pub fn r(&self) -> u8 {
        self.0[0]
    }

    #[inline]
    pub fn g(&self) -> u8 {
        self.0[1]
    }

    #[inline]
    pub fn b(&self) -> u8 {
        self.0[2]
    }

    #[inline]
    pub fn a(&self) -> u8 {
        self.0[3]
    }

    pub fn with_alpha(mut self, a: u8) -> Color {
        self.0[3] = a;
        self
    }

    /// Converts this [`Color`] to linear sRGB with channel values in range 0.0 to 1.0.
    pub fn to_linear(self) -> [f32; 4] {
        [
            channel_to_linear(self.r()),
            channel_to_linear(self.g()),
            channel_to_linear(self.b()),
            f32::from(self.a()) / 255.0,
        ]
    }
}

fn channel_to_linear(c: u8) -> f32 {
    let f = f32::from(c) / 255.0;
    if f <= 0.04045 {
        f / 12.92
    } else {
        ((f + 0.055) / 1.055).powf(2.4)
    }
}

impl fmt::Debug for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "#{:02x}{:02x}{:02x}{:02x}",
            self.r(),
            self.g(),
            self.b(),
            self.a(),
        )
    }
}

impl Index<usize> for Color {
    type Output = u8;

    #[inline]
    fn index(&self, index: usize) -> &u8 {
        &self.0[index]
    }
}
