//! Types for representing image resolutions.

use std::fmt;

use crate::image::Rect;

/// Resolution (`width x height`) of an image, window, camera, or display.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Resolution {
    width: u32,
    height: u32,
}

impl Resolution {
    /// 1080p – 1920x1080
    pub const RES_1080P: Self = Self {
        width: 1920,
        height: 1080,
    };
    /// 720p – 1280x720
    pub const RES_720P: Self = Self {
        width: 1280,
        height: 720,
    };

    pub fn new(width: u32, height: u32) -> Self {
        assert!(
            width != 0 && height != 0,
            "attempted to create a resolution with 0 width or height"
        );
        Self { width, height }
    }

    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }

    #[inline]
    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn aspect_ratio(&self) -> AspectRatio {
        AspectRatio::from(*self)
    }

    /// Computes a centered, maximally sized [`Rect`] that lies inside of `self` and has the given
    /// aspect ratio.
    ///
    /// This can be used to perform an aspect ratio aware resize operation that automatically leaves
    /// black bars around the image.
    pub fn fit_aspect_ratio(&self, ratio: AspectRatio) -> Rect {
        let to_ratio = self.aspect_ratio();

        let from_ratio = ratio.as_f32();
        let to_ratio = to_ratio.as_f32();

        let (y_min, x_min, w, h);
        if from_ratio > to_ratio {
            // Input has wider aspect ratio than output.
            // => Resulting size is limited by target width. Add Letterboxing.
            w = self.width();
            h = (self.width() as f32 / from_ratio) as u32;

            x_min = 0;
            y_min = (self.height() - h) / 2;
        } else {
            // Output has wider (or equal) aspect ratio than input.
            // => Resulting size is limited by target height. Add Pillarboxing.
            w = (self.height() as f32 * from_ratio) as u32;
            h = self.height();

            x_min = (self.width() - w) / 2;
            y_min = 0;
        }

        let rect = Rect::from_top_left(x_min as _, y_min as _, w, h);
        log::trace!(
            "fit aspect ratio {} in resolution {} -> {:?}",
            ratio,
            self,
            rect
        );
        rect
    }
}

impl fmt::Display for Resolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

impl fmt::Debug for Resolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl std::ops::Mul<u32> for Resolution {
    type Output = Self;

    fn mul(self, rhs: u32) -> Self::Output {
        assert!(rhs != 0, "attempt to multiply resolution with 0");
        Resolution {
            width: self.width * rhs,
            height: self.height * rhs,
        }
    }
}

/// Ratio of a width to a height of an image.
#[derive(PartialEq, Eq, Clone, Copy)]
pub struct AspectRatio {
    // Invariant: `width` and `height` are nonzero and as small as possible (ie. their GCD is 1).
    width: u32,
    height: u32,
}

impl AspectRatio {
    /// 1:1 aspect ratio.
    ///
    /// Common for CNN inputs.
    pub const SQUARE: Self = Self {
        width: 1,
        height: 1,
    };

    /// Creates the aspect ratio representing `width:height`.
    ///
    /// # Panics
    ///
    /// This function will panic if `width` or `height` are zero.
    pub fn new(width: u32, height: u32) -> Self {
        assert!(
            width != 0 && height != 0,
            "attempted to create an aspect ratio with 0 width or height"
        );
        let gcd = gcd(width, height);
        Self {
            width: width / gcd,
            height: height / gcd,
        }
    }

    /// Returns the `f32` corresponding to this ratio.
    #[inline]
    pub fn as_f32(&self) -> f32 {
        self.width as f32 / self.height as f32
    }
}

impl From<Resolution> for AspectRatio {
    fn from(res: Resolution) -> Self {
        Self::new(res.width(), res.height())
    }
}

impl fmt::Display for AspectRatio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.width, self.height)
    }
}

impl fmt::Debug for AspectRatio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

const fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b > 0 {
        let t = b;
        b = a % b;
        a = t;
    }

    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(6, 9), 3);
        assert_eq!(gcd(7, 13), 1);
        assert_eq!(1920 / gcd(1920, 1080), 16);
        assert_eq!(1080 / gcd(1920, 1080), 9);

        // degenerate case where one of the arguments is 0 - the other one will be returned
        assert_eq!(gcd(0, 7), 7);
        assert_eq!(gcd(7, 0), 7);
        assert_eq!(gcd(0, 0), 0);
    }

    #[test]
    fn test_aspect_ratio() {
        let ratio1 = AspectRatio::new(1920, 1080);
        let ratio2 = AspectRatio::new(1280, 720);
        assert_eq!(ratio1, ratio2);
        assert_eq!(ratio1.to_string(), "16:9");
        assert_eq!(ratio2.to_string(), "16:9");
    }

    #[test]
    fn test_fit_aspect_ratio() {
        assert_eq!(
            Resolution::new(16, 16).fit_aspect_ratio(AspectRatio::new(16, 8)),
            Rect::from_ranges(0..16, 4..12)
        );
        assert_eq!(
            Resolution::new(16, 16).fit_aspect_ratio(AspectRatio::new(8, 16)),
            Rect::from_ranges(4..12, 0..16)
        );
        assert_eq!(
            Resolution::new(16, 8).fit_aspect_ratio(AspectRatio::new(16, 8)),
            Rect::from_ranges(0..16, 0..8)
        );
    }
}
