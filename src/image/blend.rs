use super::{Color, ImageView, ImageViewMut};

/// Describes how to blend pixels together in a [`Blend`] operation.
pub enum BlendMode {
    /// All destination pixels will be overwritten with the corresponding source pixel.
    Overwrite,

    /// Performs alpha blending between source and destination pixels to make the source image
    /// appear in front of the destination image.
    Alpha,
}

/// A blending operation between two images.
///
/// Returned by [`Image::blend_from`][super::Image::blend_from] or [`ImageViewMut::blend_from`].
pub struct Blend<'a> {
    dest: ImageViewMut<'a>,
    src: ImageView<'a>,
    mode: BlendMode,
}

impl<'a> Blend<'a> {
    pub(super) fn new(dest: ImageViewMut<'a>, src: ImageView<'a>) -> Self {
        Self {
            dest,
            src,
            mode: BlendMode::Alpha,
        }
    }

    /// Sets the blend mode to use.
    pub fn mode(&mut self, mode: BlendMode) -> &mut Self {
        self.mode = mode;
        self
    }
}

impl Drop for Blend<'_> {
    fn drop(&mut self) {
        for dest_y in 0..self.dest.height() {
            for dest_x in 0..self.dest.width() {
                let src_x = ((dest_x as f32 + 0.5) / self.dest.width() as f32
                    * self.src.width() as f32) as u32;
                let src_y = ((dest_y as f32 + 0.5) / self.dest.height() as f32
                    * self.src.height() as f32) as u32;

                let src_pix = self.src.get(src_x, src_y);
                let dest_pix = self.dest.get(dest_x, dest_y);
                let result = match self.mode {
                    BlendMode::Overwrite => blend_overwrite(dest_pix, src_pix),
                    BlendMode::Alpha => blend_alpha(dest_pix, src_pix),
                };
                self.dest.set(dest_x, dest_y, result);
            }
        }
    }
}

fn blend_overwrite(_dest: Color, src: Color) -> Color {
    src
}

fn blend_alpha(dest: Color, src: Color) -> Color {
    fn blend_color(dest: f32, src: f32, dest_alpha: f32, src_alpha: f32, result_alpha: f32) -> f32 {
        (src * src_alpha + dest * dest_alpha * (1.0 - src_alpha)) / result_alpha
    }

    let dest = LinearColor::new(dest);
    let src = LinearColor::new(src);

    let result_alpha = src.a() + dest.a() * (1.0 - src.a());
    let r = blend_color(dest.r(), src.r(), dest.a(), src.a(), result_alpha);
    let g = blend_color(dest.g(), src.g(), dest.a(), src.a(), result_alpha);
    let b = blend_color(dest.b(), src.b(), dest.a(), src.a(), result_alpha);

    let result = LinearColor([r, g, b, result_alpha]);
    result.to_color()
}

struct LinearColor([f32; 4]);

impl LinearColor {
    fn new(color: Color) -> Self {
        fn to_rgb(srgb: f32) -> f32 {
            if srgb <= 0.04045 {
                srgb / 12.92
            } else {
                ((srgb + 0.055) / 1.055).powf(2.4)
            }
        }

        let (r, g, b, a) = (color.r(), color.g(), color.b(), color.a());
        let (r, g, b, a) = (f32::from(r), f32::from(g), f32::from(b), f32::from(a));
        let (r, g, b, a) = (r / 255.0, g / 255.0, b / 255.0, a / 255.0);

        let (r, g, b) = (to_rgb(r), to_rgb(g), to_rgb(b));

        Self([r, g, b, a])
    }

    fn to_color(&self) -> Color {
        fn to_srgb(rgb: f32) -> f32 {
            if rgb <= 0.0031308 {
                rgb * 12.92
            } else {
                1.055 * rgb.powf(1.0 / 2.4) - 0.055
            }
        }

        let (r, g, b, a) = (self.r(), self.g(), self.b(), self.a());
        let (r, g, b) = (to_srgb(r), to_srgb(g), to_srgb(b));
        let (r, g, b, a) = (r * 255.0, g * 255.0, b * 255.0, a * 255.0);
        let (r, g, b, a) = (r as u8, g as u8, b as u8, a as u8);

        Color([r, g, b, a])
    }

    fn r(&self) -> f32 {
        self.0[0]
    }

    fn g(&self) -> f32 {
        self.0[1]
    }

    fn b(&self) -> f32 {
        self.0[2]
    }

    fn a(&self) -> f32 {
        self.0[3]
    }
}
