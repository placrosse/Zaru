use super::*;
use Color as C;

fn mkimage<const W: usize, const H: usize>(data: [[Color; W]; H]) -> Image {
    let mut image = Image::new(W as u32, H as u32);
    for (y, row) in data.iter().enumerate() {
        for (x, color) in row.iter().enumerate() {
            image.set(x as u32, y as u32, *color);
        }
    }
    image
}

#[test]
fn view() {
    let image = mkimage([[C::RED, C::GREEN]]);

    let view = image.view(&Rect::from_corners((1, 0), (1, 0)));
    assert_eq!(view.width(), 1);
    assert_eq!(view.height(), 1);
    assert_eq!(view.get(0, 0), C::GREEN);

    let view = image.view(&Rect::from_corners((1, 0), (99, 99)));
    assert_eq!(view.width(), 1);
    assert_eq!(view.height(), 1);
    assert_eq!(view.get(0, 0), C::GREEN);
}

#[test]
fn blend() {
    let mut image = mkimage([[C::RED]]);
    let overlay = mkimage([[C::GREEN.with_alpha(0)]]);
    image.blend_from(&overlay).mode(BlendMode::Alpha);
    assert_eq!(image.get(0, 0), C::RED); // no change

    let mut image = mkimage([[C::RED]]);
    let overlay = mkimage([[C::GREEN.with_alpha(0)]]);
    image.blend_from(&overlay).mode(BlendMode::Overwrite);
    assert_eq!(image.get(0, 0), C::GREEN.with_alpha(0)); // overwrite blending

    let mut image = mkimage([[C::RED]]);
    let overlay = mkimage([[C::GREEN]]);
    image.blend_from(&overlay).mode(BlendMode::Alpha);
    assert_eq!(image.get(0, 0), C::GREEN); // alpha overwrite

    let mut image = mkimage([[C::RED]]);
    let overlay = mkimage([[C::from_rgb8(127, 0, 100)]]);
    image.blend_from(&overlay).mode(BlendMode::Multiply);
    assert_eq!(image.get(0, 0), C::from_rgb8(127, 0, 0));
    // FIXME: is this right? 127 seems correct only for linear colors

    // FIXME: very incomplete!
}
