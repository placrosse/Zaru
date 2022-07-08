use std::f32::consts::TAU;

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
fn view_data() {
    let image = mkimage([
        [C::YELLOW, C::WHITE, C::WHITE],
        [C::WHITE, C::RED, C::WHITE],
        [C::WHITE, C::WHITE, C::WHITE],
    ]);

    let view = ViewData::full(&image);
    assert_eq!(view.width(), 3);
    assert_eq!(view.height(), 3);
    assert_eq!(view.rect(), Rect::from_top_left(0, 0, 3, 3));
    assert_eq!(view.rect, view.rect().into());

    // Views of a single pixel:
    let center = view.view(Rect::from_top_left(1, 1, 1, 1));
    assert_eq!(center.rect(), Rect::from_top_left(0, 0, 1, 1));
    assert_eq!(center.rect, Rect::from_top_left(1, 1, 1, 1).into());

    let top_left = center.view(Rect::from_top_left(-1, -1, 2, 2));
    assert_eq!(top_left.rect(), Rect::from_top_left(0, 0, 2, 2));
    assert_eq!(top_left.rect, Rect::from_top_left(0, 0, 2, 2).into());

    let bottom_right = center.view(Rect::from_top_left(0, 0, 2, 2));
    assert_eq!(bottom_right.rect(), Rect::from_top_left(0, 0, 2, 2));
    assert_eq!(bottom_right.rect, Rect::from_top_left(1, 1, 2, 2).into());

    let larger = center.view(Rect::from_top_left(-1, -1, 3, 3));
    assert_eq!(larger.rect(), Rect::from_top_left(0, 0, 3, 3));
    assert_eq!(larger.rect, Rect::from_top_left(0, 0, 3, 3).into());

    // Views of 2x2 pixels:
    let bottom_right = view.view(Rect::from_top_left(1, 1, 2, 2));
    assert_eq!(bottom_right.rect(), Rect::from_top_left(0, 0, 2, 2));
    assert_eq!(bottom_right.rect, Rect::from_top_left(1, 1, 2, 2).into());

    let bottomer_righter = bottom_right.view(Rect::from_top_left(1, 1, 2, 2));
    assert_eq!(bottomer_righter.rect(), Rect::from_top_left(0, 0, 2, 2));
    assert_eq!(
        bottomer_righter.rect,
        Rect::from_top_left(2, 2, 2, 2).into()
    );
}

#[test]
fn rotated_views() {
    #[rustfmt::skip]
    let image = mkimage([
        [C::YELLOW, C::WHITE],
        [C::WHITE, C::RED],
    ]);

    let no_rot = image.view(RotatedRect::new(Rect::from_top_left(0, 0, 2, 2), 0.0));
    assert_eq!(no_rot.get(0, 0), C::YELLOW);
    assert_eq!(no_rot.get(1, 0), C::WHITE);
    assert_eq!(no_rot.get(0, 1), C::WHITE);
    assert_eq!(no_rot.get(1, 1), C::RED);

    let flip = image.view(RotatedRect::new(Rect::from_top_left(0, 0, 2, 2), TAU / 2.0));
    assert_eq!(flip.get(0, 0), C::RED);
    assert_eq!(flip.get(1, 0), C::WHITE);
    assert_eq!(flip.get(0, 1), C::WHITE);
    assert_eq!(flip.get(1, 1), C::YELLOW);

    let right_angle = image.view(RotatedRect::new(Rect::from_top_left(0, 0, 2, 2), TAU / 4.0));
    assert_eq!(right_angle.get(0, 0), C::WHITE);
    assert_eq!(right_angle.get(1, 0), C::RED);
    assert_eq!(right_angle.get(0, 1), C::YELLOW);
    assert_eq!(right_angle.get(1, 1), C::WHITE);

    // 2 chained 90° rotations
    let flip = right_angle.view(RotatedRect::new(Rect::from_top_left(0, 0, 2, 2), TAU / 4.0));
    assert_eq!(flip.get(0, 0), C::RED);
    assert_eq!(flip.get(1, 0), C::WHITE);
    assert_eq!(flip.get(0, 1), C::WHITE);
    assert_eq!(flip.get(1, 1), C::YELLOW);

    let bot_right = right_angle.view(RotatedRect::new(Rect::from_top_left(-1, 1, 2, 2), 0.0));
    assert_eq!(bot_right.get(0, 0), C::NULL);
    assert_eq!(bot_right.get(1, 0), C::YELLOW);
}

#[test]
fn view() {
    let image = mkimage([[C::RED, C::GREEN]]);

    let view = image.view(Rect::from_corners((1, 0), (1, 0)));
    assert_eq!(view.width(), 1);
    assert_eq!(view.height(), 1);
    assert_eq!(view.get(0, 0), C::GREEN);

    let view = image.view(Rect::from_corners((1, 0), (99, 99)));
    assert_eq!(view.width(), 99);
    assert_eq!(view.height(), 100);
    assert_eq!(view.get(0, 0), C::GREEN);
    assert_eq!(view.get(0, 1), C::NULL);
    assert_eq!(view.get(1, 0), C::NULL);
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
