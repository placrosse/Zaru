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
    assert_eq!(view.rect(), view.backed_area());

    // Views of a single pixel:
    let center = view.view(Rect::from_top_left(1, 1, 1, 1));
    assert_eq!(center.rect(), Rect::from_top_left(0, 0, 1, 1));
    assert_eq!(center.backed_area(), Rect::from_top_left(0, 0, 1, 1));
    assert_eq!(center.view_rect, Rect::from_top_left(1, 1, 1, 1));

    let top_left = center.view(Rect::from_top_left(-1, -1, 2, 2));
    assert_eq!(top_left.rect(), Rect::from_top_left(0, 0, 2, 2));
    assert_eq!(top_left.backed_area(), Rect::from_top_left(1, 1, 1, 1));
    assert_eq!(top_left.view_rect, Rect::from_top_left(1, 1, 1, 1));

    let bottom_right = center.view(Rect::from_top_left(0, 0, 2, 2));
    assert_eq!(bottom_right.rect(), Rect::from_top_left(0, 0, 2, 2));
    assert_eq!(bottom_right.backed_area(), Rect::from_top_left(0, 0, 1, 1));
    assert_eq!(bottom_right.view_rect, Rect::from_top_left(1, 1, 1, 1));

    let larger = center.view(Rect::from_top_left(-1, -1, 3, 3));
    assert_eq!(larger.rect(), Rect::from_top_left(0, 0, 3, 3));
    assert_eq!(larger.backed_area(), Rect::from_top_left(1, 1, 1, 1));
    assert_eq!(larger.view_rect, Rect::from_top_left(1, 1, 1, 1));

    // Views of 2x2 pixels:
    let bottom_right = view.view(Rect::from_top_left(1, 1, 2, 2));
    assert_eq!(bottom_right.rect(), Rect::from_top_left(0, 0, 2, 2));
    assert_eq!(bottom_right.backed_area(), Rect::from_top_left(0, 0, 2, 2));
    assert_eq!(bottom_right.view_rect, Rect::from_top_left(1, 1, 2, 2));

    let bottomer_righter = bottom_right.view(Rect::from_top_left(1, 1, 2, 2));
    assert_eq!(bottomer_righter.rect(), Rect::from_top_left(0, 0, 2, 2));
    assert_eq!(
        bottomer_righter.backed_area(),
        Rect::from_top_left(0, 0, 1, 1)
    );
    assert_eq!(bottomer_righter.view_rect, Rect::from_top_left(2, 2, 1, 1));
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
