use once_cell::sync::Lazy;

use crate::image::Image;

pub fn sad_linus_full() -> &'static Image {
    static IMG: Lazy<Image> = Lazy::new(|| {
        Image::load(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/3rdparty/img/sad_linus.jpg"
        ))
        .unwrap()
    });
    &IMG
}

pub fn sad_linus_cropped() -> &'static Image {
    static IMG: Lazy<Image> = Lazy::new(|| {
        Image::load(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/3rdparty/img/sad_linus_cropped.jpg"
        ))
        .unwrap()
    });
    &IMG
}
