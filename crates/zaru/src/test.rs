use std::sync::OnceLock;

use crate::image::Image;

pub fn sad_linus_full() -> &'static Image {
    static IMG: OnceLock<Image> = OnceLock::new();
    IMG.get_or_init(|| {
        Image::load(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../3rdparty/img/sad_linus.jpg"
        ))
        .unwrap()
    })
}

pub fn sad_linus_cropped() -> &'static Image {
    static IMG: OnceLock<Image> = OnceLock::new();
    IMG.get_or_init(|| {
        Image::load(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../3rdparty/img/sad_linus_cropped.jpg"
        ))
        .unwrap()
    })
}
