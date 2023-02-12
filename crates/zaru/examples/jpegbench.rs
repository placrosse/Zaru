use anyhow::bail;
use zaru::{
    gui,
    image::Image,
    timer::{FpsCounter, Timer},
};

fn main() -> anyhow::Result<()> {
    zaru::init_logger!();

    let image = match std::env::args().nth(1) {
        Some(path) => std::fs::read(path)?,
        None => {
            bail!("usage: jpegbench <file>");
        }
    };
    let t_decode = Timer::new("decode");
    let t_display = Timer::new("display");
    let mut fps = FpsCounter::new("image");
    loop {
        let image = t_decode.time(|| Image::decode_jpeg(&image))?;
        t_display.time(|| gui::show_image("jpegbench", &image));

        fps.tick_with([&t_decode, &t_display]);
    }
}
