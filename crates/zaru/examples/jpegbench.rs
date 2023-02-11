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
    let timer = Timer::new("decode");
    let mut fps = FpsCounter::new("image");
    loop {
        let image = timer.time(|| Image::decode_jpeg(&image))?;
        fps.tick_with([&timer]);

        gui::show_image("jpegbench", &image);
    }
}
