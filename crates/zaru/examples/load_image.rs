use zaru::{
    image::{gui, Image},
    timer::{FpsCounter, Timer},
};

fn main() -> anyhow::Result<()> {
    zaru::init_logger!();

    let data = match std::env::args_os().skip(1).next() {
        Some(path) => std::fs::read(&path)?,
        None => {
            eprintln!("usage: load_image <path>");
            std::process::exit(1);
        }
    };

    let t_decode = Timer::new("decode");
    let mut fps = FpsCounter::new("load image");
    loop {
        let image = t_decode.time(|| Image::decode_jpeg(&data))?;

        gui::show_image("image", &image);

        fps.tick_with([&t_decode]);
    }
}
