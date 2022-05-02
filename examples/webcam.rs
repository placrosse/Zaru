use log::LevelFilter;
use zaru::{timer::FpsCounter, webcam::Webcam};

fn main() -> Result<(), zaru::Error> {
    env_logger::Builder::new()
        .filter(Some(env!("CARGO_CRATE_NAME")), LevelFilter::Debug)
        .filter(Some("zaru"), LevelFilter::Debug)
        .filter(Some("wgpu"), LevelFilter::Warn)
        .init();

    let mut webcam = Webcam::open()?;
    let mut fps = FpsCounter::new("webcam");
    loop {
        let _image = webcam.read()?;
        fps.tick_with(webcam.timers());
    }
}
