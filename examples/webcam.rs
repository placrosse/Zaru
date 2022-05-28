use zaru::{timer::FpsCounter, webcam::Webcam};

fn main() -> Result<(), zaru::Error> {
    zaru::init_logger!();

    let mut webcam = Webcam::open()?;
    let mut fps = FpsCounter::new("webcam");
    loop {
        let _image = webcam.read()?;
        fps.tick_with(webcam.timers());
    }
}
