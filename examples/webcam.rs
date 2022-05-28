use zaru::{timer::FpsCounter, webcam::Webcam, gui};

fn main() -> Result<(), zaru::Error> {
    zaru::init_logger!();

    let mut webcam = Webcam::open()?;
    let mut fps = FpsCounter::new("webcam");
    loop {
        let image = webcam.read()?;
        fps.tick_with(webcam.timers());

        gui::show_image("webcam", &image);
    }
}
