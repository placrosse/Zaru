use zaru::{
    gui,
    timer::FpsCounter,
    webcam::{Webcam, WebcamOptions},
};

fn main() -> Result<(), zaru::Error> {
    zaru::init_logger!();

    let mut webcam = Webcam::open(WebcamOptions::default())?;
    let mut fps = FpsCounter::new("webcam");
    loop {
        let image = webcam.read()?;
        fps.tick_with(webcam.timers());

        gui::show_image("webcam", &image);
    }
}
