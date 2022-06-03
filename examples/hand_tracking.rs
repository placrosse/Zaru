use zaru::{
    gui,
    hand::detection::{FullNetwork, LiteNetwork, PalmDetector},
    timer::FpsCounter,
    webcam::Webcam,
};

const FULL_RANGE: bool = false;

fn main() -> Result<(), zaru::Error> {
    zaru::init_logger!();

    let mut detector = if FULL_RANGE {
        PalmDetector::new(FullNetwork)
    } else {
        PalmDetector::new(LiteNetwork)
    };

    let mut fps = FpsCounter::new("hand tracker");
    let mut webcam = Webcam::open()?;
    loop {
        let mut image = webcam.read()?;

        for detection in detector.detect(&image) {
            detection.draw(&mut image);
        }

        gui::show_image("hand tracking", &image);

        fps.tick_with(webcam.timers().into_iter().chain(detector.timers()));
    }
}
