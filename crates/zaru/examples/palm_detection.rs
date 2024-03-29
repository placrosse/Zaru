use zaru::{
    detection::Detector,
    gui,
    hand::detection,
    timer::FpsCounter,
    video::webcam::{Webcam, WebcamOptions},
};

const USE_FULL_DETECTION_NETWORK: bool = true;

#[zaru::main]
fn main() -> anyhow::Result<()> {
    let mut palm_detector = if USE_FULL_DETECTION_NETWORK {
        Detector::new(detection::FullNetwork)
    } else {
        Detector::new(detection::LiteNetwork)
    };

    let mut fps = FpsCounter::new("hand tracker");
    let mut webcam = Webcam::open(WebcamOptions::default())?;

    loop {
        let mut image = webcam.read()?;

        for det in palm_detector.detect(&image).iter() {
            det.draw(&mut image);
        }

        gui::show_image("palm detection", &image);

        fps.tick_with(webcam.timers().chain(palm_detector.timers()));
    }
}
