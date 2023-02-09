use zaru::{
    detection::Detector,
    face::detection::{FullRangeNetwork, ShortRangeNetwork},
    image::gui,
    timer::FpsCounter,
    video::webcam::{Webcam, WebcamOptions},
};

const FULL_RANGE: bool = false;

fn main() -> anyhow::Result<()> {
    zaru::init_logger!();

    let mut detector = if FULL_RANGE {
        Detector::new(FullRangeNetwork)
    } else {
        Detector::new(ShortRangeNetwork)
    };
    let input_ratio = detector.input_resolution().aspect_ratio().unwrap();

    let mut fps = FpsCounter::new("face detector");
    let mut webcam = Webcam::open(WebcamOptions::default())?;
    loop {
        let mut image = webcam.read()?;

        let view_rect = image.resolution().fit_aspect_ratio(input_ratio);
        let mut view = image.view_mut(view_rect);

        for detection in detector.detect(&view).iter() {
            detection.draw(&mut view);
        }

        gui::show_image("face detection", &image);

        fps.tick_with(webcam.timers().chain(detector.timers()));
    }
}
