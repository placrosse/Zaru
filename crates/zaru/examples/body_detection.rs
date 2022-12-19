use zaru::{
    body::detection::PoseDetector,
    gui,
    timer::FpsCounter,
    video::webcam::{Webcam, WebcamOptions},
};

fn main() -> anyhow::Result<()> {
    zaru::init_logger!();

    let mut detector = PoseDetector::new();

    let mut fps = FpsCounter::new("body detection");
    for result in Webcam::open(WebcamOptions::default())? {
        let mut image = result?;

        let detections = detector.detect(&image);
        for detection in detections {
            detection.draw(&mut image);
        }

        gui::show_image("body detection", &image);

        fps.tick_with(detector.timers().into_iter());
    }

    Ok(())
}
