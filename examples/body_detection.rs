use zaru::{body::detection::PoseDetector, gui, timer::FpsCounter, webcam::Webcam};

fn main() -> Result<(), zaru::Error> {
    zaru::init_logger!();

    let mut detector = PoseDetector::new();

    let mut fps = FpsCounter::new("body detection");
    for result in Webcam::open()? {
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
