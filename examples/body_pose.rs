use zaru::{body::detector::PoseDetector, gui, webcam::Webcam};

fn main() -> Result<(), zaru::Error> {
    let mut detector = PoseDetector::new();

    let webcam = Webcam::open()?;
    for result in webcam {
        let mut image = result?;

        for detection in detector.detect(&image) {
            detection.draw(&mut image);
        }

        gui::show_image("body pose", &image);
    }

    Ok(())
}
