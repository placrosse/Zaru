use zaru::{face::detector::Detector, gui, webcam::Webcam};

fn main() -> Result<(), zaru::Error> {
    let mut detector = Detector::new();

    let webcam = Webcam::open()?;
    for result in webcam {
        let mut image = result?;

        for detection in detector.detect(&image) {
            detection.draw(&mut image);
        }

        gui::show_image("face detection", &image);
    }

    Ok(())
}
