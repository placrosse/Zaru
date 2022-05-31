use zaru::{face::detector::Detector, gui, webcam::Webcam};

fn main() -> Result<(), zaru::Error> {
    let mut detector = Detector::new();
    let input_ratio = detector.input_resolution().aspect_ratio();

    let webcam = Webcam::open()?;
    for result in webcam {
        let mut image = result?;

        let view_rect = image.resolution().fit_aspect_ratio(input_ratio);
        let mut view = image.view_mut(&view_rect);

        for detection in detector.detect(&view) {
            detection.draw(&mut view);
        }

        gui::show_image("face detection", &image);
    }

    Ok(())
}
