//! This doesn't work yet.

use zaru::{
    gui,
    hand::detection::{FullNetwork, LiteNetwork, PalmDetector},
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
    let input_ratio = detector.input_resolution().aspect_ratio();

    let webcam = Webcam::open()?;
    for result in webcam {
        let mut image = result?;

        let view_rect = image.resolution().fit_aspect_ratio(input_ratio);
        for detection in detector.detect(image.view(&view_rect)) {
            detection.draw(&mut image);
        }

        gui::show_image("hand tracking", &image);
    }

    Ok(())
}
