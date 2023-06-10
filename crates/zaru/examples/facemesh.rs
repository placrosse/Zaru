use zaru::{
    detection::Detector,
    face::{
        detection::{FullRangeNetwork, ShortRangeNetwork},
        landmark::mediapipe,
    },
    gui,
    image::{draw, Color, Image, Resolution},
    landmark::{Estimator, LandmarkTracker},
    num::TotalF32,
    timer::FpsCounter,
    video::webcam::{ParamPreference, Webcam, WebcamOptions},
};

const FULL_RANGE: bool = false;

#[zaru::main]
fn main() -> anyhow::Result<()> {
    let mut detector = if FULL_RANGE {
        Detector::new(FullRangeNetwork)
    } else {
        Detector::new(ShortRangeNetwork)
    };

    let mut fps = FpsCounter::new("face detector");
    let mut webcam = Webcam::open(
        WebcamOptions::default()
            .resolution(Resolution::RES_1080P)
            .prefer(ParamPreference::Framerate),
    )?;

    let landmarker = Estimator::new(mediapipe::FaceMeshV2);
    let mut tracker = LandmarkTracker::new(landmarker);

    loop {
        let image = webcam.read()?;
        let mut canvas = Image::new(image.width(), image.height());

        if let Some(result) = tracker.track(&image) {
            result.estimate().draw(&mut canvas);
        } else {
            // Tracking lost, run detection.

            let detections = detector.detect(&image);
            for detection in detections.iter() {
                detection.draw(&mut canvas);
            }

            if let Some(detection) = detections
                .iter()
                .max_by_key(|det| TotalF32(det.confidence()))
            {
                tracker.set_roi(detection.bounding_rect());
                draw::rect(&mut canvas, detection.bounding_rect()).color(Color::BLUE);
            }
        }

        gui::show_image("facemesh", &canvas);

        fps.tick_with(
            webcam
                .timers()
                .chain(detector.timers())
                .chain(tracker.timers()),
        );
    }
}
