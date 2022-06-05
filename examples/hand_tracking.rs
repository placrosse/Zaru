use zaru::{
    gui,
    hand::{
        detection::{self, PalmDetector},
        landmark::{self, Landmarker},
    },
    image::{self, Color},
    num::TotalF32,
    timer::FpsCounter,
    webcam::Webcam,
};

const USE_FULL_DETECTION_NETWORK: bool = false;
const USE_FULL_LANDMARK_NETWORK: bool = true;

fn main() -> Result<(), zaru::Error> {
    zaru::init_logger!();

    let mut detector = if USE_FULL_DETECTION_NETWORK {
        PalmDetector::new(detection::FullNetwork)
    } else {
        PalmDetector::new(detection::LiteNetwork)
    };

    let mut landmarker = if USE_FULL_LANDMARK_NETWORK {
        Landmarker::new(landmark::FullNetwork)
    } else {
        Landmarker::new(landmark::LiteNetwork)
    };

    let mut fps = FpsCounter::new("hand tracker");
    let mut webcam = Webcam::open()?;
    loop {
        let mut image = webcam.read()?;

        let detections = detector.detect(&image);
        for detection in detections {
            detection.draw(&mut image);
        }

        if let Some(detection) = detections
            .iter()
            .max_by_key(|det| TotalF32(det.confidence()))
        {
            let grow_by = 1.5;
            let hand_rect = detection
                .bounding_rect()
                .grow_rel(grow_by, grow_by, grow_by, grow_by);
            image::draw_rect(&mut image, hand_rect).color(Color::BLUE);
            let mut hand_view = image.view_mut(&hand_rect);
            let landmarks = landmarker.compute(&hand_view);
            landmarks.draw(&mut hand_view);
        }

        gui::show_image("hand tracking", &image);

        fps.tick_with(
            webcam
                .timers()
                .chain(detector.timers())
                .chain(landmarker.timers()),
        );
    }
}
