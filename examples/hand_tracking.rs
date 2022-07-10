use std::sync::Arc;

use zaru::{
    gui,
    hand::{detection, landmark, tracking::HandTracker},
    image::{self, Image},
    timer::FpsCounter,
    webcam::Webcam,
};

const USE_FULL_DETECTION_NETWORK: bool = true;
const USE_FULL_LANDMARK_NETWORK: bool = true;

/// If `true`, the hand tracking data is drawn on an empty image instead of the camera input. This
/// can make things easier to see.
const DRAW_ON_BLANK_IMAGE: bool = false;

fn main() -> Result<(), zaru::Error> {
    zaru::init_logger!();

    let mut tracker = match (USE_FULL_DETECTION_NETWORK, USE_FULL_LANDMARK_NETWORK) {
        (false, false) => HandTracker::new(detection::LiteNetwork, landmark::LiteNetwork),
        (false, true) => HandTracker::new(detection::LiteNetwork, landmark::FullNetwork),
        (true, false) => HandTracker::new(detection::FullNetwork, landmark::LiteNetwork),
        (true, true) => HandTracker::new(detection::FullNetwork, landmark::FullNetwork),
    };

    let mut fps = FpsCounter::new("hand tracker");
    let mut webcam = Webcam::open()?;

    let mut prev = Arc::new(webcam.read()?);
    tracker.track(prev.clone());
    loop {
        let image = Arc::new(webcam.read()?);
        tracker.track(image.clone());

        let mut blank_image = Image::new(image.width(), image.height());
        let target = if DRAW_ON_BLANK_IMAGE {
            &mut blank_image
        } else {
            Arc::make_mut(&mut prev)
        };
        for hand in tracker.hands() {
            let view = hand.view_rect();
            let rect = view.rect();
            image::draw_rotated_rect(target, view);
            image::draw_text(
                target,
                rect.center().0 as _,
                rect.y(),
                &format!("{:?}", hand.id()),
            )
            .align_top();
            hand.landmark_result().draw(target);
        }

        gui::show_image("hand tracking", target);

        prev = image;

        fps.tick_with(webcam.timers());
    }
}
