use std::sync::Arc;

use zaru::{
    gui,
    hand::{detection, landmark, tracking::HandTracker},
    image::{draw, Image},
    timer::FpsCounter,
    video::webcam::{Webcam, WebcamOptions},
};

const USE_FULL_DETECTION_NETWORK: bool = true;
const USE_FULL_LANDMARK_NETWORK: bool = true;

/// If `true`, the hand tracking data is drawn on an empty image instead of the camera input. This
/// can make things easier to see.
const DRAW_ON_BLANK_IMAGE: bool = false;

#[zaru::main]
fn main() -> anyhow::Result<()> {
    let mut tracker = match (USE_FULL_DETECTION_NETWORK, USE_FULL_LANDMARK_NETWORK) {
        (false, false) => HandTracker::new(detection::LiteNetwork, landmark::LiteNetwork),
        (false, true) => HandTracker::new(detection::LiteNetwork, landmark::FullNetwork),
        (true, false) => HandTracker::new(detection::FullNetwork, landmark::LiteNetwork),
        (true, true) => HandTracker::new(detection::FullNetwork, landmark::FullNetwork),
    };

    let mut fps = FpsCounter::new("hand tracker");
    let mut webcam = Webcam::open(WebcamOptions::default().fps(60))?;

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
            draw::rotated_rect(target, view);
            draw::text(
                target,
                rect.center().x,
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
