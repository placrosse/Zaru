use std::{env, iter, thread, time::Instant};

use zaru::{
    body::{
        detection::{Keypoint, PoseNetwork},
        landmark::{FullNetwork, LiteNetwork},
    },
    detection::Detector,
    gui,
    image::{draw, Color, Image},
    landmark::{Estimator, LandmarkTracker},
    num::TotalF32,
    rect::Rect,
    timer::FpsCounter,
    video::{
        anim::Animation,
        webcam::{Webcam, WebcamOptions},
    },
};

const USE_FULL_NETWORK: bool = false;

fn main() -> anyhow::Result<()> {
    zaru::init_logger!();

    let image_path = env::args_os().nth(1);

    let video_source: Box<dyn Iterator<Item = anyhow::Result<Image>>> = match image_path {
        Some(path) => match Animation::from_path(&path) {
            Ok(a) => {
                let mut release_target = Instant::now();
                Box::new(
                    a.frames()
                        .map(|frame| (frame.image_view().to_image(), frame.duration()))
                        .collect::<Vec<_>>()
                        .into_iter()
                        .cycle()
                        .map(move |(frame, dur)| {
                            thread::sleep(release_target.saturating_duration_since(Instant::now()));
                            release_target = Instant::now() + dur;

                            Ok(frame)
                        }),
                )
            }
            Err(_) => {
                let image = Image::load(&path)?;
                Box::new(iter::repeat(image).map(Ok))
            }
        },
        None => Box::new(Webcam::open(WebcamOptions::default())?.into_iter()),
    };

    let mut detector = Detector::new(PoseNetwork);
    let landmarker = if USE_FULL_NETWORK {
        Estimator::new(FullNetwork)
    } else {
        Estimator::new(LiteNetwork)
    };
    let mut tracker = LandmarkTracker::new(landmarker);
    tracker.set_roi_padding(0.15);

    let mut fps = FpsCounter::new("body pose");
    for result in video_source {
        let mut image = result?;

        if let Some(result) = tracker.track(&image) {
            result.estimate().draw(&mut image);
        } else {
            // Tracking lost, run detection.

            let detections = detector.detect(&image);
            for detection in detections.iter() {
                detection.draw(&mut image);
            }

            if let Some(detection) = detections
                .iter()
                .max_by_key(|det| TotalF32(det.confidence()))
            {
                // TODO: rotation?
                let hips = detection.keypoints()[Keypoint::Hips as usize];
                let grow_by = 0.15;
                let body_rect =
                    Rect::bounding(detection.keypoints().iter().map(|kp| (kp.x(), kp.y())))
                        .unwrap()
                        .grow_move_center(hips.x(), hips.y())
                        .grow_rel(grow_by);
                tracker.set_roi(body_rect);
                draw::rect(&mut image, body_rect).color(Color::BLUE);
            }
        }

        gui::show_image("pose detection", &image);

        fps.tick_with(detector.timers().into_iter().chain(tracker.timers()));
    }

    Ok(())
}
