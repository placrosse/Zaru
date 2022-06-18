use std::{env, iter, thread, time::Instant};

use zaru::{
    anim::Animation,
    body::{
        detection::PoseDetector,
        landmark::{FullNetwork, Landmarker, LiteNetwork},
    },
    gui,
    image::{self, Color, Image, Rect},
    num::TotalF32,
    timer::FpsCounter,
    webcam::Webcam,
};

const USE_FULL_NETWORK: bool = false;

fn main() -> Result<(), zaru::Error> {
    zaru::init_logger!();

    let image_path = env::args_os().skip(1).next();

    let video_source: Box<dyn Iterator<Item = zaru::Result<Image>>> = match image_path {
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
        None => Box::new(Webcam::open()?.into_iter()),
    };

    let mut detector = PoseDetector::new();
    let mut landmarker = if USE_FULL_NETWORK {
        Landmarker::new(FullNetwork)
    } else {
        Landmarker::new(LiteNetwork)
    };

    let mut fps = FpsCounter::new("body pose");
    for result in video_source {
        let mut image = result?;

        let detections = detector.detect(&image);
        for detection in detections {
            detection.draw(&mut image);
        }

        if let Some(detection) = detections
            .iter()
            .max_by_key(|det| TotalF32(det.confidence()))
        {
            let hips = detection.keypoint_hips();
            let grow_by = 0.15;
            let body_rect = Rect::bounding(detection.keypoints())
                .unwrap()
                .grow_move_center(hips.0, hips.1)
                .grow_to_fit_aspect(landmarker.input_resolution().aspect_ratio().unwrap())
                .grow_rel(grow_by);
            image::draw_rect(&mut image, body_rect).color(Color::BLUE);
            let mut body_view = image.view_mut(&body_rect);
            let landmarks = landmarker.compute(&body_view);
            landmarks.draw(&mut body_view);
        }

        gui::show_image("pose detection", &image);

        fps.tick_with(detector.timers().into_iter().chain(landmarker.timers()));
    }

    Ok(())
}
