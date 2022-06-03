use std::{env, iter, thread, time::Instant};

use zaru::{
    anim::Animation,
    body::{
        detection::PoseDetector,
        landmark::{FullNetwork, Landmarker, LiteNetwork},
    },
    gui,
    image::{Image, Rect},
    num::TotalF32,
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

    for result in video_source {
        let mut image = result?;

        let detections = detector.detect(&image);
        for detection in detections {
            detection.draw(&mut image);
        }

        gui::show_image("pose detection", &image);

        let detection = match detections
            .iter()
            .max_by_key(|det| TotalF32(det.confidence()))
        {
            Some(det) => det,
            None => continue,
        };

        let grow_by = 0.75;
        let body_rect = Rect::bounding(detection.keypoints())
            .unwrap()
            .grow_to_fit_aspect(landmarker.input_resolution().aspect_ratio())
            .grow_rel(grow_by, grow_by, grow_by, grow_by);
        let body_view = image.view(&body_rect);
        let _landmarks = landmarker.compute(&body_view);
        todo!()
    }

    Ok(())
}
