mod facetracking;

use log::LevelFilter;
use mizaru::detector::Detector;
use mizaru::eye::EyeLandmarker;
use mizaru::image::{Color, Rect};
use mizaru::landmark::{self, Landmarker};
use mizaru::num::TotalF32;
use mizaru::procrustes::ProcrustesAnalyzer;
use mizaru::timer::{FpsCounter, Timer};
use mizaru::webcam::Webcam;
use mizaru::{gui, image, on_drop, pipeline, Error};

fn main() -> Result<(), Error> {
    let log_level = if cfg!(debug_assertions) {
        LevelFilter::Trace
    } else {
        LevelFilter::Debug
    };
    env_logger::Builder::new()
        .filter(Some(env!("CARGO_CRATE_NAME")), log_level)
        .filter(Some("wgpu"), LevelFilter::Warn)
        .init();

    let mut detector = Detector::new();
    let mut landmarker = Landmarker::new();
    let mut left_eye_landmarker = EyeLandmarker::new();
    let mut right_eye_landmarker = EyeLandmarker::new();
    let mut procrustes_analyzer = ProcrustesAnalyzer::new(
        landmark::reference_positions().map(|pos| (pos.x(), pos.y(), pos.z())),
    );

    let landmark_input_aspect = left_eye_landmarker.input_resolution().aspect_ratio();

    let mut webcam = Webcam::open()?;

    let (img_sender, img_recv) = pipeline::channel();
    let (face_img_sender, face_img_recv) = pipeline::channel();
    let (left_eye_img_sender, left_eye_img_recv) = pipeline::channel();
    let (right_eye_img_sender, right_eye_img_recv) = pipeline::channel();
    let (landmark_sender, landmark_recv) = pipeline::channel();
    let (left_eye_landmark_sender, left_eye_landmark_recv) = pipeline::channel();
    let (right_eye_landmark_sender, right_eye_landmark_recv) = pipeline::channel();

    // The detection pipeline uses a quite literal pipeline structure – different processing stages
    // happen in different threads, quite similar to how a pipelined CPU architecture works
    // (probably not the most helpful analogy unless you're a nerd, sorry).
    // By splitting the work up like this, what we get is a very simple to write multithreaded
    // pipeline with latency very close to a single-threaded design, but improved throughput (the
    // overhead is mostly just sending pipeline outputs through channels and waking the receiving
    // thread).
    // "Improved throughput" here means that this pipeline can easily run at 100 FPS or more, at
    // least on my workstation. We'll see if something like a Raspberry Pi is enough to run it at 60
    // FPS.
    // Also note that at the moment, *everything* runs on the CPU, a GPU is not used. I'd like to
    // change that at some point since it seems fun, but for now this works well enough.
    crossbeam::scope(|scope| {
        scope
            .builder()
            .name("Webcam Decoder".into())
            .spawn(|_| {
                let img_sender = img_sender.activate();

                let _guard = on_drop(|| log::info!("webcam thread exiting"));
                let mut fps = FpsCounter::new("webcam");
                loop {
                    let image = match webcam.read() {
                        Ok(image) => image,
                        Err(e) => {
                            log::error!("failed to fetch image from webcam: {e}");
                            break;
                        }
                    };
                    if img_sender.send(image).is_err() {
                        break;
                    }
                    fps.tick_with(webcam.timers());
                }
            })
            .unwrap();

        scope
            .builder()
            .name("Face Detector".into())
            .spawn(|_| {
                let _guard = on_drop(|| log::info!("detector thread exiting"));
                let mut fps = FpsCounter::new("detector");

                let face_img_sender = face_img_sender.activate();

                for mut image in img_recv.activate() {
                    let detections = detector.detect(&image);
                    log::trace!("{:?}", detections);

                    if let Some(target) = detections
                        .iter()
                        .max_by_key(|det| TotalF32(det.confidence()))
                    {
                        // TODO: rotate to align the eyes
                        if let Some(crop_rect) =
                            image.rect().intersection(&target.bounding_rect_loose())
                        {
                            let face = image.view(&crop_rect).to_image();
                            if face_img_sender.send(face).is_err() {
                                break;
                            }
                        }
                    }

                    for det in detections {
                        det.draw(&mut image);

                        #[allow(illegal_floating_point_literal_pattern)] // let me have fun
                        let color = match det.confidence() {
                            1.5.. => Color::GREEN,
                            0.5..=1.5 => Color::YELLOW,
                            _ => Color::RED,
                        };
                        image::draw_text(
                            &mut image,
                            det.bounding_rect_loose().x()
                                + (det.bounding_rect_loose().width() / 2) as i32,
                            det.bounding_rect_loose().y(),
                            &format!("conf={:.01}", det.confidence()),
                        )
                        .align_top()
                        .color(color);

                        let alignment_color = Color::from_rgb8(180, 180, 180);
                        let (x0, y0) = det.left_eye();
                        let (x1, y1) = det.right_eye();
                        image::draw_line(&mut image, x0, y0, x1, y1).color(alignment_color);
                        let rot = format!("{:.01}°", det.rotation_radians().to_degrees());
                        image::draw_text(&mut image, (x0 + x1) / 2, (y0 + y1) / 2 - 10, &rot)
                            .color(alignment_color);
                    }
                    gui::show_image("face_detect", &image);

                    fps.tick_with(detector.timers());
                }
            })
            .unwrap();

        scope
            .builder()
            .name("Landmarker".into())
            .spawn(|_| {
                let _guard = on_drop(|| log::info!("landmarking thread exiting"));
                let mut fps = FpsCounter::new("landmarker");
                let mut t_total = Timer::new("total");
                let mut t_procrustes = Timer::new("procrustes");

                let left_eye_img_sender = left_eye_img_sender.activate();
                let right_eye_img_sender = right_eye_img_sender.activate();
                let landmark_sender = landmark_sender.activate();

                for mut image in face_img_recv.activate() {
                    let guard = t_total.start();

                    let res = landmarker.compute(&image);
                    if res.face_confidence() >= 10.0 {
                        if landmark_sender.send(res.clone()).is_err() {
                            break;
                        }

                        use landmark::Idx::*;

                        let left = [
                            LeftEyeLeftCorner,
                            LeftEyeRightCorner,
                            LeftEyeBottom,
                            LeftEyeTop,
                        ];
                        let right = [
                            RightEyeLeftCorner,
                            RightEyeRightCorner,
                            RightEyeBottom,
                            RightEyeTop,
                        ];

                        let left = Rect::bounding(left.into_iter().map(|idx| {
                            let pos = res.landmark_position(idx.into());
                            (pos.x() as i32, pos.y() as i32)
                        }))
                        .unwrap();
                        let right = Rect::bounding(right.into_iter().map(|idx| {
                            let pos = res.landmark_position(idx.into());
                            (pos.x() as i32, pos.y() as i32)
                        }))
                        .unwrap();

                        const MARGIN: f32 = 0.5;
                        let left = left
                            .grow_rel(MARGIN, MARGIN, MARGIN, MARGIN)
                            .grow_to_fit_aspect(landmark_input_aspect);
                        let right = right
                            .grow_rel(MARGIN, MARGIN, MARGIN, MARGIN)
                            .grow_to_fit_aspect(landmark_input_aspect);

                        let left = image.rect().intersection(&left);
                        let right = image.rect().intersection(&right);
                        if let (Some(left), Some(right)) = (left, right) {
                            let left = image.view(&left).to_image();
                            let right = image.view(&right).to_image();
                            if left_eye_img_sender.send(left).is_err() {
                                break;
                            }
                            if right_eye_img_sender.send(right).is_err() {
                                break;
                            }
                        }
                    }

                    let procrustes_result = t_procrustes.time(|| {
                        procrustes_analyzer.analyze(res.raw_landmarks().positions().map(|pos| {
                            // Flip Y to bring us to canonical 3D coordinates (where Y points up).
                            // Only rotation matters, so we don't have to correct for the added
                            // translation.
                            (pos.x(), -pos.y(), pos.z())
                        }))
                    });

                    for pos in res.landmark_positions() {
                        image::draw_marker(&mut image, pos.x() as _, pos.y() as _).size(3);
                    }

                    #[allow(illegal_floating_point_literal_pattern)] // let me have fun
                    let color = match res.face_confidence() {
                        20.0.. => Color::GREEN,
                        10.0..=20.0 => Color::YELLOW,
                        _ => Color::RED,
                    };
                    let x = (image.width() / 2) as _;
                    image::draw_text(
                        &mut image,
                        x,
                        0,
                        &format!("conf={:.01}", res.face_confidence()),
                    )
                    .align_top()
                    .color(color);

                    let cx = (image.width() / 2) as i32;
                    let cy = (image.height() / 2) as i32;
                    image::draw_quaternion(
                        &mut image,
                        cx,
                        cy,
                        procrustes_result.rotation_as_quaternion(),
                    );

                    gui::show_image("raw_landmarks", &image);

                    drop(guard);
                    fps.tick_with(
                        [&t_total, &t_procrustes]
                            .into_iter()
                            .chain(landmarker.timers()),
                    );
                }
            })
            .unwrap();

        scope
            .builder()
            .name("Left Iris".into())
            .spawn(|_| {
                let _guard = on_drop(|| log::info!("left iris tracking thread exiting"));
                let mut fps = FpsCounter::new("left iris");

                let landmark_sender = left_eye_landmark_sender.activate();

                for mut left_img in left_eye_img_recv.activate() {
                    let left_marks = left_eye_landmarker.compute(&left_img);
                    if landmark_sender.send(left_marks.clone()).is_err() {
                        break;
                    }

                    left_marks.draw(&mut left_img);

                    gui::show_image("left_eye", &left_img);

                    fps.tick_with(left_eye_landmarker.timers());
                }
            })
            .unwrap();

        scope
            .builder()
            .name("Right Iris".into())
            .spawn(|_| {
                let _guard = on_drop(|| log::info!("right iris tracking thread exiting"));
                let mut fps = FpsCounter::new("right iris");

                let landmark_sender = right_eye_landmark_sender.activate();

                for mut right_img in right_eye_img_recv.activate() {
                    let right_marks = right_eye_landmarker.compute(&right_img.flip_horizontal());
                    right_marks.flip_horizontal_in_place();
                    if landmark_sender.send(right_marks.clone()).is_err() {
                        break;
                    }

                    right_marks.draw(&mut right_img);

                    gui::show_image("right_eye", &right_img);

                    fps.tick_with(right_eye_landmarker.timers());
                }
            })
            .unwrap();

        scope
            .builder()
            .name("Assembler".into())
            .spawn(|_| {
                let _guard = on_drop(|| log::info!("assembler thread exiting"));
                let mut fps = FpsCounter::new("assembler");

                let left_eye = left_eye_landmark_recv.activate();
                let right_eye = right_eye_landmark_recv.activate();

                // Receive face landmarks first to unblock the landmarking thread.
                for _face_landmarks in landmark_recv.activate() {
                    let _left = match left_eye.recv() {
                        Ok(lm) => lm,
                        Err(_) => break,
                    };
                    let _right = match right_eye.recv() {
                        Ok(lm) => lm,
                        Err(_) => break,
                    };
                    fps.tick();
                }
            })
            .unwrap();
    })
    .unwrap();

    // TODO take tracking result from end of thread pipeline and forward panics
    Ok(())
}
