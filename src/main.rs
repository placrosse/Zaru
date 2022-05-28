mod facetracking;

use log::LevelFilter;
use zaru::face::detector::{Detection, Detector};
use zaru::face::eye::EyeLandmarker;
use zaru::face::landmark::{self, LandmarkResult, LandmarkTracker, TrackedFace};
use zaru::image::{AsImageView, Color, Image, ImageView, ImageViewMut};
use zaru::num::TotalF32;
use zaru::procrustes::ProcrustesAnalyzer;
use zaru::resolution::AspectRatio;
use zaru::timer::{FpsCounter, Timer};
use zaru::webcam::Webcam;
use zaru::{defer::defer, gui, image, pipeline, Error};

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

    let mut left_eye_landmarker = EyeLandmarker::new();
    let mut right_eye_landmarker = EyeLandmarker::new();

    let eye_landmark_input_aspect = left_eye_landmarker.input_resolution().aspect_ratio();

    let (img_sender, img_recv) = pipeline::channel();
    let (left_eye_img_sender, left_eye_img_recv) = pipeline::channel();
    let (right_eye_img_sender, right_eye_img_recv) = pipeline::channel();
    let (landmark_sender, landmark_recv) = pipeline::channel();
    let (left_eye_landmark_sender, left_eye_landmark_recv) = pipeline::channel();
    let (right_eye_landmark_sender, right_eye_landmark_recv) = pipeline::channel();

    crossbeam::scope(|scope| {
        scope
            .builder()
            .name("Webcam Decoder".into())
            .spawn(|_| {
                let mut webcam = match Webcam::open() {
                    Ok(webcam) => webcam,
                    Err(e) => {
                        log::error!("failed to open webcam: {}", e);
                        return;
                    }
                };

                let img_sender = img_sender.activate();

                let _guard = defer(|| log::info!("webcam thread exiting"));
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
            .name("Face Tracker".into())
            .spawn(|_| {
                let _guard = defer(|| log::info!("tracker thread exiting"));
                let mut fps = FpsCounter::new("tracker");
                let mut t_total = Timer::new("total");
                let mut t_procrustes = Timer::new("procrustes");

                let mut procrustes_analyzer =
                    ProcrustesAnalyzer::new(landmark::reference_positions());
                let mut detector = Detector::new();
                let mut tracker = LandmarkTracker::new();
                let input_ratio = detector.input_resolution().aspect_ratio();

                let left_eye_img_sender = left_eye_img_sender.activate();
                let right_eye_img_sender = right_eye_img_sender.activate();
                let landmark_sender = landmark_sender.activate();

                for mut image in img_recv.activate() {
                    let guard = t_total.start();

                    if tracker.tracked_face().is_none() {
                        // Zoom into the camera image and perform detection there. This makes outer
                        // edges of the camera view unusable, but significantly improves the tracking
                        // distance.
                        let view_rect = image.resolution().fit_aspect_ratio(input_ratio);
                        let view = image.view_mut(&view_rect);
                        let detections = detector.detect(&view);

                        // FIXME: this draws over the image that we're about to compute landmarks on
                        draw_detections(view, detections);
                        gui::show_image("camera", &image);

                        if let Some(target) = detections
                            .iter()
                            .max_by_key(|det| TotalF32(det.confidence()))
                        {
                            let rect = target
                                .bounding_rect_loose()
                                .move_by(view_rect.x(), view_rect.y());
                            log::trace!("start tracking face at {:?}", rect);
                            tracker.set_tracked_face(TrackedFace::new(
                                rect,
                                target.rotation_radians(),
                            ));
                        }
                    }

                    if let Some(res) = tracker.track(&image) {
                        if landmark_sender.send(res.landmarks().clone()).is_err() {
                            break;
                        }

                        let mut face_image = image.view_mut(&res.view_rect());
                        let (left, right) = extract_eye_images(
                            face_image.as_view(),
                            res.landmarks(),
                            eye_landmark_input_aspect,
                        );

                        if left_eye_img_sender.send(left).is_err() {
                            break;
                        }
                        if right_eye_img_sender.send(right).is_err() {
                            break;
                        }

                        let procrustes_result = t_procrustes.time(|| {
                            procrustes_analyzer.analyze(
                                res.landmarks()
                                    .raw_landmarks()
                                    .positions()
                                    .map(|(x, y, z)| {
                                        // Flip Y to bring us to canonical 3D coordinates (where Y points up).
                                        // Only rotation matters, so we don't have to correct for the added
                                        // translation.
                                        (x, -y, z)
                                    }),
                            )
                        });

                        for (x, y, _z) in res.landmarks().landmark_positions() {
                            image::draw_marker(&mut face_image, x as _, y as _).size(3);
                        }

                        #[allow(illegal_floating_point_literal_pattern)] // let me have fun
                        let color = match res.landmarks().face_confidence() {
                            20.0.. => Color::GREEN,
                            10.0..=20.0 => Color::YELLOW,
                            _ => Color::RED,
                        };
                        let x = (face_image.width() / 2) as _;
                        image::draw_text(
                            &mut face_image,
                            x,
                            0,
                            &format!("lm_conf={:.01}", res.landmarks().face_confidence()),
                        )
                        .align_top()
                        .color(color);

                        let cx = (face_image.width() / 2) as i32;
                        let cy = (face_image.height() / 2) as i32;
                        image::draw_quaternion(
                            &mut face_image,
                            cx,
                            cy,
                            procrustes_result.rotation_as_quaternion(),
                        );

                        gui::show_image("camera", &image);
                    }

                    drop(guard);
                    fps.tick_with(
                        [&t_total, &t_procrustes]
                            .into_iter()
                            .chain(detector.timers()),
                    );
                }
            })
            .unwrap();

        scope
            .builder()
            .name("Left Iris".into())
            .spawn(|_| {
                let _guard = defer(|| log::info!("left iris tracking thread exiting"));
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
                let _guard = defer(|| log::info!("right iris tracking thread exiting"));
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
                let _guard = defer(|| log::info!("assembler thread exiting"));
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

fn extract_eye_images(
    face_image: ImageView<'_>,
    lm: &LandmarkResult,
    target_aspect: AspectRatio,
) -> (Image, Image) {
    let left = lm.left_eye();
    let right = lm.right_eye();

    const MARGIN: f32 = 0.9;
    let left = left
        .grow_rel(MARGIN, MARGIN, MARGIN, MARGIN)
        .grow_to_fit_aspect(target_aspect);
    let right = right
        .grow_rel(MARGIN, MARGIN, MARGIN, MARGIN)
        .grow_to_fit_aspect(target_aspect);

    let left = face_image.view(&left).to_image();
    let right = face_image.view(&right).to_image();
    (left, right)
}

fn draw_detections(mut target: ImageViewMut<'_>, detections: &[Detection]) {
    // FIXME move somewhere else
    for det in detections {
        det.draw(&mut target);

        #[allow(illegal_floating_point_literal_pattern)] // let me have fun
        let color = match det.confidence() {
            1.5.. => Color::GREEN,
            0.5..=1.5 => Color::YELLOW,
            _ => Color::RED,
        };
        image::draw_text(
            &mut target,
            det.bounding_rect_loose().x() + (det.bounding_rect_loose().width() / 2) as i32,
            det.bounding_rect_loose().y(),
            &format!("conf={:.01}", det.confidence()),
        )
        .align_top()
        .color(color);

        let alignment_color = Color::from_rgb8(180, 180, 180);
        let (x0, y0) = det.left_eye();
        let (x1, y1) = det.right_eye();
        image::draw_line(&mut target, x0, y0, x1, y1).color(alignment_color);
        let rot = format!("{:.01}°", det.rotation_radians().to_degrees());
        image::draw_text(&mut target, (x0 + x1) / 2, (y0 + y1) / 2 - 10, &rot)
            .color(alignment_color);
    }
}
