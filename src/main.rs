mod facetracking;

use std::io;

use pawawwewism::{promise, Promise, PromiseHandle, Worker};
use zaru::face::detection::Detector;
use zaru::face::eye::{EyeLandmarker, EyeLandmarks};
use zaru::face::landmark::mediapipe_facemesh::{self, LandmarkResult, Landmarker};
use zaru::image::Image;
use zaru::landmark::LandmarkTracker;
use zaru::num::TotalF32;
use zaru::procrustes::ProcrustesAnalyzer;
use zaru::resolution::AspectRatio;
use zaru::timer::{FpsCounter, Timer};
use zaru::webcam::Webcam;
use zaru::{gui, image, Error};

fn main() -> Result<(), Error> {
    zaru::init_logger!();

    let eye_input_aspect = EyeLandmarker::new()
        .input_resolution()
        .aspect_ratio()
        .unwrap();

    let mut face_tracker = face_track_worker(eye_input_aspect)?;

    let mut left_eye_worker = eye_worker(Eye::Left)?;
    let mut right_eye_worker = eye_worker(Eye::Right)?;

    let mut fps = FpsCounter::new("assembler");
    let mut assembler = Worker::builder().name("landmark assembler").spawn(
        move |(face_landmarks, left_eye, right_eye): (
            PromiseHandle<_>,
            PromiseHandle<_>,
            PromiseHandle<_>,
        )| {
            let _face_landmarks = match face_landmarks.block() {
                Ok(lms) => lms,
                Err(_) => return,
            };
            let _left = match left_eye.block() {
                Ok(eye) => eye,
                Err(_) => return,
            };
            let _right = match right_eye.block() {
                Ok(eye) => eye,
                Err(_) => return,
            };
            fps.tick();
        },
    )?;

    let mut webcam = Webcam::open()?;

    let mut fps = FpsCounter::new("webcam");
    loop {
        let image = webcam.read()?;
        let (landmarks, landmarks_handle) = promise();
        let (left_eye, left_eye_handle) = promise();
        let (right_eye, right_eye_handle) = promise();
        let (left_eye_lm, left_eye_lm_handle) = promise();
        let (right_eye_lm, right_eye_lm_handle) = promise();
        face_tracker.send(FaceTrackParams {
            image,
            landmarks,
            left_eye,
            right_eye,
        });
        left_eye_worker.send((left_eye_handle, left_eye_lm));
        right_eye_worker.send((right_eye_handle, right_eye_lm));
        assembler.send((landmarks_handle, left_eye_lm_handle, right_eye_lm_handle));

        fps.tick_with(webcam.timers());
    }
}

struct FaceTrackParams {
    image: Image,
    landmarks: Promise<LandmarkResult>,
    left_eye: Promise<Image>,
    right_eye: Promise<Image>,
}

fn face_track_worker(eye_input_aspect: AspectRatio) -> Result<Worker<FaceTrackParams>, io::Error> {
    let mut fps = FpsCounter::new("tracker");
    let mut t_total = Timer::new("total");
    let mut t_procrustes = Timer::new("procrustes");

    let mut procrustes_analyzer =
        ProcrustesAnalyzer::new(mediapipe_facemesh::reference_positions());
    let mut detector = Detector::default();
    let mut landmarker = Landmarker::new();
    let mut tracker = LandmarkTracker::new(landmarker.input_resolution().aspect_ratio().unwrap());
    let input_ratio = detector.input_resolution().aspect_ratio().unwrap();

    Worker::builder().name("face tracker").spawn(
        move |FaceTrackParams {
                  mut image,
                  landmarks,
                  left_eye,
                  right_eye,
              }| {
            let guard = t_total.start();

            if tracker.roi().is_none() {
                // Zoom into the camera image and perform detection there. This makes outer
                // edges of the camera view unusable, but significantly improves the tracking
                // distance.
                let view_rect = image.resolution().fit_aspect_ratio(input_ratio);
                let mut view = image.view_mut(view_rect);
                let detections = detector.detect(&view);

                // FIXME: this draws over the image that we're about to compute landmarks on
                for det in detections {
                    det.draw(&mut view);
                }
                gui::show_image("camera", &image);

                if let Some(target) = detections
                    .iter()
                    .max_by_key(|det| TotalF32(det.confidence()))
                {
                    let rect = target
                        .bounding_rect_loose()
                        .move_by(view_rect.x(), view_rect.y());
                    log::trace!("start tracking face at {:?}", rect);
                    tracker.set_roi(rect);
                }
            }

            if let Some(res) = tracker.track(&mut landmarker, &image) {
                landmarks.fulfill(res.estimation().clone());

                image::draw_rotated_rect(&mut image, res.view_rect());

                let left = res.estimation().left_eye();
                let right = res.estimation().right_eye();

                const MARGIN: f32 = 0.9;
                let left = left.grow_rel(MARGIN).grow_to_fit_aspect(eye_input_aspect);
                let right = right.grow_rel(MARGIN).grow_to_fit_aspect(eye_input_aspect);

                left_eye.fulfill(image.view(left).to_image());
                right_eye.fulfill(image.view(right).to_image());

                let procrustes_result = t_procrustes.time(|| {
                    procrustes_analyzer.analyze(
                        res.estimation()
                            .raw_landmarks()
                            .positions()
                            .iter()
                            .map(|&[x, y, z]| {
                                // Flip Y to bring us to canonical 3D coordinates (where Y points up).
                                // Only rotation matters, so we don't have to correct for the added
                                // translation.
                                (x, -y, z)
                            }),
                    )
                });

                res.estimation().draw(&mut image);

                let (cx, cy) = res.view_rect().center();
                image::draw_quaternion(
                    &mut image,
                    cx as i32,
                    cy as i32,
                    procrustes_result.rotation_as_quaternion(),
                );

                gui::show_image("camera", &image);
            }

            drop(guard);
            fps.tick_with(
                [&t_total, &t_procrustes]
                    .into_iter()
                    .chain(detector.timers())
                    .chain(landmarker.timers()),
            );
        },
    )
}

enum Eye {
    Left,
    Right,
}

type IrisWorkerParams = (PromiseHandle<Image>, Promise<EyeLandmarks>);

fn eye_worker(eye: Eye) -> Result<Worker<IrisWorkerParams>, io::Error> {
    let name = match eye {
        Eye::Left => "left iris",
        Eye::Right => "right iris",
    };
    let mut landmarker = EyeLandmarker::new();
    let mut fps = FpsCounter::new(name);

    Worker::builder()
        .name(name)
        .spawn(move |(image_handle, promise): IrisWorkerParams| {
            let mut image = match image_handle.block() {
                Ok(image) => image,
                Err(_) => return,
            };
            let marks = match eye {
                Eye::Left => landmarker.compute(&image),
                Eye::Right => {
                    let marks = landmarker.compute(&image.flip_horizontal());
                    marks.flip_horizontal_in_place();
                    marks
                }
            };

            promise.fulfill(marks.clone());

            marks.draw(&mut image);

            gui::show_image(name, &image);

            fps.tick_with(landmarker.timers());
        })
}
