mod facetracking;

use std::io;

use pawawwewism::{promise, Promise, PromiseHandle, Worker};
use zaru::face::detection::Detector;
use zaru::face::eye::{EyeLandmarks, EyeNetwork};
use zaru::face::landmark::mediapipe_facemesh::{self, LandmarkResult, MediaPipeFaceMesh};
use zaru::filter::ema::Ema;
use zaru::gui;
use zaru::image::{draw, AspectRatio, Image, Resolution, RotatedRect};
use zaru::landmark::{Estimator, LandmarkFilter, LandmarkTracker, Network};
use zaru::num::TotalF32;
use zaru::procrustes::ProcrustesAnalyzer;
use zaru::timer::{FpsCounter, Timer};
use zaru::video::webcam::{ParamPreference, Webcam, WebcamOptions};

const BLANK: bool = true;

fn main() -> anyhow::Result<()> {
    zaru::init_logger!();

    let eye_input_aspect = EyeNetwork.cnn().input_resolution().aspect_ratio().unwrap();

    let mut face_tracker = face_track_worker(eye_input_aspect)?;
    let mut left_eye_worker = eye_worker(Eye::Left)?;
    let mut right_eye_worker = eye_worker(Eye::Right)?;
    let mut assembler = assembler()?;

    let mut webcam = Webcam::open(
        WebcamOptions::default()
            .resolution(Resolution::RES_1080P)
            .fps(60)
            .prefer(ParamPreference::Framerate),
    )?;

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
        left_eye_worker.send(EyeParams {
            eye_image: left_eye_handle,
            landmarks: left_eye_lm,
        });
        right_eye_worker.send(EyeParams {
            eye_image: right_eye_handle,
            landmarks: right_eye_lm,
        });
        assembler.send(AssemblerParams {
            landmarks: landmarks_handle,
            left_eye: left_eye_lm_handle,
            right_eye: right_eye_lm_handle,
        });

        fps.tick_with(webcam.timers());
    }
}

struct AssemblerParams {
    landmarks: PromiseHandle<(Image, LandmarkResult)>,
    left_eye: PromiseHandle<EyeLandmarks>,
    right_eye: PromiseHandle<EyeLandmarks>,
}

fn assembler() -> Result<Worker<AssemblerParams>, io::Error> {
    let mut fps = FpsCounter::new("assembler");
    let t_procrustes = Timer::new("procrustes");

    let mut procrustes_analyzer =
        ProcrustesAnalyzer::new(mediapipe_facemesh::reference_positions());

    Worker::builder().name("assembler").spawn(
        move |AssemblerParams {
                  landmarks,
                  left_eye,
                  right_eye,
              }| {
            let Ok((mut image, face_landmark)) = landmarks.block() else { return };

            if BLANK {
                image = Image::new(image.width(), image.height());
            }

            let procrustes_result = t_procrustes.time(|| {
                procrustes_analyzer.analyze(face_landmark.landmarks().positions().iter().map(
                    |&[x, y, z]| {
                        // Flip Y to bring us to canonical 3D coordinates (where Y points up).
                        // Only rotation matters, so we don't have to correct for the added
                        // translation.
                        (x, -y, z)
                    },
                ))
            });

            let Ok(left) = left_eye.block() else { return };
            let Ok(right) = right_eye.block() else { return };

            let center = face_landmark.landmarks().average();
            draw::quaternion(
                &mut image,
                center[0] as i32,
                center[1] as i32,
                procrustes_result.rotation(),
            );

            face_landmark.draw(&mut image);
            left.draw(&mut image);
            right.draw(&mut image);
            gui::show_image("tracking", &image);

            fps.tick_with([&t_procrustes]);
        },
    )
}

struct FaceTrackParams {
    image: Image,
    landmarks: Promise<(Image, LandmarkResult)>,
    left_eye: Promise<(Image, RotatedRect)>,
    right_eye: Promise<(Image, RotatedRect)>,
}

fn face_track_worker(eye_input_aspect: AspectRatio) -> Result<Worker<FaceTrackParams>, io::Error> {
    let mut fps = FpsCounter::new("tracker");
    let t_total = Timer::new("total");

    let mut detector = Detector::default();
    let mut estimator = Estimator::new(MediaPipeFaceMesh);
    estimator.set_filter(LandmarkFilter::new(
        Ema::new(0.7),
        LandmarkResult::NUM_LANDMARKS,
    ));
    let mut tracker = LandmarkTracker::new(estimator.input_resolution().aspect_ratio().unwrap());
    let input_ratio = detector.input_resolution().aspect_ratio().unwrap();

    Worker::builder().name("face tracker").spawn(
        move |FaceTrackParams {
                  image,
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
                let view = image.view(view_rect);
                let detections = detector.detect(&view);

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

            if let Some(res) = tracker.track(&mut estimator, &image) {
                let left = res.estimation().left_eye();
                let right = res.estimation().right_eye();

                const MARGIN: f32 = 0.9;
                let left = left.grow_rel(MARGIN).grow_to_fit_aspect(eye_input_aspect);
                let right = right.grow_rel(MARGIN).grow_to_fit_aspect(eye_input_aspect);

                left_eye.fulfill((image.view(left).to_image(), left));
                right_eye.fulfill((image.view(right).to_image(), right));
                landmarks.fulfill((image, res.estimation().clone()));
            }

            drop(guard);
            fps.tick_with(
                [&t_total]
                    .into_iter()
                    .chain(detector.timers())
                    .chain(estimator.timers()),
            );
        },
    )
}

enum Eye {
    Left,
    Right,
}

struct EyeParams {
    eye_image: PromiseHandle<(Image, RotatedRect)>,
    landmarks: Promise<EyeLandmarks>,
}

fn eye_worker(eye: Eye) -> Result<Worker<EyeParams>, io::Error> {
    let name = match eye {
        Eye::Left => "left iris",
        Eye::Right => "right iris",
    };
    let mut fps = FpsCounter::new(name);
    let mut estimator = Estimator::new(EyeNetwork);
    estimator.set_filter(LandmarkFilter::new(
        Ema::new(0.7),
        EyeLandmarks::NUM_LANDMARKS,
    ));

    Worker::builder().name(name).spawn(
        move |EyeParams {
                  eye_image,
                  landmarks,
              }| {
            let Ok((image, rect)) = eye_image.block() else { return };
            let marks = match eye {
                Eye::Left => estimator.estimate(&image),
                Eye::Right => {
                    let marks = estimator.estimate(&image.flip_horizontal());
                    marks.flip_horizontal_in_place(image.resolution());
                    marks
                }
            };

            marks.landmarks_mut().map_positions(|[x, y, z]| {
                let [x, y] = rect.transform_out_f32(x, y);
                [x, y, z]
            });
            landmarks.fulfill(marks.clone());

            fps.tick_with(estimator.timers());
        },
    )
}
