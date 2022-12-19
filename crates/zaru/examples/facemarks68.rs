use std::any::type_name;

use zaru::{
    face::{
        detection::Detector,
        landmark::multipie68::{self, LandmarkResult},
    },
    gui,
    image::Color,
    landmark::{Estimation, Estimator, Network},
    timer::FpsCounter,
    video::webcam::{Webcam, WebcamOptions},
};
use zaru_image::draw;

struct Algo {
    estimator: Estimator<LandmarkResult>,
    color: Color,
    fps: FpsCounter,
}

impl Algo {
    fn new<L: Network<Output = LandmarkResult>>(network: L, color: Color) -> Self {
        Self {
            estimator: Estimator::new(network),
            color,
            fps: FpsCounter::new(type_name::<L>()),
        }
    }
}

fn main() -> anyhow::Result<()> {
    zaru::init_logger!();

    let mut detector = Detector::default();
    let mut algos = [
        Algo::new(multipie68::PeppaFacialLandmark, Color::GREEN),
        Algo::new(multipie68::FaceOnnx, Color::RED),
    ];

    let webcam = Webcam::open(WebcamOptions::default())?;
    for image in webcam {
        let mut image = image?;
        if let Some(det) = detector.detect(&image).first() {
            for algo in &mut algos {
                let rect = det
                    .bounding_rect_raw()
                    .grow_rel(0.15)
                    .grow_to_fit_aspect(algo.estimator.input_resolution().aspect_ratio().unwrap());
                draw::rect(&mut image, rect).color(algo.color);
                let mut view = image.view_mut(rect);
                let lms = algo.estimator.estimate(&view);
                for &[x, y, _] in lms.landmarks_mut().positions() {
                    draw::marker(&mut view, x as i32, y as i32).color(algo.color);
                }
                algo.fps.tick_with(algo.estimator.timers());
            }
        }
        gui::show_image("facemarks", &image);
    }

    Ok(())
}
