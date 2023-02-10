use std::any::type_name;

use zaru::{
    detection::Detector,
    face::{
        detection::ShortRangeNetwork,
        landmark::multipie68::{self, LandmarkResult},
    },
    gui,
    image::{draw, Color},
    landmark::{Estimation, Estimator, Network},
    timer::FpsCounter,
    video::webcam::{Webcam, WebcamOptions},
};

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
            fps: FpsCounter::new(type_name::<L>().split("::").last().unwrap()),
        }
    }
}

fn main() -> anyhow::Result<()> {
    zaru::init_logger!();

    let mut detector = Detector::new(ShortRangeNetwork);
    let mut algos = [
        Algo::new(multipie68::PeppaFacialLandmark, Color::GREEN),
        Algo::new(multipie68::FaceOnnx, Color::RED),
    ];

    let webcam = Webcam::open(WebcamOptions::default())?;
    for image in webcam {
        let mut image = image?;
        if let Some(det) = detector.detect(&image).iter().next() {
            for algo in &mut algos {
                let rect = det
                    .bounding_rect()
                    .to_rect()
                    .grow_rel(0.15)
                    .grow_to_fit_aspect(algo.estimator.input_resolution().aspect_ratio().unwrap());
                draw::rect(&mut image, rect).color(algo.color);
                let mut view = image.view_mut(rect);
                let lms = algo.estimator.estimate(&view);
                for &[x, y, _] in lms.landmarks_mut().positions() {
                    draw::marker(&mut view, x, y).color(algo.color);
                }
                algo.fps.tick_with(algo.estimator.timers());
            }
        }
        gui::show_image("facemarks", &image);
    }

    Ok(())
}
