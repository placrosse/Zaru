use std::any::type_name;

use zaru::{
    face::{
        detection::Detector,
        landmark::multipie68::{self, LandmarkNetwork, Landmarker},
    },
    gui,
    image::{self, Color},
    timer::FpsCounter,
    webcam::Webcam,
};

struct Algo {
    landmarker: Landmarker,
    color: Color,
    fps: FpsCounter,
}

impl Algo {
    fn new<L: LandmarkNetwork>(network: L, color: Color) -> Self {
        Self {
            landmarker: Landmarker::new(network),
            color,
            fps: FpsCounter::new(type_name::<L>()),
        }
    }
}

fn main() -> Result<(), zaru::Error> {
    zaru::init_logger!();

    let mut detector = Detector::default();
    let mut algos = [
        Algo::new(multipie68::PeppaFacialLandmark, Color::GREEN),
        Algo::new(multipie68::FaceOnnx, Color::RED),
    ];

    let webcam = Webcam::open()?;
    for image in webcam {
        let mut image = image?;
        if let Some(det) = detector.detect(&image).first() {
            for algo in &mut algos {
                let rect = det
                    .bounding_rect_raw()
                    .grow_rel(0.15)
                    .grow_to_fit_aspect(algo.landmarker.input_resolution().aspect_ratio().unwrap());
                image::draw_rect(&mut image, rect).color(algo.color);
                let mut view = image.view_mut(rect);
                let lms = algo.landmarker.compute(&view);
                for &[x, y, _] in lms.positions() {
                    image::draw_marker(&mut view, x as i32, y as i32).color(algo.color);
                }
                algo.fps.tick_with(algo.landmarker.timers());
            }
        }
        gui::show_image("facemarks", &image);
    }

    Ok(())
}
