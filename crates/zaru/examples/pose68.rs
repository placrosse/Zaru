//! Computes the head orientation from 2D Multi-PIE landmarks using Perspective-n-Point (PnP).
//!
//! Known bug: the recovered rotation sometimes flips by 180° around the Z axis, depending on how
//! the face is turned.

use nalgebra::UnitQuaternion;
use zaru::{
    detection::Detector,
    face::{
        detection::ShortRangeNetwork,
        landmark::multipie68::{self},
    },
    gui,
    image::Color,
    landmark::{Estimation, Estimator},
    video::webcam::{Webcam, WebcamOptions},
};
use zaru_image::draw;
use zaru_utils::pnp;

fn main() -> anyhow::Result<()> {
    zaru::init_logger!();

    let mut detector = Detector::new(ShortRangeNetwork);
    let mut estimator = Estimator::new(multipie68::FaceOnnx);
    let mut dlt = pnp::Dlt::new(multipie68::reference_positions());

    let webcam = Webcam::open(WebcamOptions::default())?;
    for image in webcam {
        let mut image = image?;
        if let Some(det) = detector.detect(&image).iter().next() {
            let rect = det
                .bounding_rect()
                .to_rect()
                .grow_rel(0.15)
                .grow_to_fit_aspect(estimator.input_resolution().aspect_ratio().unwrap());
            draw::rect(&mut image, rect).color(Color::RED);
            let mut view = image.view_mut(rect);
            let lms = estimator.estimate(&view);
            for &[x, y, _] in lms.landmarks_mut().positions() {
                draw::marker(&mut view, x as i32, y as i32).color(Color::RED);
            }

            let result = dlt.solve(lms.landmarks().positions().iter().map(|&[x, y, _]| [x, -y]));
            let rot = UnitQuaternion::from(*result.rotation());
            let (r, p, y) = rot.euler_angles();
            println!(
                "r={}°, p={}°, y={}°",
                r.to_degrees(),
                p.to_degrees(),
                y.to_degrees(),
            );

            let [x, y, _] = lms.landmarks().average();
            draw::quaternion(&mut view, x as _, y as _, rot);
        }
        gui::show_image("facemarks", &image);
    }

    Ok(())
}