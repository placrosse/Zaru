//! Biblically accurate texture blitting benchmark.

use zaru::{
    face::{
        detection::Detector,
        landmark::{LandmarkTracker, TrackedFace},
    },
    gui,
    image::{Color, Image, Rect},
    num::TotalF32,
    timer::{FpsCounter, Timer},
    webcam::Webcam,
};

const W: u32 = 512;
const H: u32 = 512;

fn main() -> Result<(), zaru::Error> {
    zaru::init_logger!();

    let mut detector = Detector::default();
    let mut tracker = LandmarkTracker::new();

    let mut canvas = Image::new(W, H);
    let positions = (0..1000)
        .map(|_| {
            (
                fastrand::i32(0..W as i32),
                fastrand::i32(0..H as i32),
                fastrand::bool(),
            )
        })
        .collect::<Vec<_>>();

    let mut fps = FpsCounter::new("FPS");
    let mut blit_timer = Timer::new("blit");
    let mut webcam = Webcam::open()?;
    loop {
        let image = webcam.read()?;

        if tracker.tracked_face().is_none() {
            if let Some(det) = detector
                .detect(&image)
                .iter()
                .max_by_key(|det| TotalF32(det.confidence()))
            {
                tracker.set_tracked_face(TrackedFace::new(
                    det.bounding_rect_loose(),
                    det.rotation_radians(),
                ));
            }
        }

        if let Some(res) = tracker.track(&image) {
            let left_rect = res
                .landmarks()
                .left_eye()
                .move_by(res.view_rect().x(), res.view_rect().y());
            let right_rect = res
                .landmarks()
                .right_eye()
                .move_by(res.view_rect().x(), res.view_rect().y());
            let left_eye = image.view(&left_rect);
            let right_eye = image.view(&right_rect);

            canvas.clear(Color::BLACK);

            blit_timer.time(|| {
                for (x, y, right) in &positions {
                    let src_view = if *right { &right_eye } else { &left_eye };
                    let src_rect = if *right { &right_rect } else { &left_rect };
                    let mut dest_view = canvas.view_mut(&Rect::from_center(
                        *x,
                        *y,
                        src_rect.width(),
                        src_rect.height(),
                    ));
                    dest_view.blend_from(src_view);
                }
            });
            gui::show_image("biblical accuracy", &canvas);

            fps.tick_with(
                webcam
                    .timers()
                    .into_iter()
                    .chain(tracker.landmarker().timers())
                    .chain([&blit_timer]),
            );
        }
    }
}
