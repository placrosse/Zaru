use std::time::Instant;

use zaru::{
    anim::{Animation, AnimationFormat},
    face::detector::Detector,
    gui,
    webcam::Webcam,
};

fn main() -> Result<(), zaru::Error> {
    let animation = Animation::from_data(
        include_bytes!("../3rdparty/image/laughing_man.gif"),
        AnimationFormat::Gif,
    )?;
    let mut detector = Detector::new();

    let webcam = Webcam::open()?;
    let mut frames = animation.frames().cycle();
    let mut current_frame = frames.next().unwrap();
    let mut last_frame_start = Instant::now();
    for result in webcam {
        let mut image = result?;

        for detection in detector.detect(&image) {
            let dest_rect = detection
                .bounding_rect_raw()
                .grow_rel(0.4, 0.4, 0.4, 0.4)
                .grow_to_fit_aspect(current_frame.image_view().resolution().aspect_ratio());
            let mut dest = image.view_mut(&dest_rect);
            dest.blend_from(current_frame.image_view());
        }

        gui::show_image("laughing man", &image);

        let now = Instant::now();
        while current_frame.duration() < now.duration_since(last_frame_start) {
            last_frame_start += current_frame.duration();
            current_frame = frames.next().unwrap();
        }
    }

    Ok(())
}
