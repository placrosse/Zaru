use zaru::{anim::Animation, gui};

fn main() -> Result<(), zaru::Error> {
    let path = std::env::args()
        .skip(1)
        .next()
        .ok_or_else(|| format!("usage: anim <path-to-animation>"))?;
    let animation = Animation::from_path(&path)?;

    let time_per_frame = animation
        .frames()
        .map(|frame| frame.duration().as_secs_f32())
        .sum::<f32>()
        / animation.frames().len() as f32;
    let fps = 1.0 / time_per_frame;
    eprintln!("~{} FPS", fps);

    for frame in animation.frames().cycle() {
        gui::show_image("animation", &frame.image_view().to_image());
        std::thread::sleep(frame.duration());
    }

    Ok(())
}
