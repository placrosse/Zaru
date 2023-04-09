use anyhow::anyhow;
use zaru::{gui, video::anim::Animation};

#[zaru::main]
fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow!("usage: animation <path-to-animation>"))?;
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
