use zaru::{
    gui,
    timer::FpsCounter,
    video::webcam::{Webcam, WebcamOptions},
};

#[zaru::main]
fn main() -> anyhow::Result<()> {
    let mut webcam = Webcam::open(WebcamOptions::default())?;
    let mut fps = FpsCounter::new("webcam");
    loop {
        let image = webcam.read()?;
        fps.tick_with(webcam.timers());

        gui::show_image("webcam", &image);
    }
}
