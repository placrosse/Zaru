use zaru::{
    body::detection::PoseNetwork,
    detection::Detector,
    gui,
    timer::FpsCounter,
    video::webcam::{Webcam, WebcamOptions},
};

#[zaru::main]
fn main() -> anyhow::Result<()> {
    let mut detector = Detector::new(PoseNetwork);

    let mut fps = FpsCounter::new("body detection");
    for result in Webcam::open(WebcamOptions::default())? {
        let mut image = result?;

        let detections = detector.detect(&image);
        for detection in detections.iter() {
            detection.draw(&mut image);
        }

        gui::show_image("body detection", &image);

        fps.tick_with(detector.timers().into_iter());
    }

    Ok(())
}
