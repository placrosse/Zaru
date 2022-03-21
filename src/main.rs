use log::LevelFilter;
use mizaru::detector::Detector;
use mizaru::landmark::Landmarker;
use mizaru::num::TotalF32;
use mizaru::timer::FpsCounter;
use mizaru::webcam::Webcam;
use mizaru::{gui, image, on_drop, pipeline, Error};

fn main() -> Result<(), Error> {
    let log_level = if cfg!(debug_assertions) {
        LevelFilter::Trace
    } else {
        LevelFilter::Debug
    };
    env_logger::Builder::new()
        .filter(Some(env!("CARGO_CRATE_NAME")), log_level)
        .filter(Some("wgpu"), LevelFilter::Warn)
        .init();

    let mut detector = Detector::new();
    let mut landmarker = Landmarker::new();

    let landmark_input_res = landmarker.input_resolution();

    let mut webcam = Webcam::open()?;

    let (img_sender, img_recv) = pipeline::channel();
    let (face_img_sender, face_img_recv) = pipeline::channel();

    // The detection pipeline uses a quite literal pipeline structure – different processing stages
    // happen in different threads, quite similar to how a pipelined CPU architecture works
    // (probably not the most helpful analogy unless you're a nerd, sorry).
    // By splitting the work up like this, what we get is a very simple to write multithreaded
    // pipeline with latency very close to a single-threaded design, but improved throughput (the
    // overhead is mostly just sending pipeline outputs through channels and waking the receiving
    // thread).
    // "Improved throughput" here means that this pipeline can easily run at 100 FPS or more, at
    // least on my workstation. We'll see if something like a Raspberry Pi is enough to run it at 60
    // FPS.
    // Also note that at the moment, *everything* runs on the CPU, a GPU is not used. I'd like to
    // change that at some point since it seems fun, but for now this works well enough.
    crossbeam::scope(|scope| {
        scope
            .builder()
            .name("Webcam Decoder".into())
            .spawn(|_| {
                let mut img_sender = img_sender.activate();

                let _guard = on_drop(|| log::info!("webcam thread exiting"));
                let mut fps = FpsCounter::new("webcam");
                loop {
                    let image = match webcam.read() {
                        Ok(image) => image,
                        Err(e) => {
                            log::error!("failed to fetch image from webcam: {e}");
                            break;
                        }
                    };
                    if img_sender.send(image).is_err() {
                        break;
                    }
                    fps.tick_with(webcam.timers());
                }
            })
            .unwrap();

        scope
            .builder()
            .name("Face Detector".into())
            .spawn(|_| {
                let mut face_img_sender = face_img_sender.activate();

                let _guard = on_drop(|| log::info!("detector thread exiting"));
                let mut fps = FpsCounter::new("detector");

                for mut image in img_recv.activate() {
                    let detections = detector.detect(&image);
                    log::trace!("{:?}", detections);

                    if let Some(target) = detections
                        .iter()
                        .max_by_key(|det| TotalF32(det.confidence()))
                    {
                        // TODO: rotate to align the eyes
                        let rect = target.bounding_rect_loose();
                        let face = image
                            .view(&image.rect().intersection(&rect))
                            .aspect_aware_resize(landmark_input_res);
                        if face_img_sender.send(face).is_err() {
                            break;
                        }
                    }

                    for det in detections {
                        det.draw(&mut image);
                    }
                    gui::show_image("face_detect", &image);

                    fps.tick_with(detector.timers());
                }
            })
            .unwrap();

        scope
            .builder()
            .name("Landmarker".into())
            .spawn(|_| {
                let _guard = on_drop(|| log::info!("landmarking thread exiting"));
                let mut fps = FpsCounter::new("landmarker");

                for mut image in face_img_recv.activate() {
                    let res = landmarker.compute(&image);
                    for (x, y, _z) in res.landmarks() {
                        image::draw_marker(&mut image, *x as _, *y as _).size(3);
                    }
                    gui::show_image("landmarks", &image);

                    fps.tick_with(landmarker.timers());
                }
            })
            .unwrap();
    })
    .unwrap();

    // TODO take tracking result from end of thread pipeline and forward panics
    Ok(())
}
