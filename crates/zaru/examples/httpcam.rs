use std::net::SocketAddr;

use zaru::{image::gui, timer::FpsCounter, video::httpcam::HttpStream};

fn main() -> anyhow::Result<()> {
    zaru::init_logger!();

    let Some(addr) = std::env::args().skip(1).next() else {
        eprintln!("usage: httpcam <addr:port>");
        std::process::exit(1);
    };

    let addr = addr.parse::<SocketAddr>()?;

    let mut webcam = HttpStream::connect(addr)?;
    let mut fps = FpsCounter::new("webcam");
    loop {
        let image = webcam.read()?;
        fps.tick_with(webcam.timers());

        gui::show_image("webcam", &image);
    }
}
