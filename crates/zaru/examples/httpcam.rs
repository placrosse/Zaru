use std::net::SocketAddr;

use zaru::{gui, timer::FpsCounter, video::httpcam::HttpStream};

#[zaru::main]
fn main() -> anyhow::Result<()> {
    let Some(addr) = std::env::args().nth(1) else {
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
