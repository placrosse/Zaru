use std::net::SocketAddr;

use zaru::{droidcam::Droidcam, gui, timer::FpsCounter};

fn main() -> Result<(), zaru::Error> {
    zaru::init_logger!();

    let Some(addr) = std::env::args().skip(1).next() else {
        eprintln!("usage: droidcam <addr>");
        std::process::exit(1);
    };

    let addr = addr
        .parse::<SocketAddr>()
        .or_else(|_| Ok::<_, zaru::Error>(SocketAddr::new(addr.parse()?, 4747)))?;

    let mut webcam = Droidcam::connect(addr)?;
    let mut fps = FpsCounter::new("webcam");
    loop {
        let image = webcam.read()?;
        fps.tick_with(webcam.timers());

        gui::show_image("webcam", &image);
    }
}
