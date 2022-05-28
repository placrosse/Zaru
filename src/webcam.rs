//! V4L2 webcam access.
//!
//! Currently, only V4L2 `VIDEO_CAPTURE` devices yielding JFIF JPEG or Motion JPEG frames are
//! supported.

use linuxvideo::{
    format::{PixFormat, Pixelformat},
    stream::ReadStream,
    CapabilityFlags, Device,
};

use crate::{image::Image, timer::Timer};

/// A webcam yielding a stream of [`Image`]s.
pub struct Webcam {
    stream: ReadStream,
    width: u32,
    height: u32,
    t_dequeue: Timer,
    t_decode: Timer,
}

impl Webcam {
    /// Opens the first supported webcam found.
    ///
    /// This function can block for a significant amount of time while the webcam initializes (on
    /// the order of hundreds of milliseconds).
    pub fn open() -> Result<Self, crate::Error> {
        for res in linuxvideo::list()? {
            match res {
                Ok(dev) => match Self::open_impl(dev) {
                    Ok(Some(webcam)) => return Ok(webcam),
                    Ok(None) => {}
                    Err(e) => {
                        log::warn!("{}", e);
                    }
                },
                Err(e) => {
                    log::warn!("{}", e);
                }
            }
        }

        Err("no supported webcam device found".into())
    }

    fn open_impl(dev: Device) -> Result<Option<Self>, crate::Error> {
        let caps = dev.capabilities()?.device_capabilities();
        let path = dev.path()?;
        log::debug!("device {} capabilities: {:?}", path.display(), caps);

        if !caps.contains(CapabilityFlags::VIDEO_CAPTURE) {
            return Ok(None);
        }

        // TODO do actual format negotiation
        let capture = dev.video_capture(PixFormat::new(1920, 1080, Pixelformat::MJPG))?;

        let format = capture.format();
        let width = format.width();
        let height = format.height();
        match format.pixelformat() {
            Pixelformat::JPEG | Pixelformat::MJPG => {}
            e => return Err(format!("unsupported pixel format {}", e).into()),
        }

        let actual = capture.set_frame_interval(linuxvideo::Fract::new(1, 200))?;

        log::info!(
            "opened {}, {}x{} @ {:.1}Hz",
            path.display(),
            format.width(),
            format.height(),
            1.0 / actual.as_f32(),
        );

        let stream = capture.into_stream(2)?;

        Ok(Some(Self {
            stream,
            width,
            height,
            t_dequeue: Timer::new("dequeue"),
            t_decode: Timer::new("decode"),
        }))
    }

    /// Reads the next frame from the camera.
    ///
    /// If no frame is available, this method will block until one is.
    pub fn read(&mut self) -> Result<Image, crate::Error> {
        let dequeue_guard = self.t_dequeue.start();
        self.stream.dequeue(|buf| {
            drop(dequeue_guard);
            let image = match self.t_decode.time(|| Image::decode_jpeg(&buf)) {
                Ok(image) => image,
                Err(e) => {
                    // As sad as it is, but even high-quality webcams produce occasional corrupted
                    // MJPG frames, presumably due to USB data corruption (the alternative would be
                    // a silicon bug in the webcam's MJPG encoder, which is a possibility I chose to
                    // ignore for the sake of my own sanity).
                    log::error!("webcam decode error: {}", e);

                    // Hand back a blank image. The alternative would be to skip the image, which
                    // causes 2x latency spikes (OTOH, a blank image isn't going to result in any
                    // usable tracking data until next frame either).
                    // In the future we might have a better error type that lets the caller
                    // distinguish this case and recover in a better way.
                    Image::new(self.width, self.height)
                }
            };
            Ok(image)
        })
    }

    /// Returns a borrowing iterator over the frames produced by this webcam.
    pub fn iter_mut(&mut self) -> IterMut<'_> {
        IterMut { webcam: self }
    }

    /// Returns profiling timers for webcam access and decoding.
    pub fn timers(&self) -> impl IntoIterator<Item = &Timer> + '_ {
        [&self.t_dequeue, &self.t_decode]
    }
}

impl IntoIterator for Webcam {
    type Item = Result<Image, crate::Error>;
    type IntoIter = IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter { webcam: self }
    }
}

impl<'a> IntoIterator for &'a mut Webcam {
    type Item = Result<Image, crate::Error>;
    type IntoIter = IterMut<'a>;

    fn into_iter(self) -> Self::IntoIter {
        IterMut { webcam: self }
    }
}

/// An owned iterator over the frames captured by a [`Webcam`].
pub struct IntoIter {
    webcam: Webcam,
}

/// A borrowing iterator over the frames captured by a [`Webcam`].
pub struct IterMut<'a> {
    webcam: &'a mut Webcam,
}

impl Iterator for IntoIter {
    type Item = Result<Image, crate::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.webcam.read())
    }
}

impl Iterator for IterMut<'_> {
    type Item = Result<Image, crate::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.webcam.read())
    }
}
