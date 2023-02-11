//! V4L2 webcam access.
//!
//! Currently, only V4L2 `VIDEO_CAPTURE` devices yielding JFIF JPEG or Motion JPEG frames are
//! supported.

use std::{cmp::Reverse, env};

use crate::image::{Image, Resolution};
use crate::timer::Timer;
use anyhow::bail;
use linuxvideo::{
    format::{FrameIntervals, FrameSizes, PixFormat, Pixelformat},
    stream::ReadStream,
    BufType, CapabilityFlags, Device, Fract,
};

/// Indicates whether to prefer a higher resolution or frame rate.
///
/// By default, [`ParamPreference::Resolution`] is used, selecting the maximum resolution at the
/// desired frame rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParamPreference {
    /// Prefer increased resolution over higher frame rates.
    Resolution,
    /// Prefer higher frame rate over higher image resolution.
    Framerate,
}

impl Default for ParamPreference {
    #[inline]
    fn default() -> Self {
        Self::Resolution
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct FramePrefs {
    resolution: Option<Resolution>,
    fps: Option<u32>,
    pref: ParamPreference,
}

/// Format negotiation options.
#[derive(Default)]
pub struct WebcamOptions {
    name: Option<String>,
    frame: FramePrefs,
}

impl WebcamOptions {
    /// Sets the name of the webcam device to open.
    ///
    /// If no webcam with the given name can be found, opening the webcam will result in an error.
    #[inline]
    pub fn name(self, name: impl Into<String>) -> Self {
        Self {
            name: Some(name.into()),
            ..self
        }
    }

    /// Sets the desired image resolution.
    ///
    /// A lower resolution might be selected if the webcam cannot deliver the desired resolution.
    #[inline]
    pub fn resolution(mut self, resolution: Resolution) -> Self {
        self.frame.resolution = Some(resolution);
        self
    }

    /// Sets the desired frame rate.
    ///
    /// A lower frame rate might be selected if the webcam cannot deliver the desired resolution.
    #[inline]
    pub fn fps(mut self, fps: u32) -> Self {
        self.frame.fps = Some(fps);
        self
    }

    /// Selects whether to prefer a higher resolution or frame rate.
    ///
    /// When the camera cannot deliver the desired frame rate or resolution, this parameter controls
    /// which one will be maintained.
    ///
    /// If the camera *can* deliver the desired frame rate and resolution, this parameter controls
    /// which camera parameter will be maximized while keeping the other at its desired
    /// configuration value.
    #[inline]
    pub fn prefer(mut self, pref: ParamPreference) -> Self {
        self.frame.pref = pref;
        self
    }
}

#[derive(Clone, Copy)]
struct FrameFormat {
    resolution: Resolution,
    frame_interval: Fract,
}

fn negotiate_format(device: &Device, mut prefs: FramePrefs) -> anyhow::Result<(PixFormat, Fract)> {
    let mut pixel_format = None;
    for format in device.formats(BufType::VIDEO_CAPTURE) {
        let format = format?;
        if format.pixelformat() == Pixelformat::JPEG || format.pixelformat() == Pixelformat::MJPG {
            pixel_format = Some(format.pixelformat());
            break;
        }
    }

    let Some(pixel_format) = pixel_format else {
        bail!("no supported pixel format found");
    };

    let mut formats = Vec::new();
    match device.frame_sizes(pixel_format)? {
        FrameSizes::Discrete(sizes) => {
            for size in sizes {
                let intervals =
                    match device.frame_intervals(pixel_format, size.width(), size.height())? {
                        FrameIntervals::Discrete(intervals) => intervals,
                        FrameIntervals::Stepwise(_) | FrameIntervals::Continuous(_) => {
                            bail!("stepwise or continuous frame rates are not supported")
                        }
                    };
                for rate in intervals {
                    formats.push(FrameFormat {
                        resolution: Resolution::new(size.width(), size.height()),
                        frame_interval: *rate.fract(),
                    });
                }
            }
        }
        FrameSizes::Stepwise(_) | FrameSizes::Continuous(_) => {
            bail!("stepwise or continuous resolutions are not supported");
        }
    }

    loop {
        if let Some(fmt) = negotiate_format_step(&formats, prefs) {
            return Ok((
                PixFormat::new(
                    fmt.resolution.width(),
                    fmt.resolution.height(),
                    pixel_format,
                ),
                fmt.frame_interval,
            ));
        }

        log::debug!("failed to negotiate format with prefs {:?}", prefs);
        match prefs.pref {
            ParamPreference::Resolution => {
                if prefs.resolution.take().is_none() && prefs.fps.take().is_none() {
                    break;
                }
            }
            ParamPreference::Framerate => {
                if prefs.fps.take().is_none() && prefs.resolution.take().is_none() {
                    break;
                }
            }
        }
        log::debug!("retrying with new prefs {:?}", prefs);
    }

    bail!("failed to negotiate a webcam format")
}

fn negotiate_format_step(formats: &[FrameFormat], prefs: FramePrefs) -> Option<FrameFormat> {
    let eligible = formats
        .iter()
        .filter(|fmt| {
            prefs.resolution.map_or(true, |res| {
                fmt.resolution.width() >= res.width() && fmt.resolution.height() >= res.height()
            }) && prefs.fps.map_or(true, |fps| {
                (1.0 / fmt.frame_interval.as_f32()).round() >= fps as f32
            })
        })
        .copied();
    let mut formats = eligible.collect::<Vec<_>>();
    match prefs.pref {
        ParamPreference::Resolution => {
            formats.sort_by_key(|fmt| (fmt.resolution.num_pixels(), Reverse(fmt.frame_interval)))
        }
        ParamPreference::Framerate => {
            formats.sort_by_key(|fmt| (Reverse(fmt.frame_interval), fmt.resolution.num_pixels()))
        }
    }
    formats.last().copied()
}

/// A webcam yielding a stream of [`Image`]s.
pub struct Webcam {
    stream: ReadStream,
    width: u32,
    height: u32,
    t_dequeue: Timer,
    t_decode: Timer,
}

const ENV_VAR_WEBCAM_NAME: &str = "ZARU_WEBCAM_NAME";

impl Webcam {
    /// Opens the first supported webcam found.
    ///
    /// This function can block for a significant amount of time while the webcam initializes (on
    /// the order of hundreds of milliseconds).
    pub fn open(options: WebcamOptions) -> anyhow::Result<Self> {
        if let Ok(name) = env::var(ENV_VAR_WEBCAM_NAME) {
            log::debug!(
                "webcam override: `{}` is set to '{}'",
                ENV_VAR_WEBCAM_NAME,
                name,
            );
        }
        for res in linuxvideo::list()? {
            match res {
                Ok(dev) => match Self::open_impl(dev, &options) {
                    Ok(Some(webcam)) => return Ok(webcam),
                    Ok(None) => {}
                    Err(e) => {
                        log::debug!("{}", e);
                    }
                },
                Err(e) => {
                    log::warn!("{}", e);
                }
            }
        }

        bail!("no supported webcam device found")
    }

    fn open_impl(dev: Device, options: &WebcamOptions) -> anyhow::Result<Option<Self>> {
        let caps = dev.capabilities()?;
        let cam_name_from_env = env::var(ENV_VAR_WEBCAM_NAME).ok();
        if let Some(name) = &options.name.as_deref().or(cam_name_from_env.as_deref()) {
            if caps.card() != *name {
                return Ok(None);
            }
        }

        let cap_flags = caps.device_capabilities();
        let path = dev.path()?;
        log::debug!(
            "device {} ({}) capabilities: {:?}",
            caps.card(),
            path.display(),
            cap_flags,
        );

        if !cap_flags.contains(CapabilityFlags::VIDEO_CAPTURE) {
            return Ok(None);
        }

        let (pixfmt, fract) = negotiate_format(&dev, options.frame)?;

        let capture = dev.video_capture(pixfmt)?;

        let format = capture.format();
        let width = format.width();
        let height = format.height();

        let actual = capture.set_frame_interval(fract)?;

        log::info!(
            "opened {} ({}), {}x{} @ {:.1}Hz",
            caps.card(),
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
    pub fn read(&mut self) -> anyhow::Result<Image> {
        let dequeue_guard = self.t_dequeue.start();
        self.stream
            .dequeue(|buf| {
                drop(dequeue_guard);
                let image = match self.t_decode.time(|| Image::decode_jpeg(&buf)) {
                    Ok(image) => image,
                    Err(e) => {
                        // As sad as it is, but even high-quality webcams produce occasional corrupted
                        // MJPG frames, presumably due to USB data corruption (the alternative would be
                        // a silicon bug in the webcam's MJPG encoder, which is a possibility I chose to
                        // ignore for the sake of my own sanity).
                        log::error!("webcam decode error: {}", e);

                        //std::fs::write("error.jpg", &*buf).ok();

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
            .map_err(Into::into)
    }

    /// Returns a borrowing iterator over the frames produced by this webcam.
    pub fn iter_mut(&mut self) -> IterMut<'_> {
        IterMut { webcam: self }
    }

    /// Returns profiling timers for webcam access and decoding.
    pub fn timers(&self) -> impl Iterator<Item = &Timer> + '_ {
        [&self.t_dequeue, &self.t_decode].into_iter()
    }
}

impl IntoIterator for Webcam {
    type Item = anyhow::Result<Image>;
    type IntoIter = IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter { webcam: self }
    }
}

impl<'a> IntoIterator for &'a mut Webcam {
    type Item = anyhow::Result<Image>;
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
    type Item = anyhow::Result<Image>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.webcam.read())
    }
}

impl Iterator for IterMut<'_> {
    type Item = anyhow::Result<Image>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.webcam.read())
    }
}
