//! Animated images.

use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
    slice,
    time::Duration,
};

use anyhow::bail;
use image::{
    codecs::{gif::GifDecoder, png::PngDecoder},
    AnimationDecoder,
};
use zaru_image::{AsImageView, Image, ImageView, Resolution};

#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum AnimationFormat {
    Gif,
    Apng,
}

/// A timed sequence of images.
pub struct Animation {
    frames: Vec<(Image, Duration)>,
}

impl Animation {
    /// Loads an animation from the filesystem.
    ///
    /// The path must have a supported extension.
    pub fn from_path<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        Self::from_path_impl(path.as_ref())
    }

    fn from_path_impl(path: &Path) -> anyhow::Result<Self> {
        match path.extension() {
            Some(ext) => {
                let format = if ext == "gif" {
                    AnimationFormat::Gif
                } else if ext == "apng" || ext == "png" {
                    AnimationFormat::Apng
                } else {
                    bail!(
                        "file extension `{}` is not supported for animations",
                        ext.to_string_lossy()
                    );
                };

                Self::from_reader(BufReader::new(File::open(path)?), format)
            }
            _ => bail!("animation path must have a supported extension"),
        }
    }

    /// Loads a animation from an in-memory byte slice.
    pub fn from_data(data: &[u8], format: AnimationFormat) -> anyhow::Result<Self> {
        Self::from_reader(data, format)
    }

    /// Loads a animation from a [`BufRead`] implementor.
    pub fn from_reader<R: BufRead>(mut reader: R, format: AnimationFormat) -> anyhow::Result<Self> {
        Self::from_reader_impl(&mut reader, format)
    }

    fn from_reader_impl(reader: &mut dyn BufRead, format: AnimationFormat) -> anyhow::Result<Self> {
        let frames = match format {
            AnimationFormat::Gif => GifDecoder::new(reader)?.into_frames(),
            AnimationFormat::Apng => {
                let dec = PngDecoder::new(reader)?;
                if !dec.is_apng() {
                    bail!("attempted to load APNG animation from still image PNG");
                }
                dec.apng().into_frames()
            }
        };

        let frames = frames
            .map(|res| {
                let frame = res?;
                let dur = Duration::from(frame.delay());
                let image = Image::from_rgba8(
                    Resolution::new(frame.buffer().width(), frame.buffer().height()),
                    frame.buffer(),
                );
                Ok((image, dur))
            })
            .collect::<anyhow::Result<_>>()?;

        Ok(Self { frames })
    }

    /// Creates an animation from a sequence of frames.
    ///
    /// This copies the image data from each frame into the returned [`Animation`].
    pub fn from_frames<'a, I>(frames: I) -> Self
    where
        I: IntoIterator<Item = AnimationFrame<'a>>,
    {
        Self {
            frames: frames
                .into_iter()
                .map(|fr| (fr.image.to_image(), fr.duration))
                .collect(),
        }
    }

    /// Returns an iterator over the frames of this animation.
    ///
    /// Note that every frame is only yielded *once* (ie. the iterator does not loop, even if the
    /// animation does). Call [`Iterator::cycle`] to loop the animation.
    pub fn frames(&self) -> FrameIter<'_> {
        FrameIter {
            frames: self.frames.iter(),
        }
    }
}

/// An iterator over the [`AnimationFrame`]s that make up an [`Animation`].
#[derive(Clone)]
pub struct FrameIter<'a> {
    frames: slice::Iter<'a, (Image, Duration)>,
}

impl<'a> Iterator for FrameIter<'a> {
    type Item = AnimationFrame<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.frames
            .next()
            .map(|&(ref image, duration)| AnimationFrame {
                image: image.as_view(),
                duration,
            })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.frames.size_hint()
    }
}

impl<'a> ExactSizeIterator for FrameIter<'a> {}

/// A frame of an animation, consisting of image data and a duration.
pub struct AnimationFrame<'a> {
    // NB: only exposes `ImageView` here so that all frames could be stored in a texture atlas.
    image: ImageView<'a>,
    duration: Duration,
}

impl<'a> AnimationFrame<'a> {
    /// Creates a new [`AnimationFrame`] that will display `image` for `duration`.
    pub fn new(image: ImageView<'a>, duration: Duration) -> Self {
        Self { image, duration }
    }

    /// Returns an [`ImageView`] of the image data for this frame.
    pub fn image_view(&self) -> &ImageView<'a> {
        &self.image
    }

    /// Returns the [`Duration`] for which this frame should be displayed before proceeding to the
    /// next one.
    pub fn duration(&self) -> Duration {
        self.duration
    }
}
