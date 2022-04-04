//! Animated images.

use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
    slice,
    time::Duration,
};

use image::{codecs::gif::GifDecoder, AnimationDecoder, Frame, SubImage};

use crate::image::ImageView;

/// A timed sequence of images.
pub struct Animation {
    frames: Vec<Frame>,
}

impl Animation {
    /// Loads a gif animation from a filesystem path.
    ///
    /// The path must have a `.gif` extension.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, crate::Error> {
        let path = path.as_ref();
        match path.extension() {
            Some(ext) if ext == "gif" => {}
            _ => return Err(format!("animation path must have `.gif` extension").into()),
        }

        Self::from_gif_reader(BufReader::new(File::open(path)?))
    }

    /// Loads a gif animation from an in-memory byte slice.
    pub fn from_gif_data(data: &[u8]) -> Result<Self, crate::Error> {
        Self::from_gif_reader(data)
    }

    /// Loads a gif animation from a [`BufRead`] implementor.
    pub fn from_gif_reader<R: BufRead>(reader: R) -> Result<Self, crate::Error> {
        let frames = GifDecoder::new(reader)?.into_frames().collect_frames()?;

        Ok(Self { frames })
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
    frames: slice::Iter<'a, Frame>,
}

impl<'a> Iterator for FrameIter<'a> {
    type Item = AnimationFrame<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.frames.next().map(|frame| AnimationFrame {
            image: ImageView {
                sub_image: SubImage::new(
                    frame.buffer(),
                    0,
                    0,
                    frame.buffer().width(),
                    frame.buffer().height(),
                ),
            },
            duration: frame.delay().into(),
        })
    }
}

/// A frame of an animation, consisting of image data and a duration.
pub struct AnimationFrame<'a> {
    // NB: only exposes `ImageView` here so that all frames could be stored in a texture atlas.
    image: ImageView<'a>,
    duration: Duration,
}

impl<'a> AnimationFrame<'a> {
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
