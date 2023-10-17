use std::path::Path;

use crate::{jpeg, Image, Resolution};

/// Enumeration of image formats supported by this library.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum ImageFormat {
    /// JFIF JPEG or Motion JPEG.
    Jpeg,
    /// Portable Network Graphics.
    Png,
}

impl ImageFormat {
    pub fn from_extension(path: &Path) -> anyhow::Result<Self> {
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("jpg" | "jpeg") => Ok(Self::Jpeg),
            Some("png") => Ok(Self::Png),
            _ => anyhow::bail!(
                "invalid image path '{}' (must have one of the supported extensions)",
                path.display()
            ),
        }
    }
}

/// Image decoding and loading.
impl Image {
    /// Loads an image from the filesystem.
    ///
    /// The path must have a supported file extension (`jpeg`, `jpg` or `png`).
    pub fn load<A: AsRef<Path>>(path: A) -> anyhow::Result<Self> {
        Self::load_impl(path.as_ref())
    }

    fn load_impl(path: &Path) -> anyhow::Result<Self> {
        // TODO: add a default file size limit; this loads the whole file into memory!
        match ImageFormat::from_extension(path)? {
            ImageFormat::Jpeg => {
                let data = std::fs::read(path)?;
                Self::decode_jpeg(&data)
            }
            ImageFormat::Png => {
                let data = std::fs::read(path)?;
                let mut decoder = png::Decoder::new(&*data);
                decoder.set_transformations(
                    png::Transformations::STRIP_16
                        | png::Transformations::EXPAND
                        | png::Transformations::ALPHA,
                );
                let mut reader = decoder.read_info()?;
                let info = reader.info();
                let (width, height) = (info.width, info.height);
                let mut buf = vec![0; width as usize * height as usize * 4];
                reader.next_frame(&mut buf)?;
                Ok(Self::from_rgba8(Resolution::new(width, height), &buf))
            }
        }
    }

    /// Decodes a JFIF JPEG or Motion JPEG from a byte slice.
    pub fn decode_jpeg(data: &[u8]) -> anyhow::Result<Self> {
        let jpeg::DecodedImage {
            width,
            height,
            data,
        } = jpeg::decode_jpeg(data)?;

        Ok(Self::from_rgba8(Resolution::new(width, height), &data))
    }

    // TODO: make this a `decode` method taking an `ImageFormat`
}
