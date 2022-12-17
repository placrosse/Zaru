use std::{
    env::{self, VarError},
    num::NonZeroU32,
    process,
};

use image::ImageBuffer;
use once_cell::sync::Lazy;

use super::Image;

/// Because computers, we support several different JPEG decoding backends.
enum JpegBackend {
    /// Uses the `jpeg-decoder` crate, a robust but slow pure-Rust JPEG decoder.
    JpegDecoder,
    /// Uses the `mozjpeg` crate, a wrapper around Mozilla's libjpeg fork. Robust and fast, but C.
    MozJpeg,
    /// Uses a specific patched commit of the `zune-jpeg` crate, which can perform better than
    /// mozjpeg, but errors on some valid images and incorrectly decodes some images. Only useful in
    /// specific circumstances.
    FastButWrong,
}

const DEFAULT_BACKEND: JpegBackend = JpegBackend::MozJpeg;

static JPEG_BACKEND: Lazy<JpegBackend> = Lazy::new(|| match env::var("ZARU_JPEG_BACKEND") {
    Ok(v) if v == "fast-but-wrong" => JpegBackend::FastButWrong,
    Ok(v) if v == "mozjpeg" => JpegBackend::MozJpeg,
    Ok(v) if v == "jpeg-decoder" => JpegBackend::JpegDecoder,
    Ok(v) => {
        eprintln!("invalid value set for `ZARU_JPEG_BACKEND` variable: '{v}'; exiting");
        process::exit(1);
    }
    Err(VarError::NotPresent) => DEFAULT_BACKEND,
    Err(VarError::NotUnicode(s)) => {
        eprintln!(
            "invalid value set for `ZARU_JPEG_BACKEND` variable: {}; exiting",
            s.to_string_lossy()
        );
        process::exit(1);
    }
});

pub(super) fn decode_jpeg(data: &[u8]) -> Result<Image, crate::Error> {
    let buf = match *JPEG_BACKEND {
        JpegBackend::JpegDecoder => {
            image::load_from_memory_with_format(data, image::ImageFormat::Jpeg)?.to_rgba8()
        }
        JpegBackend::MozJpeg => {
            let mut decompress = mozjpeg::Decompress::new_mem(data)?.rgba()?;
            let buf = decompress
                .read_scanlines_flat()
                .ok_or_else(|| "failed to decode image")?;
            ImageBuffer::from_raw(
                decompress.width().try_into().unwrap(),
                decompress.height().try_into().unwrap(),
                buf,
            )
            .expect("failed to create ImageBuffer")
        }
        JpegBackend::FastButWrong => {
            let mut decomp = zune_jpeg::Decoder::new_with_options(
                zune_jpeg::ZuneJpegOptions::new()
                    .set_num_threads(NonZeroU32::new(1).unwrap())
                    .set_out_colorspace(zune_jpeg::ColorSpace::RGBA),
            );
            let buf = decomp.decode_buffer(data)?;
            let width = u32::from(decomp.width());
            let height = u32::from(decomp.height());
            ImageBuffer::from_raw(width, height, buf).expect("failed to create ImageBuffer")
        }
    };

    Ok(Image { buf })
}
