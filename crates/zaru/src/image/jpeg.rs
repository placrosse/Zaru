use std::{
    cell::RefCell,
    env::{self, VarError},
    num::NonZeroU32,
    process,
};

use anyhow::bail;
use image::ImageBuffer;
use once_cell::sync::Lazy;
use v_ayylmao::{
    display::Display,
    jpeg::{JpegDecodeSession, JpegInfo},
};

use crate::gui;
use crate::image::Resolution;

use super::Image;

#[derive(PartialEq, Eq)]
enum VaApi {
    On,
    Off,
    Force,
}

/// Because computers, we support several different JPEG decoding backends.
#[derive(Debug)]
enum JpegBackend {
    /// Uses the `jpeg-decoder` crate, a robust but slow pure-Rust JPEG decoder.
    JpegDecoder,
    /// Uses the `mozjpeg` crate, a wrapper around Mozilla's libjpeg fork. Robust and fast-ish, but
    /// C.
    MozJpeg,
    /// Uses a specific patched commit of the `zune-jpeg` crate, which can perform better than
    /// mozjpeg, but errors on some valid images and incorrectly decodes some images. Only useful in
    /// specific circumstances.
    FastButWrong,
}

/// Turned off by default, because VA-API does not work at all on Mesa/AMD, and on Intel the
/// performance is too unpredictable (10-50ms for a 4K image) to be even remotely usable.
const DEFAULT_VAAPI: VaApi = VaApi::Off;

const DEFAULT_BACKEND: JpegBackend = JpegBackend::MozJpeg;

static USE_VAAPI: Lazy<VaApi> = Lazy::new(|| match env::var("ZARU_JPEG_VAAPI") {
    Ok(v) if v == "1" || v == "true" || v == "on" => VaApi::On,
    Ok(v) if v == "0" || v == "false" || v == "off" => VaApi::Off,
    Ok(v) if v == "force" => VaApi::Force,
    Ok(v) => {
        eprintln!("invalid value set for `ZARU_JPEG_VAAPI` variable: '{v}'; exiting");
        process::exit(1);
    }
    Err(VarError::NotPresent) => DEFAULT_VAAPI,
    Err(VarError::NotUnicode(s)) => {
        eprintln!(
            "invalid value set for `ZARU_JPEG_VAAPI` variable: {}; exiting",
            s.to_string_lossy()
        );
        process::exit(1);
    }
});

static JPEG_BACKEND: Lazy<JpegBackend> = Lazy::new(|| {
    let backend = match env::var("ZARU_JPEG_BACKEND") {
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
    };
    log::debug!("using JPEG decode backend: {:?}", backend);
    backend
});

pub(super) fn decode_jpeg(data: &[u8]) -> anyhow::Result<Image> {
    match decode_jpeg_vaapi(data) {
        Ok(image) => return Ok(image),
        Err(e) => {
            log::trace!("VA-API decode failed: {e}; falling back to software decoding");
        }
    }

    let buf = match *JPEG_BACKEND {
        JpegBackend::JpegDecoder => {
            image::load_from_memory_with_format(data, image::ImageFormat::Jpeg)?.to_rgba8()
        }
        JpegBackend::MozJpeg => {
            let mut decompress = mozjpeg::Decompress::new_mem(data)?;

            // Tune settings for decode performance.
            decompress.do_fancy_upsampling(false);
            decompress.dct_method(mozjpeg::DctMethod::IntegerFast);

            let mut decompress = decompress.rgba()?;
            let buf = decompress
                .read_scanlines_flat()
                .ok_or_else(|| anyhow::anyhow!("failed to decode image"))?;
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

fn decode_jpeg_vaapi(jpeg: &[u8]) -> anyhow::Result<Image> {
    let display = match *USE_VAAPI {
        VaApi::Off => bail!("VA-API use disabled by env var"),
        VaApi::Force | VaApi::On => {
            static DISPLAY: Lazy<Option<Display>> = Lazy::new(|| {
                let display = match unsafe { Display::new_unmanaged(gui::Display::get()) } {
                    Ok(display) => display,
                    Err(e) => {
                        log::warn!("failed to open VA-API display: {e}");
                        return None;
                    }
                };

                match display.query_vendor_string() {
                    Ok(vendor) => {
                        if !vendor.contains("Intel") && *USE_VAAPI == VaApi::On {
                            log::debug!("VA-API implementation vendor '{vendor}' is not supported; not using VA-API");
                            log::debug!("(set ZARU_JPEG_VAAPI=force to override; it probably won't work, but it might be funny)");
                            return None;
                        }

                        log::debug!("using VA-API for JPEG decoding (vendor string: {vendor})");
                        log::debug!("(set ZARU_JPEG_VAAPI=0 to disable)");
                        Some(display)
                    }
                    Err(e) => {
                        log::warn!("failed to query VA-API vendor: {e}");
                        None
                    }
                }
            });

            match &*DISPLAY {
                Some(display) => display,
                None => bail!("VA-API not initialized"),
            }
        }
    };

    // A per-thread VA-API object cache reuses the same set of objects for each image resolution.
    // VA-API session/surface creation can be expensive, and a thread typically only decodes images
    // from the same source, so this should improve efficiency.
    thread_local! {
        static SESSION: RefCell<Option<(JpegInfo, JpegDecodeSession)>> = const { RefCell::new(None) };
    }

    let info = JpegInfo::new(jpeg)?;
    let image = SESSION.with(|cache| -> anyhow::Result<Image> {
        let mut cache = cache.borrow_mut();
        let sess = match &mut *cache {
            Some((i, sess)) if i.height() == info.height() || i.width() == info.width() => sess,
            _ => {
                log::debug!(
                    "creating new JPEG decode session for {}x{} images",
                    info.width(),
                    info.height()
                );
                let session = JpegDecodeSession::new(display, info.width(), info.height())?;
                &mut cache.insert((info, session)).1
            }
        };

        let mapping = sess.decode(jpeg)?;
        Ok(Image::from_rgba8(
            Resolution::new(info.width().into(), info.height().into()),
            &mapping[..info.width() as usize * info.height() as usize * 4],
        ))
    })?;

    Ok(image)
}
