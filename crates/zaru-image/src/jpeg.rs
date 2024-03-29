use std::{
    cell::RefCell,
    env::{self, VarError},
    num::NonZeroU32,
    panic::catch_unwind,
    process,
    sync::OnceLock,
};

use anyhow::bail;
use fev::{
    display::Display,
    jpeg::{JpegDecodeSession, JpegInfo},
};

/// Turned off by default, because VA-API does not work at all on Mesa/AMD, and on Intel the
/// performance is too unpredictable (10-50ms for a 4K image) to be even remotely usable.
const DEFAULT_VAAPI: VaApi = VaApi::Off;

const DEFAULT_BACKEND: JpegBackend = JpegBackend::ZuneJpeg;

#[derive(PartialEq, Eq)]
enum VaApi {
    On,
    Off,
    Force,
}

impl VaApi {
    fn get() -> &'static VaApi {
        static USE_VAAPI: OnceLock<VaApi> = OnceLock::new();
        USE_VAAPI.get_or_init(|| match env::var("ZARU_JPEG_VAAPI") {
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
        })
    }
}

/// Because computers, we support several different JPEG decoding backends.
#[derive(Debug)]
enum JpegBackend {
    /// Uses the `jpeg-decoder` crate, a robust but slow pure-Rust JPEG decoder.
    JpegDecoder,
    /// Uses the `mozjpeg` crate, a wrapper around Mozilla's libjpeg fork. Somewhat buggy, fails to
    /// decode images from my webcam frequently.
    MozJpeg,
    /// Uses the `turbojpeg` crate, a wrapper around *libjpeg-turbo*.
    LibjpegTurbo,
    /// Uses the `zune-jpeg` crate, a pure-Rust JPEG decoder somewhat faster than `jpeg-decoder`.
    /// Tends to be slower than `mozjpeg` on some images.
    ZuneJpeg,
    /// Uses a specific patched commit of the `zune-jpeg` crate, which can perform better than
    /// mozjpeg, but errors on some valid images and incorrectly decodes some images. Only useful in
    /// specific circumstances.
    FastButWrong,
}

impl JpegBackend {
    fn get() -> &'static JpegBackend {
        static JPEG_BACKEND: OnceLock<JpegBackend> = OnceLock::new();
        JPEG_BACKEND.get_or_init(|| {
            let backend = match env::var("ZARU_JPEG_BACKEND") {
                Ok(v) if v == "fast-but-wrong" => JpegBackend::FastButWrong,
                Ok(v) if v == "mozjpeg" => JpegBackend::MozJpeg,
                Ok(v) if v == "libjpeg-turbo" => JpegBackend::LibjpegTurbo,
                Ok(v) if v == "zune-jpeg" => JpegBackend::ZuneJpeg,
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
        })
    }
}

// FIXME: make those private again once everything is using zaru_image::Image
pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

pub fn decode_jpeg(data: &[u8]) -> anyhow::Result<DecodedImage> {
    match decode_jpeg_vaapi(data) {
        Ok(image) => return Ok(image),
        Err(e) => {
            log::trace!("VA-API decode failed: {e}; falling back to software decoding");
        }
    }

    let buf = match JpegBackend::get() {
        JpegBackend::JpegDecoder => {
            let mut decoder = jpeg_decoder::Decoder::new(data);
            let data = decoder.decode()?;
            let info = decoder.info().unwrap();

            if info.pixel_format != jpeg_decoder::PixelFormat::RGB24 {
                anyhow::bail!("unsupported JPEG pixel format {:?}", info.pixel_format);
            }

            let mut output = vec![0; info.width as usize * info.height as usize * 4];
            for (dest, src) in output.chunks_exact_mut(4).zip(data.chunks(3)) {
                dest[0] = src[0];
                dest[1] = src[1];
                dest[2] = src[2];
            }

            DecodedImage {
                width: info.width.into(),
                height: info.height.into(),
                data: output,
            }
        }
        JpegBackend::MozJpeg => {
            // mozjpeg crate unfortunately reports errors only via unwinding
            let (buf, width, height) = catch_unwind(|| -> anyhow::Result<_> {
                let mut decompress = mozjpeg::Decompress::new_mem(data)?;

                // Tune settings for decode performance.
                decompress.do_fancy_upsampling(false);
                decompress.dct_method(mozjpeg::DctMethod::IntegerFast);

                let mut decompress = decompress.rgba()?;
                let buf = decompress
                    .read_scanlines_flat()
                    .ok_or_else(|| anyhow::anyhow!("failed to decode image"))?;
                Ok((buf, decompress.width(), decompress.height()))
            })
            .map_err(|payload| match payload.downcast::<String>() {
                Ok(string) => anyhow::Error::msg(string),
                Err(_) => anyhow::anyhow!("<unknown panic message>"),
            })??;

            DecodedImage {
                width: width.try_into().unwrap(),
                height: height.try_into().unwrap(),
                data: buf,
            }
        }
        JpegBackend::LibjpegTurbo => {
            let mut decomp = turbojpeg::Decompressor::new()?;
            let header = decomp.read_header(data)?;

            let mut image = turbojpeg::Image {
                pixels: vec![0; header.width * header.height * 4],
                width: header.width,
                pitch: header.width * 4,
                height: header.height,
                format: turbojpeg::PixelFormat::RGBA,
            };
            decomp.decompress(data, image.as_deref_mut())?;

            DecodedImage {
                width: header.width.try_into().unwrap(),
                height: header.height.try_into().unwrap(),
                data: image.pixels,
            }
        }
        JpegBackend::ZuneJpeg => {
            use zune_jpeg::zune_core::colorspace::ColorSpace;
            use zune_jpeg::zune_core::options::DecoderOptions;

            let mut decomp = zune_jpeg::JpegDecoder::new_with_options(
                DecoderOptions::new_fast().jpeg_set_out_colorspace(ColorSpace::RGBA),
                data,
            );
            decomp.decode_headers()?;
            let colorspace = decomp.get_output_colorspace().unwrap();
            if colorspace != ColorSpace::RGBA {
                bail!("unsupported colorspace {colorspace:?} (expected RGBA)");
            }

            let mut buf = vec![0; decomp.output_buffer_size().unwrap()];
            decomp.decode_into(&mut buf)?;
            let (width, height) = decomp.dimensions().unwrap();
            DecodedImage {
                width: width.into(),
                height: height.into(),
                data: buf,
            }
        }
        JpegBackend::FastButWrong => {
            let mut decomp = fast_but_wrong::Decoder::new_with_options(
                fast_but_wrong::ZuneJpegOptions::new()
                    .set_num_threads(NonZeroU32::new(1).unwrap())
                    .set_out_colorspace(fast_but_wrong::ColorSpace::RGBA),
            );
            let buf = decomp.decode_buffer(data)?;
            DecodedImage {
                width: decomp.width().into(),
                height: decomp.height().into(),
                data: buf,
            }
        }
    };

    Ok(buf)
}

#[allow(unreachable_code, unused_variables)]
fn decode_jpeg_vaapi(jpeg: &[u8]) -> anyhow::Result<DecodedImage> {
    let display = match VaApi::get() {
        VaApi::Off => bail!("VA-API use disabled by env var"),
        VaApi::Force | VaApi::On => {
            static DISPLAY: OnceLock<Option<Display>> = OnceLock::new();

            let display = DISPLAY.get_or_init(|| {
                /*let display = match unsafe { Display::new_unmanaged(gui::Display::get()) } {
                    Ok(display) => display,
                    Err(e) => {
                        log::warn!("failed to open VA-API display: {e}");
                        return None;
                    }
                };*/

                // TODO: VA-API decoding needs display connection
                let display: Display = todo!("VA-API");

                match display.query_vendor_string() {
                    Ok(vendor) => {
                        if !vendor.contains("Intel") && *VaApi::get() == VaApi::On {
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

            match display {
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
    let image = SESSION.with(|cache| -> anyhow::Result<_> {
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

        let surface = sess.decode_and_convert(jpeg)?;
        let mapping = surface.map_sync()?;
        Ok(DecodedImage {
            width: info.width().into(),
            height: info.height().into(),
            data: mapping[..info.width() as usize * info.height() as usize * 4].into(),
        })
    })?;

    Ok(image)
}
