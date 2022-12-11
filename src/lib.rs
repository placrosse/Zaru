//! Zaru Machine Perception library.
//!
//! # 3D Coordinates
//!
//! For 3D coordinates, Zaru uses the coordinate system from WebGPU, wgpu, Direct3D, and Metal
//! (unless otherwise noted): X points to the right, Y points up, Z points from the camera into the
//! scene.
//!
//! One notable exceptions to this are neural networks outputting 3D coordinates – depending on the
//! network, they might use X and Y coordinates from the input image, so Y will point *down*.

#![allow(illegal_floating_point_literal_pattern)] // let me have fun

use log::LevelFilter;

pub mod anim;
pub mod body;
pub mod detection;
pub mod drop;
pub mod face;
pub mod filter;
pub mod gui;
pub mod hand;
pub mod image;
pub mod iter;
pub mod landmark;
pub mod nn;
pub mod num;
pub mod procrustes;
pub mod resolution;
mod slice;
pub mod timer;
pub mod webcam;

#[cfg(test)]
mod test;

pub type Error = Box<dyn std::error::Error + Sync + Send>;
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// macro-use only, not part of public API.
#[doc(hidden)]
pub fn init_logger(calling_crate: &'static str) {
    let log_level = LevelFilter::Debug;
    env_logger::Builder::new()
        .filter(Some(calling_crate), log_level)
        .filter(Some(env!("CARGO_PKG_NAME")), log_level)
        .filter(Some("wgpu"), LevelFilter::Warn)
        .parse_default_env()
        .try_init()
        .ok();
}

/// Initializes logging to *stderr*.
///
/// If `cfg!(debug_assertions)` is enabled, the calling crate and Zaru will log at *trace* level.
/// Otherwise, they will log at *debug* level.
///
/// `wgpu` will always log at *warn* level.
///
/// If a global logger is already registered, this macro will do nothing.
#[macro_export]
macro_rules! init_logger {
    () => {
        $crate::init_logger(env!("CARGO_CRATE_NAME"))
    };
}
