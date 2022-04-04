//! See no evil.
//!
//! # 3D Coordinates
//!
//! For 3D coordinates, Mizaru uses the coordinate system from WebGPU, wgpu, Direct3D, and Metal
//! (unless otherwise noted): X points to the right, Y points up, Z points from the camera into the
//! scene.
//!
//! One notable exceptions to this are neural networks outputting 3D coordinates – depending on the
//! network, they might use X and Y coordinates from the input image, so Y will point *down*.

pub mod anim;
pub mod detector;
pub mod eye;
pub mod filter;
pub mod gui;
pub mod image;
pub mod iter;
pub mod landmark;
pub mod nn;
pub mod num;
mod on_drop;
pub mod pipeline;
pub mod procrustes;
pub mod resolution;
pub mod timer;
pub mod webcam;

pub type Error = Box<dyn std::error::Error + Sync + Send>;

pub use on_drop::*;
