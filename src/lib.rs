//! See no evil.

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
