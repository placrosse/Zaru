//! See no evil.

pub mod detector;
pub mod filter;
pub mod gui;
pub mod image;
pub mod landmark;
pub mod nn;
pub mod num;
pub mod resolution;
pub mod timer;
pub mod webcam;

pub type Error = Box<dyn std::error::Error + Sync + Send>;
