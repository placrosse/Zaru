//! GPU-based image manipulation for computer vision applications.
//!
//! # Overview
//!
//! ## Images and Views
//!
//! Zaru provides owned sRGBA images via the [`Image`] type. Each [`Image`] corresponds to a texture
//! stored on the GPU.
//!
//! Zaru also supports constructing *views* into [`Image`]s, lightweight objects that borrow the
//! original [`Image`] while allowing you to look at only part of the underlying pixel data.
//!
//! Two types of views are supported: immutable [`ImageView`]s, and mutable [`ImageViewMut`]s.
//! Additionally, the [`AsImageView`] and [`AsImageViewMut`] traits allow abstracting away the
//! concrete type of view or image in use. They all behave in accordance to Rust's mutability rules.
//! Almost all APIs in this crate are written in terms of the [`AsImageView`] and [`AsImageViewMut`]
//! traits, which allows using them on owned [`Image`]s as well as [`ImageView`]s.
//!
//! Views are not restricted to axis-aligned rectangles. They can be created with a
//! [`rect::RotatedRect`] to allow for arbitrary rotation and non-integer sizes. This flexibility is
//! useful in many computer vision tasks.
//!
//! ## Drawing
//!
//! A few primitive drawing operations are available in the [`draw`] module. They are primarily
//! meant for quickly visualizing objects and are not meant to be exhaustive.

pub mod draw;
pub mod num;
pub mod rect;

mod blend;
mod color;
mod decode;
mod gpu;
mod image;
mod jpeg;
mod resolution;
mod view;

pub use blend::*;
pub use color::Color;
pub use gpu::Gpu;
pub use image::*;
pub use resolution::{AspectRatio, Resolution};
pub use view::*;
