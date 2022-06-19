//! Detection and pose estimation of human hands.
//!
//! TODO:
//!
//! - hand tracking performs poorly when hands are not upright, does the landmark network require us
//!   to rotate the image?

pub mod detection;
pub mod landmark;
pub mod tracking;
