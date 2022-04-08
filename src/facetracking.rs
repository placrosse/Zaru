//! Mizaru's public face tracking data format.
//!
//! The face tracking data is transmitted as a unidirectional stream of [`TrackerMessage`]s over a
//! TCP port. The messages are encoded as newline-delimited JSON objects (["JSON Lines"]).
//!
//! Tracking service discovery happens via DNS-SD over mDNS. (TODO: to be specified later)
//!
//! # Goals of the format
//!
//! - Simplicity and Debuggability: Achieved by supporting a JSON transport, which is trivial to
//!   consume from almost all programming languages, and has tooling such as `jq` to inspect the
//!   data. Also achieved by limiting the format to face tracking only (no full-body tracking is or
//!   will be supported).
//! - High Fidelity: The format should include enough information to accurately reproduce any facial
//!   expression, and also convey accurate eye and eyebrow movement. This is achieved by encoding
//!   keypoint positions and facial feature dimensions rather than more granular information. For
//!   example, we encode the eyelid position rather than a status flag that distinguishes between
//!   "eye open" and "eye closed".
//! - Multi-Seat Ready: The format should support tracking the faces of multiple people at once, and
//!   should allow trackers to tell subjects apart via face recognition, while still permitting
//!   implementations that only support a single face.
//! - Technology-Independence: The format makes no assumptions about how the face tracking
//!   information is derived, processed, or consumed. Trackers may use regular webcams, stereoscopic
//!   3D cameras, infrared, laser technology, or anything else (however, note that the assumption is
//!   made that the tracker views the scene from a single position, so whole-room VR tracking
//!   systems are out). Tracking data can be derived using Neural Networks, classic computer vision
//!   algorithms, a mix of the two, or something else entirely. Data doesn't have to be sourced from
//!   the real world either, replaying recorded tracking data also works. The use of the
//!   JSON-object-per-line convention allows using essentially any technology to produce or consume
//!   this data. The format does not assume that any specific operating system or class of operating
//!   systems is used to produce or consume it.
//!
//! Disclaimer: I have no idea how face tracking is actually done in practice, and every existing
//! format appears to be documented exlusively in Japanese (or not at all), so this is very much
//! guesswork.
//!
//! # Coordinates
//!
//! The tracking format makes use of both 2D and 3D coordinates to describe the tracked subject.
//!
//! The 2D coordinate system is as follows: The origin `(0,0)` lies in the center of the object the
//! coordinates are relative to, X points to the right, Y points up. Positions, vectors, and sizes
//! using the [`Vec2`] type use this coordinate system.
//!
//! The 3D coordinate system works the same way, and Z points *into* the scene. Position, vectors,
//! and dimensions using the [`Vec3`] type use this coordinate system.
//!
//! ["JSON Lines"]: https://jsonlines.org/

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct TrackerMessage {
    /// Timestamp of this message, in units of microseconds.
    ///
    /// No time reference is defined, so these timestamps don't relate to real-world time. Instead,
    /// they only indicate the passage of time between two or more subsequent messages.
    ///
    /// TODO: milliseconds might be enough
    pub timestamp: u64,

    /// The list of subjects / people that are present in this frame.
    ///
    /// If the tracker only supports tracking a single person at a time, this list will only have a
    /// single entry. If no faces have been detected, the list will be empty.
    pub subjects: Vec<Subject>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Subject {
    /// ID for identifying a person across subsequent messages or globally.
    ///
    /// The ID must be unique withing a message: no two [`Subject`]s within the same
    /// [`TrackerMessage`] may have the same ID.
    ///
    /// If the tracker possesses face recognition abilities, this ID can be chosen to be unique
    /// per-person, even when the person re-enters the tracking range while other people are being
    /// tracked.
    ///
    /// If the tracker doesn't have face recognition abilities, but does support tracking multiple
    /// faces at once, the ID must still be unique across frames in which the face is present, at
    /// least to the best of the tracker's ability.
    ///
    /// TODO: this raises some interesting questions about what's supposed to happen if one subject
    /// leaves the tracking area and reenters it – what are consumers supposed to do if the ID they
    /// were tracking vanishes, but another ID remains? also, if the person re-enters the scene, do
    /// they get a new, higher ID, or do they always get the lowest free ID?
    pub id: u32,

    /// Position of the center of the face, in 2D.
    ///
    /// This position is in global tracker coordinates – the origin is the center of the tracker's
    /// field of vision. The field of vision has a resolution-independent width of 1.0, while the
    /// height is determined by the tracker's aspect ratio or equivalent concept.
    pub position: Vec2,

    /// Approximate size of the face.
    pub size: Vec2,

    /// The inferred rotation of the subject's head in 3D space.
    ///
    /// The reference orientation is the subject facing the camera head-on.
    pub head_rotation: Quaternion,

    pub features: Features,
}

/// Facial features.
///
/// These are all described relative to the feature's center, and assuming that the face faces the
/// tracker's input hardware head-on (the head's rotation does not affect the feature coordinates).
#[derive(Debug, Serialize, Deserialize)]
pub struct Features {
    // TODO
}

/// A quaternion of the form `q = r * x*i * y*j * z*k`.
///
/// Note that these are not necessarily normalized to have unit norm, so they cannot be used as
/// rotations as-is. Tracker implementations are encouraged to multiply every quaternion they
/// produce by a distinct random value to keep consumers on their toes :)
#[derive(Debug, Serialize, Deserialize)]
pub struct Quaternion {
    pub r: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// A position or vector in 2D space.
#[derive(Debug, Serialize, Deserialize)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

/// A position or vector in 3D space.
#[derive(Debug, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

// TODO: tongue tracking data, you perverts
