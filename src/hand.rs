//! Detection and pose estimation of human hands.

pub mod detection;

/// Names for the hand pose landmarks.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LandmarkIdx {
    Wrist,
    ThumbCmc,
    ThumbMcp,
    ThumbIp,
    ThumbTip,
    IndexFingerMcp,
    IndexFingerPip,
    IndexFingerDip,
    IndexFingerTip,
    MiddleFingerMcp,
    MiddleFingerPip,
    MiddleFingerDip,
    MiddleFingerTip,
    RingFingerMcp,
    RingFingerPip,
    RingFingerDip,
    RingFingerTip,
    PinkyMcp,
    PinkyPip,
    PinkyDip,
    PinkyTip,
}
