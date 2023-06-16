//! Estimators for [68 facial landmark points] popularized by the now defunct [Multi-PIE dataset].
//!
//! These networks generally only output 2-dimensional landmarks instead of 3-dimensional ones.
//!
//! [68 facial landmark points]: https://ibug.doc.ic.ac.uk/media/uploads/images/annotpics/figure_68_markup.jpg
//! [Multi-PIE dataset]: http://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html

use include_blob::include_blob;
use once_cell::sync::Lazy;

use crate::{
    iter::zip_exact,
    landmark::{Estimate, Landmarks, Network},
    nn::{Cnn, CnnInputShape, ColorMapper, NeuralNetwork, Outputs},
    slice::SliceExt,
};

const NUM_LANDMARKS: usize = 68;

pub struct LandmarkResult {
    landmarks: Landmarks,
}

impl LandmarkResult {
    pub fn landmarks(&self) -> &Landmarks {
        &self.landmarks
    }
}

impl Default for LandmarkResult {
    fn default() -> Self {
        Self {
            landmarks: Landmarks::new(NUM_LANDMARKS),
        }
    }
}

impl Estimate for LandmarkResult {
    fn landmarks_mut(&mut self) -> &mut Landmarks {
        &mut self.landmarks
    }
}

/// The network from [`Peppa-Facial-Landmark-PyTorch`].
///
/// This network is fairly fast to infer, but generates inaccurate results.
///
/// [`Peppa-Facial-Landmark-PyTorch`]: https://github.com/ainrichman/Peppa-Facial-Landmark-PyTorch
pub struct PeppaFacialLandmark;

impl Network for PeppaFacialLandmark {
    type Output = LandmarkResult;

    fn cnn(&self) -> &Cnn {
        static MODEL: Lazy<Cnn> = Lazy::new(|| {
            let model_data = include_blob!("../../3rdparty/onnx/slim_160_latest.onnx");
            Cnn::new(
                NeuralNetwork::from_onnx(model_data).load().unwrap(),
                CnnInputShape::NCHW,
                ColorMapper::linear(-1.0..=1.0),
            )
            .unwrap()
        });

        &MODEL
    }

    fn extract(&self, outputs: &Outputs, estimate: &mut Self::Output) {
        let res = self.cnn().input_resolution();
        for (&[x, y], out) in zip_exact(
            outputs[0].index([0]).as_slice()[..NUM_LANDMARKS * 2].array_chunks_exact::<2>(),
            estimate.landmarks.positions_mut(),
        ) {
            out[0] = x * res.width() as f32;
            out[1] = y * res.height() as f32;
        }
    }
}

/// The landmark network used by [FaceONNX].
///
/// Slower than [`PeppaFacialLandmark`], taking about twice the time to infer, but more accurate.
/// Sadly this network still doesn't compute accurate landmarks when making anything resembling a
/// grimace.
///
/// [FaceONNX]: https://github.com/FaceONNX/FaceONNX
pub struct FaceOnnx;

impl Network for FaceOnnx {
    type Output = LandmarkResult;

    fn cnn(&self) -> &Cnn {
        static MODEL: Lazy<Cnn> = Lazy::new(|| {
            let model_data = include_blob!("../../3rdparty/onnx/landmarks_68_pfld.onnx");
            Cnn::new(
                NeuralNetwork::from_onnx(model_data).load().unwrap(),
                CnnInputShape::NCHW,
                ColorMapper::linear(0.0..=1.0),
            )
            .unwrap()
        });

        &MODEL
    }

    fn extract(&self, outputs: &Outputs, estimate: &mut Self::Output) {
        let res = self.cnn().input_resolution();
        for (&[x, y], out) in zip_exact(
            outputs[0].index([0]).as_slice()[..NUM_LANDMARKS * 2].array_chunks_exact::<2>(),
            estimate.landmarks.positions_mut(),
        ) {
            out[0] = x * res.width() as f32;
            out[1] = y * res.height() as f32;
        }
    }
}

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../3rdparty/3d/multipie68.rs"
));

/// Returns an iterator over the reference landmark positions.
pub fn reference_positions() -> impl Iterator<Item = [f32; 3]> {
    REFERENCE_POSITIONS.iter().copied()
}
