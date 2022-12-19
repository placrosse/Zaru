//! Estimators for [68 facial landmark points] popularized by the now defunct [Multi-PIE dataset].
//!
//! These networks generally only output 2-dimensional landmarks instead of 3-dimensional ones.
//!
//! [68 facial landmark points]: https://ibug.doc.ic.ac.uk/media/uploads/images/annotpics/figure_68_markup.jpg
//! [Multi-PIE dataset]: http://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html

use once_cell::sync::Lazy;
use zaru_utils::iter::zip_exact;

use crate::{
    landmark::{Estimation, Landmarks, Network},
    nn::{create_linear_color_mapper, Cnn, CnnInputShape, NeuralNetwork, Outputs},
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

impl Estimation for LandmarkResult {
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
        const MODEL_DATA: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../3rdparty/onnx/slim_160_latest.onnx"
        ));

        static MODEL: Lazy<Cnn> = Lazy::new(|| {
            Cnn::new(
                NeuralNetwork::from_onnx(MODEL_DATA)
                    .unwrap()
                    .load()
                    .unwrap(),
                CnnInputShape::NCHW,
                create_linear_color_mapper(-1.0..=1.0),
            )
            .unwrap()
        });

        &MODEL
    }

    fn extract(&self, outputs: &Outputs, estimation: &mut Self::Output) {
        let res = self.cnn().input_resolution();
        for (&[x, y], out) in zip_exact(
            outputs[0].index([0]).as_slice()[..NUM_LANDMARKS * 2].array_chunks_exact::<2>(),
            estimation.landmarks.positions_mut(),
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
        const MODEL_DATA: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../3rdparty/onnx/landmarks_68_pfld.onnx"
        ));

        static MODEL: Lazy<Cnn> = Lazy::new(|| {
            Cnn::new(
                NeuralNetwork::from_onnx(MODEL_DATA)
                    .unwrap()
                    .load()
                    .unwrap(),
                CnnInputShape::NCHW,
                create_linear_color_mapper(0.0..=1.0),
            )
            .unwrap()
        });

        &MODEL
    }

    fn extract(&self, outputs: &Outputs, estimation: &mut Self::Output) {
        let res = self.cnn().input_resolution();
        for (&[x, y], out) in zip_exact(
            outputs[0].index([0]).as_slice()[..NUM_LANDMARKS * 2].array_chunks_exact::<2>(),
            estimation.landmarks.positions_mut(),
        ) {
            out[0] = x * res.width() as f32;
            out[1] = y * res.height() as f32;
        }
    }
}
