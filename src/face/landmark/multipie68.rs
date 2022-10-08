//! Estimators for [68 facial landmark points] popularized by the now defunct [Multi-PIE dataset].
//!
//! These networks generally only output 2-dimensional landmarks instead of 3-dimensional ones.
//!
//! [68 facial landmark points]: https://ibug.doc.ic.ac.uk/media/uploads/images/annotpics/figure_68_markup.jpg
//! [Multi-PIE dataset]: http://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html

use once_cell::sync::Lazy;

use crate::{
    image::{AsImageView, ImageView},
    iter::zip_exact,
    landmark::Landmarks,
    nn::{create_linear_color_mapper, unadjust_aspect_ratio, Cnn, CnnInputShape, NeuralNetwork},
    resolution::Resolution,
    timer::Timer,
};

pub struct Landmarker {
    model: &'static Cnn,
    t_resize: Timer,
    t_infer: Timer,
    landmarks: Landmarks,
}

impl Landmarker {
    pub const NUM_LANDMARKS: usize = 68;

    pub fn new<N: LandmarkNetwork>(network: N) -> Self {
        drop(network);
        Self {
            model: N::cnn(),
            t_resize: Timer::new("resize"),
            t_infer: Timer::new("infer"),
            landmarks: Landmarks::new(Self::NUM_LANDMARKS),
        }
    }

    /// Returns the expected input resolution of the internal neural network.
    ///
    /// If an image is passed that has a different resolution, it will first be resized (while
    /// adding black bars to retain the original aspect ratio).
    pub fn input_resolution(&self) -> Resolution {
        self.model.input_resolution()
    }

    /// Computes facial landmarks in `image`.
    pub fn compute<V: AsImageView>(&mut self, image: &V) -> &mut Landmarks {
        self.compute_impl(image.as_view())
    }

    fn compute_impl(&mut self, image: ImageView<'_>) -> &mut Landmarks {
        let input_res = self.model.input_resolution();
        let full_res = image.resolution();
        let orig_aspect = full_res.aspect_ratio().unwrap();

        let mut image = image;
        let resized;
        if image.resolution() != input_res {
            resized = self.t_resize.time(|| image.aspect_aware_resize(input_res));
            image = resized.as_view();
        }
        let result = self.t_infer.time(|| self.model.estimate(&image)).unwrap();
        log::trace!("inference result: {:?}", result);

        for (coords, out) in zip_exact(
            result[0].index([0]).as_slice()[..Self::NUM_LANDMARKS * 2].chunks(2),
            self.landmarks.positions_mut(),
        ) {
            let [x, y] = [coords[0], coords[1]];
            out[0] = x;
            out[1] = y;
        }

        // Map landmark coordinates back into the input image.
        for pos in self.landmarks.positions_mut() {
            let [x, y] = [pos[0], pos[1]];
            let (x, y) = unadjust_aspect_ratio(
                x, // / input_res.width() as f32,
                y, // / input_res.height() as f32,
                orig_aspect,
            );
            let (x, y) = (x * full_res.width() as f32, y * full_res.height() as f32);

            pos[0] = x;
            pos[1] = y;
        }

        &mut self.landmarks
    }

    /// Returns profiling timers for image resizing and neural inference.
    pub fn timers(&self) -> impl Iterator<Item = &Timer> + '_ {
        [&self.t_resize, &self.t_infer].into_iter()
    }
}

/// Trait for neural networks that estimate the 68-landmark set.
pub trait LandmarkNetwork {
    fn cnn() -> &'static Cnn;
}

/// The network from [`Peppa-Facial-Landmark-PyTorch`].
///
/// This network is fairly fast to infer, but generates inaccurate results.
///
/// [`Peppa-Facial-Landmark-PyTorch`]: https://github.com/ainrichman/Peppa-Facial-Landmark-PyTorch
pub struct PeppaFacialLandmark;

impl LandmarkNetwork for PeppaFacialLandmark {
    fn cnn() -> &'static Cnn {
        const MODEL_DATA: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/3rdparty/onnx/slim_160_latest.onnx"
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
}

/// The landmark network used by [FaceONNX].
///
/// Slower than [`PeppaFacialLandmark`], taking about twice the time to infer, but more accurate.
/// Sadly this network still doesn't compute accurate landmarks when making anything resembling a
/// grimace.
///
/// [FaceONNX]: https://github.com/FaceONNX/FaceONNX
pub struct FaceOnnx;

impl LandmarkNetwork for FaceOnnx {
    fn cnn() -> &'static Cnn {
        const MODEL_DATA: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/3rdparty/onnx/landmarks_68_pfld.onnx"
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
}
