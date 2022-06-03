//! This does not currently work, either because of a bug in tract or because I misunderstood the
//! model.

use once_cell::sync::Lazy;

use crate::{
    detection::RawDetection,
    image::{self, AsImageView, AsImageViewMut, ImageView, ImageViewMut, Rect},
    nn::{create_linear_color_mapper, Cnn, CnnInputShape, NeuralNetwork},
    num::{sigmoid, TotalF32},
    resolution::Resolution,
    timer::Timer,
};

pub struct PalmDetector {
    model: &'static Cnn,
    t_resize: Timer,
    t_infer: Timer,
    t_nms: Timer,
    raw_detections: Vec<RawDetection>,
    detections: Vec<Detection>,
}

impl PalmDetector {
    pub fn new<P: PalmDetectionNetwork>(model: P) -> Self {
        drop(model);
        Self {
            model: P::cnn(),
            t_resize: Timer::new("resize"),
            t_infer: Timer::new("infer"),
            t_nms: Timer::new("NMS"),
            raw_detections: Vec::new(),
            detections: Vec::new(),
        }
    }

    /// Returns the expected input resolution of the internal neural network.
    pub fn input_resolution(&self) -> Resolution {
        self.model.input_resolution()
    }

    pub fn detect<A: AsImageView>(&mut self, image: A) -> &[Detection] {
        self.detect_impl(image.as_view())
    }

    fn detect_impl(&mut self, image: ImageView<'_>) -> &[Detection] {
        self.raw_detections.clear();
        self.detections.clear();

        //let full_res = image.resolution();

        let mut image = image.reborrow();
        let resized;
        if image.resolution() != self.input_resolution() {
            resized = self
                .t_resize
                .time(|| image.aspect_aware_resize(self.model.input_resolution()));
            image = resized.as_view();
        }
        let result = self.t_infer.time(|| self.model.estimate(&image)).unwrap();
        log::trace!("inference result: {:?}", result);

        self.t_nms.time(|| {
            let boxes = &result[0];
            let confidences = &result[1];

            assert_eq!(boxes.shape(), &[1, 2016, 18]);
            assert_eq!(confidences.shape(), &[1, 2016, 1]);
            let max = confidences
                .index([0])
                .iter()
                .map(|view| sigmoid(view.as_slice()[0]))
                .min_by_key(|conf| TotalF32(*conf))
                .unwrap();
            // Bug: the score is lower than expected, and actually gets lower if a hand is in view

            eprintln!("{}", max);

            /*for (index, view) in confidences.index([0]).iter().enumerate() {
                let conf = view.as_slice()[0];

                // The confidence can be negative, in which case we don't want to contribute
                // negative weights to the NMA average.
                if conf <= 0.0 {
                    continue;
                }

                let tensor_view = boxes.index([0, index]);
                let box_params = &tensor_view.as_slice()[..16];
                self.raw_detections
                    .push(extract_detection(&self.anchors[index], box_params, conf));
            }

            let detections = self.nms.process(&mut self.raw_detections);
            for raw in detections {
                self.detections.push(Detection { raw, full_res });
            }*/
        });

        &self.detections
    }
}

pub struct Detection {
    raw: RawDetection,
    full_res: Resolution,
}

impl Detection {
    pub fn bounding_rect(&self) -> Rect {
        self.raw.bounding_rect().to_rect(&self.full_res)
    }

    /// Draws the bounding box of this detection onto an image.
    ///
    /// # Panics
    ///
    /// The image must have the same resolution as the image the detection was performed on,
    /// otherwise this method will panic.
    pub fn draw<I: AsImageViewMut>(&self, image: &mut I) {
        self.draw_impl(&mut image.as_view_mut());
    }

    fn draw_impl(&self, image: &mut ImageViewMut<'_>) {
        let res = Resolution::new(image.width(), image.height());
        assert_eq!(
            res, self.full_res,
            "attempted to draw `Detection` onto canvas with mismatched size",
        );

        image::draw_rect(image, self.bounding_rect());
    }
}

pub trait PalmDetectionNetwork {
    fn cnn() -> &'static Cnn;
}

pub struct LiteNetwork;

impl PalmDetectionNetwork for LiteNetwork {
    fn cnn() -> &'static Cnn {
        const MODEL_DATA: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/3rdparty/onnx/palm_detection_lite.onnx"
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

pub struct FullNetwork;

impl PalmDetectionNetwork for FullNetwork {
    fn cnn() -> &'static Cnn {
        const MODEL_DATA: &[u8] = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/3rdparty/onnx/palm_detection_full.onnx"
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
