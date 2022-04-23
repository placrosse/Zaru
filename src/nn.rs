//! Neural Network inference.

use std::{
    ops::{Index, Range},
    path::Path,
    sync::Arc,
};

use tract_onnx::{
    prelude::{
        tvec, Framework, Graph, InferenceModelExt, SimplePlan, TVec, Tensor, TypedFact, TypedOp,
    },
    tract_hir::tract_ndarray::Array4,
};

use crate::{
    image::{AsImageView, ImageView},
    resolution::{AspectRatio, Resolution},
    Error,
};

type Model = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// A convolutional neural network (CNN) that operates on image data.
///
/// Neural networks using this interface are expected use take sRGB color data scaled down to a
/// range of `[-1.0, 1.0]`. Now that I think about it, this sounds wrong, these values might have to
/// be linearized first. This should also be made configurable since not every CNN takes the same
/// input.
pub struct Cnn {
    nn: NeuralNetwork,
    shape: CnnInputShape,
    input_res: Resolution,
    color_map: fn(u8) -> f32,
}

impl Cnn {
    /// Creates a CNN wrapper from a [`NeuralNetwork`].
    ///
    /// The network must have exactly one input with a shape that matches `fmt`.
    pub fn new(nn: NeuralNetwork, shape: CnnInputShape) -> Result<Self, Error> {
        if nn.num_inputs() != 1 {
            return Err(format!(
                "CNN network has to take 1 input, this one takes {}",
                nn.num_inputs()
            )
            .into());
        }

        let input_info = nn.inputs().next().unwrap();
        let tensor_shape = input_info.shape();

        let (w, h) = match (shape, tensor_shape) {
            (CnnInputShape::NCHW, [1, 3, h, w]) | (CnnInputShape::NHWC, [1, h, w, 3]) => (*w, *h),
            _ => {
                return Err(format!(
                    "invalid model input shape for {:?} CNN: {:?}",
                    shape, tensor_shape
                )
                .into());
            }
        };

        let (w, h): (u32, u32) = (w.try_into()?, h.try_into()?);
        let input_res = Resolution::new(w, h);

        Ok(Self {
            nn,
            shape,
            input_res,
            color_map: map_color,
        })
    }

    pub fn set_color_map(&mut self, map: fn(u8) -> f32) {
        self.color_map = map;
    }

    /// Returns the expected input image size.
    #[inline]
    pub fn input_resolution(&self) -> Resolution {
        self.input_res
    }

    /// Runs network inference on the given input image.
    ///
    /// The image's resolution must match the CNN's [`input_resolution`][Self::input_resolution],
    /// otherwise this method will panic.
    pub fn infer<V: AsImageView>(&self, image: &V) -> Result<Outputs, Error> {
        self.infer_impl(image.as_view())
    }

    fn infer_impl(&self, image: ImageView<'_>) -> Result<Outputs, Error> {
        assert_eq!(
            image.resolution(),
            self.input_resolution(),
            "CNN input image does not have expected resolution"
        );

        let (h, w) = (
            self.input_res.height() as usize,
            self.input_res.width() as usize,
        );
        let tensor = match self.shape {
            CnnInputShape::NCHW => Array4::from_shape_fn((1, 3, h, w), |(_, c, y, x)| {
                (self.color_map)(image.get(x as _, y as _)[c])
            }),
            CnnInputShape::NHWC => Array4::from_shape_fn((1, h, w, 3), |(_, y, x, c)| {
                (self.color_map)(image.get(x as _, y as _)[c])
            }),
        };

        self.nn.infer(Inputs::single(tensor.into()))
    }
}

fn map_color(value: u8) -> f32 {
    // Output range: -1.0 ... 1.0
    (value as f32 / 255.0 - 0.5) * 2.0
}

/// Adjusts `f32` coordinates from a 1:1 aspect ratio back to `orig_ratio`.
///
/// This assumes that `orig_ratio` was originally fitted to a 1:1 ratio by adding black bars
/// ([`Image::aspect_aware_resize`]).
///
/// [`Image::aspect_aware_resize`]: crate::image::Image::aspect_aware_resize
pub(crate) fn unadjust_aspect_ratio(
    mut x: f32,
    mut y: f32,
    orig_aspect: AspectRatio,
) -> (f32, f32) {
    let ratio = orig_aspect.as_f32();
    if ratio > 1.0 {
        // going from 1:1 to something wider, undo letterboxing
        y = (y - 0.5) * ratio + 0.5;
    } else {
        // going from 1:1 to something taller, undo pillarboxing
        x = (x - 0.5) / ratio + 0.5;
    }

    (x, y)
}

/// Translates `f32` coordinates back to coordinates of an image with resolution `full_res`.
///
/// The input coordinates are assumed to be in range `[0.0, 1.0]` (and thus from a square image with
/// aspect ratio 1:1). The original image may have any aspect ratio, but this function assumes that
/// it was translated to a 1:1 ratio using [`Image::aspect_aware_resize`].
///
/// [`Image::aspect_aware_resize`]: crate::image::Image::aspect_aware_resize
pub(crate) fn point_to_img(x: f32, y: f32, full_res: &Resolution) -> (i32, i32) {
    let (x, y) = unadjust_aspect_ratio(x, y, full_res.aspect_ratio());

    let x = (x * full_res.width() as f32) as i32;
    let y = (y * full_res.height() as f32) as i32;
    (x, y)
}

/// Describes in what order a CNN expects its input image data.
///
/// - `N` is the number of images, often fixed at 1.
/// - `C` is the number of color channels, often 3 for RGB inputs.
/// - `H` and `W` are the height and width of the input, respectively.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CnnInputShape {
    /// Shape is `(N, C, H, W)`.
    NCHW,
    /// Shape is `(N, H, W, C)`.
    NHWC,
}

/// A neural network that can be used for inference.
pub struct NeuralNetwork {
    inner: Model,
}

impl NeuralNetwork {
    /// Loads a pre-trained model from an ONNX file path.
    ///
    /// The path must have a `.onnx` extension. In the future, other model formats may be supported.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        Self::load_impl(path.as_ref())
    }

    fn load_impl(path: &Path) -> Result<Self, Error> {
        match path.extension() {
            Some(ext) if ext == "onnx" => {}
            _ => return Err(format!("neural network path must have `.onnx` extension").into()),
        }

        let graph = tract_onnx::onnx().model_for_path(path)?;
        let model = graph.into_optimized()?.into_runnable()?;

        Ok(Self { inner: model })
    }

    /// Loads a pre-trained model from an in-memory ONNX file.
    pub fn from_onnx(mut raw: &[u8]) -> Result<Self, Error> {
        let graph = tract_onnx::onnx().model_for_read(&mut raw)?;
        let model = graph.into_optimized()?.into_runnable()?;

        Ok(Self { inner: model })
    }

    /// Returns the number of input nodes of the network.
    pub fn num_inputs(&self) -> usize {
        self.inner.model().inputs.len()
    }

    /// Returns an iterator over the network's inputs.
    ///
    /// To perform inference, a matching input tensor has to be provided for each input.
    pub fn inputs(&self) -> InputInfoIter<'_> {
        InputInfoIter {
            net: self,
            ids: 0..self.num_inputs(),
        }
    }

    /// Runs an inference pass on the given input tensors.
    pub fn infer(&self, inputs: Inputs) -> Result<Outputs, Error> {
        let outputs = self.inner.run(inputs.inner)?;

        Ok(Outputs { inner: outputs })
    }
}

/// Iterator over a [`NeuralNetwork`]s input information.
pub struct InputInfoIter<'a> {
    net: &'a NeuralNetwork,
    ids: Range<usize>,
}

impl<'a> Iterator for InputInfoIter<'a> {
    type Item = InputInfo<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let id = self.ids.next()?;

        let fact = self
            .net
            .inner
            .model()
            .input_fact(id)
            .expect("`input_fact` returned error");

        Some(InputInfo {
            shape: fact.shape.as_concrete().expect(
                "symbolic network input shape, this makes no \
                sense, by which I mean I don't understand what this means",
            ),
        })
    }
}

/// Information about a neural network input node.
#[derive(Debug)]
pub struct InputInfo<'a> {
    shape: &'a [usize],
}

impl<'a> InputInfo<'a> {
    /// Returns the tensor shape for this input.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.shape
    }
}

/// The result of a neural network inference pass.
///
/// This is a list of tensors corresponding to the network's output nodes.
#[derive(Debug)]
pub struct Outputs {
    inner: TVec<Arc<Tensor>>,
}

impl Outputs {
    /// Returns the number of tensors in this inference output.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns an iterator over the output tensors.
    pub fn iter(&self) -> OutputIter<'_> {
        OutputIter {
            inner: self.inner.iter(),
        }
    }
}

impl Index<usize> for Outputs {
    type Output = Tensor;

    fn index(&self, index: usize) -> &Self::Output {
        &*self.inner[index]
    }
}

impl<'a> IntoIterator for &'a Outputs {
    type Item = &'a Tensor;
    type IntoIter = OutputIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator over a list of output tensors.
pub struct OutputIter<'a> {
    inner: std::slice::Iter<'a, Arc<Tensor>>,
}

impl<'a> Iterator for OutputIter<'a> {
    type Item = &'a Tensor;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|arc| &**arc)
    }
}

/// List of input tensors for neural network inference.
#[derive(Debug)]
pub struct Inputs {
    inner: TVec<Tensor>,
}

impl Inputs {
    /// Creates a network input from a single input tensor.
    pub fn single(tensor: Tensor) -> Self {
        Self {
            inner: tvec![tensor],
        }
    }

    /// Creates a network input from an array of tensors.
    pub fn from_array<const N: usize>(tensors: [Tensor; N]) -> Self {
        Self {
            inner: tensors.into_iter().collect(),
        }
    }

    /// Returns the number of input tensors stored in `self`.
    pub fn len(&self) -> usize {
        self.inner.len()
    }
}
