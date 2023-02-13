//! Neural Network inference.

pub mod tensor;

use crate::image::{AsImageView, Color, ImageView, Resolution};
use tensor::Tensor;
use tract_onnx::prelude::{
    tvec, Framework, Graph, InferenceModelExt, SimplePlan, TValue, TVec, TypedFact, TypedOp,
};
use wonnx::utils::{InputTensor, OutputTensor};

use std::{
    borrow::Cow,
    ops::{Index, Range, RangeInclusive},
    path::Path,
    sync::Arc,
};

type Model = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// A convolutional neural network (CNN) that operates on image data.
///
/// Like the underlying [`NeuralNetwork`], this is a cheaply [`Clone`]able handle to the underlying
/// data.
#[derive(Clone)]
pub struct Cnn {
    nn: NeuralNetwork,
    input_res: Resolution,
    image_map: Arc<dyn Fn(ImageView<'_>) -> Tensor + Send + Sync>,
}

impl Cnn {
    /// Creates a CNN wrapper from a [`NeuralNetwork`].
    ///
    /// The network must have exactly one input with a shape that matches the given
    /// [`CnnInputShape`].
    pub fn new(
        nn: NeuralNetwork,
        shape: CnnInputShape,
        color_mapper: ColorMapper,
    ) -> anyhow::Result<Self> {
        let input_res = Self::get_input_res(&nn, shape)?;
        let (h, w) = (input_res.height() as usize, input_res.width() as usize);

        fn sample(view: &ImageView<'_>, u: f32, v: f32) -> Color {
            let x = (u * view.resolution().width() as f32).round() as u32;
            let y = (v * view.resolution().height() as f32).round() as u32;
            view.get(x, y)
        }

        // Box a closure that maps the whole input image to a tensor. That way we avoid dynamic
        // dispatch as much as possible.
        let image_map: Arc<dyn Fn(ImageView<'_>) -> _ + Send + Sync> = match shape {
            CnnInputShape::NCHW => Arc::new(move |view| {
                Tensor::from_array_shape_fn([1, 3, h, w], |[_, c, y, x]| {
                    color_mapper.map(sample(&view, x as f32 / w as f32, y as f32 / h as f32))[c]
                })
            }),
            CnnInputShape::NHWC => Arc::new(move |view| {
                Tensor::from_array_shape_fn([1, h, w, 3], |[_, y, x, c]| {
                    color_mapper.map(sample(&view, x as f32 / w as f32, y as f32 / h as f32))[c]
                })
            }),
        };

        Ok(Self {
            nn,
            input_res,
            image_map,
        })
    }

    fn get_input_res(nn: &NeuralNetwork, shape: CnnInputShape) -> anyhow::Result<Resolution> {
        if nn.num_inputs() != 1 {
            anyhow::bail!(
                "CNN network has to take exactly 1 input, this one takes {}",
                nn.num_inputs(),
            );
        }

        let input_info = nn.inputs().next().unwrap();
        let tensor_shape = input_info.shape();

        let (w, h) = match (shape, tensor_shape) {
            (CnnInputShape::NCHW, [1, 3, h, w]) | (CnnInputShape::NHWC, [1, h, w, 3]) => (*w, *h),
            _ => {
                anyhow::bail!(
                    "invalid model input shape for {:?} CNN: {:?}",
                    shape,
                    tensor_shape,
                );
            }
        };

        let (w, h): (u32, u32) = (w.try_into()?, h.try_into()?);
        Ok(Resolution::new(w, h))
    }

    /// Returns the expected input image size.
    #[inline]
    pub fn input_resolution(&self) -> Resolution {
        self.input_res
    }

    /// Runs the network on an input image, returning the estimated outputs.
    ///
    /// The input image will be sampled to create the network's input tensor. If the image's aspect
    /// ratio does not match the network's input aspect ratio, the image will be stretched.
    pub fn estimate<V: AsImageView>(&self, image: &V) -> anyhow::Result<Outputs> {
        self.estimate_impl(image.as_view())
    }

    fn estimate_impl(&self, image: ImageView<'_>) -> anyhow::Result<Outputs> {
        let tensor = (self.image_map)(image);

        self.nn.estimate(&Inputs::from(tensor))
    }
}

enum ColorMapperKind {
    Linear { target_range: RangeInclusive<f32> },
}

pub struct ColorMapper {
    kind: ColorMapperKind,
}

impl ColorMapper {
    /// Creates a simple color mapper that uniformly maps sRGB values to `target_range`.
    ///
    /// The returned object can be passed directly to [`Cnn::new`] as its color map.
    ///
    /// Note that this operates on *non-linear* sRGB colors, but maps them linearly to the target range.
    /// The assumption is that sRGB is the color space most (all?) CNNs expect their inputs to be in,
    /// but in practice none of them document this.
    pub fn linear(target_range: RangeInclusive<f32>) -> Self {
        let start = *target_range.start();
        let end = *target_range.end();
        assert!(end > start);

        Self {
            kind: ColorMapperKind::Linear { target_range },
        }
    }

    fn map(&self, color: Color) -> [f32; 3] {
        match &self.kind {
            ColorMapperKind::Linear { target_range } => {
                let start = *target_range.start();
                let end = *target_range.end();

                let adjust_range = (end - start) / 255.0;
                let rgb = [color.r(), color.g(), color.b()];
                rgb.map(|col| col as f32 * adjust_range + start)
            }
        }
    }
}

/// Describes in what order a CNN expects its input image data.
///
/// - `N` is the number of images, often fixed at 1.
/// - `C` is the number of color channels, often 3 for RGB inputs.
/// - `H` and `W` are the height and width of the input, respectively.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive] // shouldn't be matched on by user code
pub enum CnnInputShape {
    /// Shape is `[N, C, H, W]`.
    NCHW,
    /// Shape is `[N, H, W, C]`.
    NHWC,
}

/// Neural network loader.
pub struct Loader<'a> {
    model_data: Cow<'a, [u8]>,
    outputs: Option<Vec<usize>>,
    enable_gpu: bool,
}

impl<'a> Loader<'a> {
    fn new(data: Cow<'a, [u8]>) -> Self {
        Self {
            model_data: data,
            outputs: None,
            enable_gpu: false,
        }
    }

    /// Instructs the neural network loader to enable GPU support for this network.
    ///
    /// If this method is called and the GPU backend does not support the network, [`Loader::load`]
    /// will return an error.
    ///
    /// Note that the GPU backend [`wonnx`] is still in early stages and does not support most of
    /// the networks used in this project.
    pub fn with_gpu_support(mut self) -> Self {
        self.enable_gpu = true;
        self
    }

    /// Only compute the specified outputs during inference.
    ///
    /// This takes a list of [`usize`]s corresponding to network output indices. When called, the
    /// [`Outputs`] returned from [`NeuralNetwork::estimate`] will only contain the chosen output
    /// tensors.
    ///
    /// # Example
    ///
    /// A network computes 5 output tensors by default: `A`, `B`, `C`, `D`, and `E`. We only want
    /// `A`, `D`, and `E`, so we pass `[0, 3, 4]` to [`Loader::with_output_selection`]. The loaded
    /// [`NeuralNetwork`] will now return 3 [`Tensor`]s in its [`Outputs`], corresponding to `A`,
    /// `D`, and `E`.
    pub fn with_output_selection<O>(mut self, outputs: O) -> Self
    where
        O: Into<Vec<usize>>,
    {
        self.outputs = Some(outputs.into());
        self
    }

    /// Loads and optimizes the network.
    ///
    /// Returns an error if the network data is malformed, if the network data is incomplete, or if
    /// the network uses unimplemented operations.
    pub fn load(self) -> anyhow::Result<NeuralNetwork> {
        let graph = tract_onnx::onnx()
            .model_for_read(&mut &*self.model_data)?
            .into_optimized()?;
        let outputs = graph.output_outlets()?;
        let selected_outputs = match self.outputs {
            Some(indices) => indices.iter().map(|&i| outputs[i]).collect::<Vec<_>>(),
            None => outputs.to_vec(),
        };
        let model = SimplePlan::new_for_outputs(graph, &selected_outputs)?;

        let gpu = if self.enable_gpu {
            Some(pollster::block_on(wonnx::Session::from_bytes(
                &self.model_data,
            ))?)
        } else {
            None
        };

        Ok(NeuralNetwork(Arc::new(NeuralNetworkImpl {
            inner: model,
            gpu,
        })))
    }
}

/// A neural network that can be used for inference.
///
/// This is a cheaply [`Clone`]able handle to the underlying network structures.
#[derive(Clone)]
pub struct NeuralNetwork(Arc<NeuralNetworkImpl>);

struct NeuralNetworkImpl {
    inner: Model,
    gpu: Option<wonnx::Session>,
}

impl NeuralNetwork {
    /// Loads a pre-trained model from an ONNX file path.
    ///
    /// The path must have a `.onnx` extension. In the future, other model formats may be supported.
    pub fn from_path<'a, P: AsRef<Path>>(path: P) -> anyhow::Result<Loader<'a>> {
        Self::from_path_impl(path.as_ref())
    }

    fn from_path_impl<'a>(path: &Path) -> anyhow::Result<Loader<'a>> {
        match path.extension() {
            Some(ext) if ext == "onnx" => {}
            _ => anyhow::bail!("neural network file must have `.onnx` extension"),
        }

        let model_data = std::fs::read(path)?;
        Ok(Loader::new(model_data.into()))
    }

    /// Loads a pre-trained model from an in-memory ONNX file.
    pub fn from_onnx(raw: &[u8]) -> anyhow::Result<Loader<'_>> {
        Ok(Loader::new(raw.into()))
    }

    /// Returns the number of input nodes of the network.
    pub fn num_inputs(&self) -> usize {
        self.0.inner.model().inputs.len()
    }

    /// Returns the number of output nodes of the network.
    pub fn num_outputs(&self) -> usize {
        self.0.inner.model().outputs.len()
    }

    /// Returns an iterator over the network's input node information.
    ///
    /// To perform inference, a matching input tensor has to be provided for each input.
    pub fn inputs(&self) -> InputInfoIter<'_> {
        InputInfoIter {
            net: self,
            ids: 0..self.num_inputs(),
        }
    }

    /// Returns an iterator over the network's output node information.
    pub fn outputs(&self) -> OutputInfoIter<'_> {
        OutputInfoIter {
            net: self,
            ids: 0..self.num_outputs(),
        }
    }

    /// Runs the network on a set of [`Inputs`], returning the estimated [`Outputs`].
    ///
    /// If the network was loaded with GPU support enabled, computation will happen there.
    /// Otherwise, it is done on the CPU.
    #[doc(alias = "infer")]
    pub fn estimate(&self, inputs: &Inputs) -> anyhow::Result<Outputs> {
        let outputs = match &self.0.gpu {
            Some(gpu) => {
                let inputs = self
                    .inputs()
                    .zip(inputs.iter())
                    .map(|(info, tensor)| {
                        let name = info.name().to_string();
                        let input = InputTensor::F32(tensor.as_raw_data().into());
                        (name, input)
                    })
                    .collect();

                let output_map = pollster::block_on(gpu.run(&inputs))?;
                let mut outputs = TVec::new();
                for info in self.outputs() {
                    let tensor = &output_map[info.name()];
                    match tensor {
                        OutputTensor::F32(tensor) => {
                            outputs.push(Tensor::from_iter(info.shape(), tensor.iter().copied()));
                        }
                        _ => unreachable!(),
                    }
                }

                Outputs { inner: outputs }
            }
            None => {
                let outputs = self.0.inner.run(
                    inputs
                        .iter()
                        .map(|t| TValue::from_const(Arc::new(t.to_tract())))
                        .collect(),
                )?;
                let outputs = outputs
                    .into_iter()
                    .map(|tract| Tensor::from_tract(&tract))
                    .collect();
                Outputs { inner: outputs }
            }
        };

        Ok(outputs)
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

        let model = &self.net.0.inner.model();
        let fact = model.input_fact(id).expect("`input_fact` returned error");

        let node = model.input_outlets().unwrap()[id].node;

        Some(InputInfo {
            shape: fact.shape.as_concrete().expect(
                "symbolic network input shape, this makes no \
                sense, by which I mean I don't understand what this means",
            ),
            name: &model.node(node).name,
        })
    }
}

/// Information about a neural network input node.
#[derive(Debug)]
pub struct InputInfo<'a> {
    shape: &'a [usize],
    name: &'a str,
}

impl<'a> InputInfo<'a> {
    /// Returns the tensor shape for this input.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.shape
    }

    /// Returns the name of this input.
    #[inline]
    pub fn name(&self) -> &str {
        self.name
    }
}

/// Iterator over a [`NeuralNetwork`]s output node information.
pub struct OutputInfoIter<'a> {
    net: &'a NeuralNetwork,
    ids: Range<usize>,
}

impl<'a> Iterator for OutputInfoIter<'a> {
    type Item = OutputInfo<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let id = self.ids.next()?;

        let model = &self.net.0.inner.model();
        let fact = model.output_fact(id).expect("`output_fact` returned error");

        let node = model.output_outlets().unwrap()[id].node;

        Some(OutputInfo {
            shape: fact.shape.as_concrete().expect(
                "symbolic network output shape, this makes no \
                sense, by which I mean I don't understand what this means",
            ),
            name: &model.node(node).name,
        })
    }
}

/// Information about a neural network output node.
#[derive(Debug)]
pub struct OutputInfo<'a> {
    shape: &'a [usize],
    name: &'a str,
}

impl<'a> OutputInfo<'a> {
    /// Returns the tensor shape for this output.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.shape
    }

    /// Returns the name of this output.
    #[inline]
    pub fn name(&self) -> &str {
        self.name
    }
}

/// The result of a neural network inference pass.
///
/// This is a list of tensors corresponding to the network's output nodes.
#[derive(Debug)]
pub struct Outputs {
    inner: TVec<Tensor>,
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

    fn index(&self, index: usize) -> &Tensor {
        &self.inner[index]
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
    inner: std::slice::Iter<'a, Tensor>,
}

impl<'a> Iterator for OutputIter<'a> {
    type Item = &'a Tensor;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

/// List of input tensors for neural network inference.
#[derive(Debug)]
pub struct Inputs {
    inner: TVec<Tensor>,
}

impl Inputs {
    /// Returns the number of input tensors stored in `self`.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    fn iter(&self) -> impl Iterator<Item = &Tensor> {
        self.inner.iter()
    }
}

impl From<Tensor> for Inputs {
    fn from(t: Tensor) -> Self {
        Self { inner: tvec![t] }
    }
}

impl<const N: usize> From<[Tensor; N]> for Inputs {
    fn from(tensors: [Tensor; N]) -> Self {
        Self {
            inner: tensors.into_iter().collect(),
        }
    }
}

impl FromIterator<Tensor> for Inputs {
    fn from_iter<T: IntoIterator<Item = Tensor>>(iter: T) -> Self {
        Self {
            inner: iter.into_iter().collect(),
        }
    }
}

impl Extend<Tensor> for Inputs {
    fn extend<T: IntoIterator<Item = Tensor>>(&mut self, iter: T) {
        self.inner.extend(iter);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_mapper() {
        let mapper = ColorMapper::linear(-1.0..=1.0);
        assert_eq!(mapper.map(Color::BLACK), [-1.0, -1.0, -1.0]);
        assert_eq!(mapper.map(Color::WHITE), [1.0, 1.0, 1.0]);

        let mapper = ColorMapper::linear(1.0..=2.0);
        assert_eq!(mapper.map(Color::BLACK), [1.0, 1.0, 1.0]);
        assert_eq!(mapper.map(Color::WHITE), [2.0, 2.0, 2.0]);
    }
}
