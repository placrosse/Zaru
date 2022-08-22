# 3rd party files

Files in this directory and its subdirectories were authored by 3rd parties.
This file documents their origin and license.

Note that files may have been converted from other formats.

| File | Source | License |
|------|--------|---------|
| `3d/canonical_face_model.obj` | [MediaPipe] | Apache-2.0
| `3d/canonical_face_model.rs` | [MediaPipe] | Apache-2.0
| `onnx/face_detection_full_range.onnx` | [MediaPipe] | Apache-2.0
| `onnx/face_detection_short_range.onnx` | [MediaPipe] | Apache-2.0
| `onnx/face_landmark.onnx` | [MediaPipe] | Apache-2.0
| `onnx/hand_landmark_full.onnx` | [MediaPipe] | Apache-2.0
| `onnx/hand_landmark_lite.onnx` | [MediaPipe] | Apache-2.0
| `onnx/iris_landmark.onnx` | [MediaPipe] | Apache-2.0
| `onnx/mobilefacenet.onnx` | [InsightFace_PyTorch] | MIT
| `onnx/palm_detection_full.onnx` | [MediaPipe] | Apache-2.0
| `onnx/palm_detection_lite.onnx` | [MediaPipe] | Apache-2.0
| `onnx/pose_detection.onnx` | [MediaPipe] | Apache-2.0
| `onnx/pose_landmark_full.onnx` | [MediaPipe] | Apache-2.0
| `onnx/pose_landmark_lite.onnx` | [MediaPipe] | Apache-2.0

[MediaPipe]: https://github.com/google/mediapipe
[InsightFace_Pytorch]: https://github.com/TreB1eN/InsightFace_Pytorch

## Neural Network conversion

The deep learning tooling situation is a nightmare. Here's some random and unhelpful notes:

- Targeting ONNX opset 9 may introduce the deprecated and (in tract) unimplemented op "Upsample",
  which has been replaced with "Resize". Target opset 13 instead, which makes the converter emit
  "Resize" instead.
- `tf2onnx` by default will use the TensorFlow CNN format, which introduces an unnecessary
  "Transpose" node at the input. Pass `--inputs-as-nchw input_1` to it to avoid this.
- Attempting to convert sparse TFLite models with `tf2onnx` might fail due to a segfault. In that
  case, [`tflite2tensorflow`] can be used to convert the model to a regular, dense TensorFlow
  `saved_model`, which can then be converted to an ONNX model.

[`tflite2tensorflow`]: https://github.com/PINTO0309/tflite2tensorflow
