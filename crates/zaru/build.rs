use include_blob::include_bytes;

fn main() {
    // hmm.... globs?
    include_bytes("../../3rdparty/onnx/face_detection_full_range.onnx");
    include_bytes("../../3rdparty/onnx/face_detection_short_range.onnx");
    include_bytes("../../3rdparty/onnx/face_landmark.onnx");
    include_bytes("../../3rdparty/onnx/hand_landmark_full.onnx");
    include_bytes("../../3rdparty/onnx/hand_landmark_lite.onnx");
    include_bytes("../../3rdparty/onnx/iris_landmark.onnx");
    include_bytes("../../3rdparty/onnx/landmarks_68_pfld.onnx");
    include_bytes("../../3rdparty/onnx/mobilefacenet.onnx");
    include_bytes("../../3rdparty/onnx/palm_detection_full.onnx");
    include_bytes("../../3rdparty/onnx/palm_detection_lite.onnx");
    include_bytes("../../3rdparty/onnx/pose_detection.onnx");
    include_bytes("../../3rdparty/onnx/pose_landmark_full.onnx");
    include_bytes("../../3rdparty/onnx/pose_landmark_lite.onnx");
    include_bytes("../../3rdparty/onnx/slim_160_latest.onnx");
}
