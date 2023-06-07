use std::{
    env::{self, VarError},
    process,
    sync::OnceLock,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OnnxBackend {
    Tract,
    OnnxRuntime,
}

static ONNX_RT: OnceLock<OnnxBackend> = OnceLock::new();

pub(super) fn get() -> OnnxBackend {
    *ONNX_RT.get_or_init(|| {
        let backend = match env::var("ZARU_ONNX_BACKEND").as_deref() {
            Ok("tract") | Err(VarError::NotPresent) => OnnxBackend::Tract,
            Ok("onnxruntime" | "ort") => OnnxBackend::OnnxRuntime,
            Ok(invalid) => {
                eprintln!(
                    "invalid value set for `ZARU_ONNX_BACKEND` variable: '{invalid}'; exiting"
                );
                process::exit(1);
            }
            Err(VarError::NotUnicode(s)) => {
                eprintln!(
                    "invalid value set for `ZARU_ONNX_BACKEND` variable: {}; exiting",
                    s.to_string_lossy()
                );
                process::exit(1);
            }
        };
        log::debug!("using ONNX backend {:?}", backend);
        backend
    })
}
