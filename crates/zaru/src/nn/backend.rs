use std::{
    env::{self, VarError},
    process,
    sync::OnceLock,
};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) enum OnnxBackend {
    Tract,
    #[default]
    OnnxRuntime,
}

static BACKEND: OnceLock<OnnxBackend> = OnceLock::new();

pub(super) fn get() -> OnnxBackend {
    *BACKEND.get_or_init(|| {
        let backend = match env::var("ZARU_ONNX_BACKEND").as_deref() {
            Ok("tract") => OnnxBackend::Tract,
            Ok("onnxruntime" | "ort") => OnnxBackend::OnnxRuntime,
            Err(VarError::NotPresent) => OnnxBackend::default(),
            Ok(invalid) => {
                eprintln!(
                    "invalid value set for `ZARU_ONNX_BACKEND` variable: '{invalid}'; exiting"
                );
                process::exit(1);
            }
            Err(VarError::NotUnicode(s)) => {
                eprintln!(
                    "invalid value set for `ZARU_ONNX_BACKEND` variable: '{}'; exiting",
                    s.to_string_lossy()
                );
                process::exit(1);
            }
        };
        log::debug!("using ONNX backend {:?}", backend);
        backend
    })
}
