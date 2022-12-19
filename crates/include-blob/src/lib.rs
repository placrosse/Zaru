pub use include_blob_macros::*;

use std::{
    collections::hash_map::DefaultHasher,
    env,
    fs::{self, File},
    hash::{Hash, Hasher},
    path::Path,
};

use object::{
    write::{Object, StandardSection, Symbol, SymbolSection},
    Architecture, BinaryFormat, Endianness, SymbolFlags, SymbolKind, SymbolScope,
};

/// Call this from your build script.
pub fn include_bytes<A: AsRef<Path>>(path: A) {
    include_bytes_impl(path.as_ref());
}

fn include_bytes_impl(path: &Path) {
    let path = path.canonicalize().unwrap_or_else(|_| {
        panic!(
            "could not find file '{}' (working directory is '{}')",
            path.display(),
            std::env::current_dir().unwrap().display(),
        );
    });
    let metadata = fs::metadata(&path).unwrap();
    assert!(metadata.is_file());

    let mut hasher = DefaultHasher::new();
    path.hash(&mut hasher);
    metadata.modified().unwrap().hash(&mut hasher);
    let unique_name = format!("include_blob_{:016x}", hasher.finish());

    let content = fs::read(&path).unwrap();

    let out_dir = env::var("OUT_DIR").unwrap();
    let mut out_file = File::create(format!("{out_dir}/lib{unique_name}.a")).unwrap();

    let info = TargetInfo::from_build_script_vars();
    let mut obj_buf = Vec::new();
    let mut object = Object::new(info.binfmt, info.arch, info.endian);
    let (section, _) = object.add_subsection(
        StandardSection::ReadOnlyData,
        unique_name.as_bytes(),
        &[],
        1,
    );
    let sym = object.add_symbol(Symbol {
        name: unique_name.as_bytes().to_vec(),
        value: 0,
        size: content.len() as _,
        kind: SymbolKind::Data,
        scope: SymbolScope::Linkage,
        weak: false,
        section: SymbolSection::Section(section),
        flags: SymbolFlags::None,
    });
    object.add_symbol_data(sym, section, &content, 1);
    object.write_stream(&mut obj_buf).unwrap();

    ar::Builder::new(&mut out_file)
        .append(
            &ar::Header::new(
                format!("{unique_name}.o").into_bytes(),
                obj_buf.len() as u64,
            ),
            &obj_buf[..],
        )
        .unwrap();

    println!("cargo:rustc-link-lib=static={unique_name}");
    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rerun-if-changed={}", path.display());
}

struct TargetInfo {
    binfmt: BinaryFormat,
    arch: Architecture,
    endian: Endianness,
}

impl TargetInfo {
    fn from_build_script_vars() -> Self {
        let binfmt = match &*env::var("CARGO_CFG_TARGET_OS").unwrap() {
            "macos" | "ios" => BinaryFormat::MachO,
            "windows" => BinaryFormat::Pe,
            "linux" | "android" => BinaryFormat::Elf,
            unk => panic!("unhandled operating system '{unk}'"),
        };
        let arch = match &*env::var("CARGO_CFG_TARGET_ARCH").unwrap() {
            // NB: this is guesswork, because apparently the Rust team can't be bothered to document
            // the *full* list anywhere (they differ from what the target triples use, which *are*
            // fully documented)
            "x86" => Architecture::I386,
            "x86_64" => Architecture::X86_64,
            "aarch64" => Architecture::Aarch64,
            "arm" => Architecture::Arm,
            "riscv32" => Architecture::Riscv32,
            "riscv64" => Architecture::Riscv64,
            unk => panic!("unhandled architecture '{unk}'"),
        };
        let endian = match &*env::var("CARGO_CFG_TARGET_ENDIAN").unwrap() {
            "little" => Endianness::Little,
            "big" => Endianness::Big,
            unk => unreachable!("unhandled endianness '{unk}'"),
        };

        Self {
            binfmt,
            arch,
            endian,
        }
    }
}
