use std::{
    collections::hash_map::DefaultHasher,
    env, fs,
    hash::{Hash, Hasher},
    path::PathBuf,
    str::FromStr,
};

use proc_macro::{TokenStream, TokenTree};

/// Includes a file that was prepared for inclusion by a build script.
///
/// Takes a string literal as its argument, denoting the file's path (relative to the directory
/// containing the package's `Cargo.toml`).
#[proc_macro]
pub fn include_bytes(args: TokenStream) -> TokenStream {
    let tts = args.into_iter().collect::<Vec<_>>();
    if tts.len() != 1 {
        panic!(
            "`include_bytes!` requires exactly 1 token as its argument (the file path to include)"
        );
    }
    let TokenTree::Literal(lit) = &tts[0] else {
        panic!("`include_bytes!` requires a string literal as its argument");
    };

    let lit = lit.to_string();
    if lit.chars().next() != Some('"') || lit.chars().last() != Some('"') {
        panic!("`include_bytes!` expects a simple string literal as its argument (no raw strings or byte strings are allowed)");
    }
    let lit = &lit[1..lit.len() - 1];
    // TODO: handle escapes and the entire string literal syntax

    let mut path = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    path.push(lit);

    let path = path.canonicalize().unwrap_or_else(|_| {
        panic!("could not find file '{}'", path.display(),);
    });
    let metadata = fs::metadata(&path).unwrap();
    assert!(metadata.is_file());
    let len = metadata.len();

    let mut hasher = DefaultHasher::new();
    path.hash(&mut hasher);
    metadata.modified().unwrap().hash(&mut hasher);
    let unique_name = format!("include_blob_{:016x}", hasher.finish());

    TokenStream::from_str(&format!(
        r#"
        {{
            extern "C" {{
                #[link_name = "{unique_name}"]
                static STATIC: [u8; {len}];
            }}
            unsafe {{ &STATIC }}
        }}
    "#
    ))
    .unwrap()
}
