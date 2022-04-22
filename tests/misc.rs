use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
};

#[test]
fn thirdparty_attribution() {
    let thirdparty_dir = "3rdparty";

    let file = std::fs::read_to_string(format!("{}/README.md", thirdparty_dir)).unwrap();

    let mut present = HashSet::new();
    let mut in_table = false;
    for line in file.lines() {
        if in_table {
            if line.starts_with('|') {
                let rest = &line[line.find('`').unwrap() + 1..];
                let path = &rest[..rest.find('`').unwrap()];
                present.insert(path);
            } else {
                break;
            }
        } else {
            if line.starts_with("|---") {
                in_table = true;
            }
        }
    }

    let files = walkdir(thirdparty_dir.as_ref());
    assert!(!files.is_empty());
    for path in files {
        let path = path.to_str().unwrap();
        assert!(path.starts_with(thirdparty_dir));
        let path = &path[thirdparty_dir.len() + 1..];
        if path.ends_with("README.md") {
            continue;
        }

        if !present.remove(path) {
            panic!("3rdparty file `{}` is not attributed in readme", path);
        }
    }

    assert!(
        present.is_empty(),
        "3rdparty readme contains non-existent files: {:?}",
        present
    );
}

fn walkdir(dir: &Path) -> Vec<PathBuf> {
    let mut vec = Vec::new();
    for entry in fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();

        if entry.file_name().into_string().unwrap().starts_with('.') {
            continue;
        }

        if entry.file_type().unwrap().is_dir() {
            vec.extend(walkdir(entry.path().as_path()));
        } else {
            vec.push(entry.path());
        }
    }
    vec
}
