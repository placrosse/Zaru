//! Face recognition testbed and evaluation.
//!
//! To be added to the main library once I understand it better.
//!
//! With uncorrected images:
//! ```text
//! diff=26.585608 in /home/sludge/Downloads/lfw/Hugo_Chavez
//! diff=24.186064 in /home/sludge/Downloads/lfw/George_W_Bush
//! diff=24.090973 in /home/sludge/Downloads/lfw/Yoriko_Kawaguchi
//! diff=23.725494 in /home/sludge/Downloads/lfw/Gerhard_Schroeder
//! ```

use std::{collections::HashMap, convert::identity, fs, time::Instant};

use itertools::Itertools;
use nalgebra::RealField;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use zaru::{
    detection::Detector,
    face::detection::ShortRangeNetwork,
    image::Image,
    iter::zip_exact,
    nn::{create_linear_color_mapper, Cnn, CnnInputShape, NeuralNetwork},
    num::TotalF32,
};

pub struct Embedding {
    raw: [f32; 128],
}

impl Embedding {
    pub fn difference(&self, other: &Self) -> f32 {
        let mut diff = [0.0; 128];
        for (diff, (a, b)) in zip_exact(
            diff.iter_mut(),
            zip_exact(self.raw.iter(), other.raw.iter()),
        ) {
            *diff = *a - *b;
        }

        let norm = diff.iter().map(|&d| d * d).sum::<f32>();
        norm.sqrt()
    }
}

fn main() -> anyhow::Result<()> {
    let face_dir = std::env::args_os().nth(1).unwrap();

    let nn = NeuralNetwork::from_path("3rdparty/onnx/mobilefacenet.onnx")?.load()?;
    let cnn = Cnn::new(
        nn,
        CnnInputShape::NCHW,
        create_linear_color_mapper(-1.0..=1.0),
    )?;
    let target_aspect = cnn.input_resolution().aspect_ratio().unwrap();

    let mut classes = Vec::new();
    let mut image_paths = Vec::new();
    for subdir in fs::read_dir(&face_dir)? {
        let subdir = subdir?;
        classes.push(subdir.path());
        for file in fs::read_dir(subdir.path())? {
            image_paths.push((file?.path(), classes.len() - 1));
        }
    }

    println!("Found {} image classes", classes.len());
    println!("Total: {} files", image_paths.len());

    let start = Instant::now();
    let embeddings = image_paths
        .par_iter()
        .map_init(
            || Detector::new(ShortRangeNetwork),
            |det, (path, class)| {
                let image = Image::load(path).unwrap();

                let dets = det.detect(&image);
                let rect = match dets.iter().next() {
                    Some(det) => det.bounding_rect(),
                    None => {
                        println!("No faces detected in '{}'", path.display());
                        return None;
                    }
                };
                let grow_by = 0.4;
                let rect = rect.grow_rel(grow_by).grow_to_fit_aspect(target_aspect);
                let face = image.view(rect);
                let out = cnn.estimate(&face).unwrap();
                let view = out[0].index([0]);
                let f = view.as_slice();
                let emb = Embedding {
                    raw: f.try_into().unwrap(),
                };
                Some((emb, *class))
            },
        )
        .filter_map(identity)
        .collect::<Vec<_>>();

    println!(
        "Computed {} embeddings in {:?}",
        embeddings.len(),
        start.elapsed(),
    );

    let mut class_map: HashMap<_, Vec<_>> = HashMap::with_capacity(classes.len());
    for (emb, class) in &embeddings {
        class_map
            .entry(*class)
            .or_insert_with(|| Vec::with_capacity(1))
            .push(emb);
    }

    let mut max_intra_class_difference = Vec::new();
    for (class, embeddings) in &class_map {
        if embeddings.len() == 1 {
            continue;
        }

        let mut max_diff = 0.0;
        for (a, b) in embeddings.iter().tuple_combinations() {
            max_diff = max_diff.max(a.difference(b));
        }

        max_intra_class_difference.push((*class, max_diff));
    }

    println!("Max. intra-class differences:");
    max_intra_class_difference.sort_by_key(|(_, diff)| TotalF32(-*diff));
    for (class, diff) in max_intra_class_difference.iter().take(10) {
        println!("diff={} in {}", diff, classes[*class].display());
    }

    Ok(())
}
