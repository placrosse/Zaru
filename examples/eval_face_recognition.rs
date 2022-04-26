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

use std::{collections::HashMap, fs, time::Instant};

use itertools::Itertools;
use mizaru::{
    detector::Detector,
    image::Image,
    iter::zip_exact,
    nn::{Cnn, CnnInputShape, NeuralNetwork},
    num::TotalF32,
};
use nalgebra::RealField;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

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

fn main() -> Result<(), mizaru::Error> {
    let face_dir = std::env::args_os().skip(1).next().unwrap();

    let nn = NeuralNetwork::load("3rdparty/onnx/mobilefacenet.onnx")?;
    let cnn = Cnn::new(nn, CnnInputShape::NCHW)?;
    let target_aspect = cnn.input_resolution().aspect_ratio();

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
        .filter_map(|(path, class)| {
            let image = Image::load(path).unwrap();

            // FIXME: `Detector` filters between frames which we don't want here. It should probably not do that.
            let mut det = Detector::new();
            let dets = det.detect(&image);
            if dets.is_empty() {
                println!("No faces detected in '{}'", path.display());
                return None;
            }
            let grow_by = 0.4;
            let rect = dets[0]
                .bounding_rect_raw()
                .grow_rel(grow_by, grow_by, grow_by, grow_by)
                .grow_to_fit_aspect(target_aspect);
            let face = image
                .view(&rect)
                .aspect_aware_resize(cnn.input_resolution());
            let out = cnn.infer(&face).unwrap();
            let f = out[0].as_slice::<f32>().unwrap();
            let emb = Embedding {
                raw: f.try_into().unwrap(),
            };
            Some((emb, *class))
        })
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
            max_diff = max_diff.max(a.difference(*b));
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