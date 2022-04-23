use mizaru::{
    detector::Detector,
    gui, image,
    iter::zip_exact,
    nn::{Cnn, CnnInputShape, NeuralNetwork},
    num::TotalF32,
    webcam::Webcam,
};

const THRESHOLD: f32 = 0.33;

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
    let webcam = Webcam::open()?;
    let mut det = Detector::new();

    let nn = NeuralNetwork::load("3rdparty/onnx/mobilefacenet.onnx")?;
    for inp in nn.inputs() {
        println!("{inp:?}");
    }
    let mut cnn = Cnn::new(nn, CnnInputShape::NCHW)?;
    cnn.set_color_map(|c| {
        // Due to a bug, the recognition network has been trained on image data that has been mapped
        // from sRGB to [-1.0, 1.0] tensor inputs *twice*, so we have to do the same here, otherwise
        // the embeddings vary wildly across very similar frames.
        let first = (f32::from(c) - 127.5) / 128.0;
        let second = (first - 127.5) / 128.0;
        second
    });

    let mut known_ppl = Vec::new();
    for image in webcam {
        let image = image?;
        let det = det.detect(&image);

        let rect = det[0]
            .bounding_rect_raw()
            .grow_to_fit_aspect(cnn.input_resolution().aspect_ratio());
        let mut input = image
            .view(&rect)
            .aspect_aware_resize(cnn.input_resolution());
        let outputs = cnn.infer(&input)?;
        let raw = outputs[0].as_slice::<f32>().unwrap();
        assert_eq!(raw.len(), 128);
        let emb = Embedding {
            raw: raw.try_into().unwrap(),
        };

        match known_ppl
            .iter()
            .enumerate()
            .map(|(i, known)| {
                let difference = emb.difference(known);
                println!("{}", difference);
                (i, TotalF32(difference))
            })
            .filter(|(_, val)| val.0 < THRESHOLD)
            .min_by_key(|(_, val)| *val)
        {
            Some((idx, difference)) => {
                let x = input.width() / 2;
                let y = input.height() / 2;
                image::draw_text(
                    &mut input,
                    x as _,
                    y as _,
                    &format!("{} @ {}", idx, difference.0),
                );
            }
            None => {
                known_ppl.push(emb);
            }
        }

        gui::show_image("face", &input);
    }

    Ok(())
}
