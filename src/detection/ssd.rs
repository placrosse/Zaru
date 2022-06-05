//! Anchor/Prior generation for [Single Shot MultiBox Detectors] (SSDs).
//!
//! Note that the implementation in this module is extremely limited and is only meant to work for
//! our specific networks, not more general networks.
//!
//! [Single Shot MultiBox Detectors]: https://arxiv.org/abs/1512.02325

use std::ops::Index;

use crate::resolution::Resolution;

/// An anchor of an SSD network.
pub struct Anchor {
    // values range from 0 to 1
    x_center: f32,
    y_center: f32,
}

impl Anchor {
    pub fn x_center(&self) -> f32 {
        self.x_center
    }

    pub fn y_center(&self) -> f32 {
        self.y_center
    }
}

/// Describes an output layer of an SSD network.
pub struct LayerInfo {
    /// Number of anchors per feature map cell/pixel. Must be non-zero.
    boxes_per_cell: u32,
    /// Feature map resolution of this layer.
    resolution: Resolution,
}

impl LayerInfo {
    /// Creates a new SSD layer description.
    ///
    /// # Parameters
    ///
    /// - `boxes_per_cell`: the number of boxes associated with each cell in this feature map.
    /// - `width`/`height`: size of this layer's feature map, in output cells.
    pub fn new(boxes_per_cell: u32, width: u32, height: u32) -> Self {
        assert_ne!(boxes_per_cell, 0);
        Self {
            boxes_per_cell,
            resolution: Resolution::new(width, height),
        }
    }
}

pub struct AnchorParams<'a> {
    /// List of output layers.
    ///
    /// The easiest way to figure out the right values is to use a tool like [Netron] to visualize
    /// the network graph, and look at how the *confidence tensor* (not the actual box data) is
    /// composed. Note that `Concat` nodes typically have their inputs displayed in reverse order,
    /// and `Transpose`/`Reshape` nodes might be inserted before the output. The layer information
    /// can be derived from the tensor shape *before* it goes through these nodes.
    ///
    /// Example: `face_detection_short_range` has a `1×6×8×8` and a `1×2×16×16` tensor produced by
    /// `Conv` nodes, which then go through `Transpose`, `Reshape`, and `Concat` before being output
    /// as confidence values. Since the `Concat` order is reversed, the `1×2×16×16` data comes
    /// first, then the `1×6×8×8` data. Therefore, the anchors can be computed from two
    /// [`LayerInfo`]s, the first with 2 boxes per cell and 16x16 cells, the second with 6 boxes per
    /// cell and 8x8 cells.
    ///
    /// [Netron]: https://netron.app/
    pub layers: &'a [LayerInfo],
}

pub struct Anchors {
    anchors: Vec<Anchor>,
}

impl Anchors {
    pub fn calculate(params: &AnchorParams<'_>) -> Self {
        let mut anchors = Vec::new();

        for layer in params.layers {
            let height = layer.resolution.height();
            let width = layer.resolution.width();

            for y in 0..height {
                for x in 0..width {
                    // FIXME `boxes_per_cell` is ignored, what should it do?
                    for _ in 0..layer.boxes_per_cell {
                        let x_center = (x as f32 + 0.5) / width as f32;
                        let y_center = (y as f32 + 0.5) / height as f32;

                        anchors.push(Anchor { x_center, y_center });
                    }
                }
            }
        }

        Self { anchors }
    }

    /// Returns the total number of SSD anchors/priors.
    pub fn anchor_count(&self) -> usize {
        self.anchors.len()
    }
}

impl Index<usize> for Anchors {
    type Output = Anchor;

    fn index(&self, index: usize) -> &Anchor {
        &self.anchors[index]
    }
}
