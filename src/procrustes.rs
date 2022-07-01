//! Implements [Procrustes analysis].
//!
//! [Procrustes analysis]: https://en.wikipedia.org/wiki/Procrustes_analysis

use nalgebra::{
    Const, Dynamic, Matrix, Matrix3, Matrix4, Rotation3, UnitQuaternion, VecStorage, Vector3,
};

use crate::iter::zip_exact;

/// Performs procrustes analysis.
///
/// This type is created with a set of reference points passed to [`ProcrustesAnalyzer::new`], and
/// can then compute the linear transformation needed to turn these reference points into a new set
/// of points passed to [`ProcrustesAnalyzer::analyze`].
#[derive(Clone)]
pub struct ProcrustesAnalyzer {
    /// Transform to apply to the reference data to remove its translation and scaling, yielding
    /// "base" data.
    ref_to_base_transform: Matrix4<f32>,

    buf: Vec<Vector3<f32>>,
    /// `Q` matrix for Kabsch algorithm, computed from reference points.
    q: Matrix<f32, Dynamic, Const<3>, VecStorage<f32, Dynamic, Const<3>>>,
    /// Transposed `P` matrix for Kabsch algorithm.
    p_t: Matrix<f32, Const<3>, Dynamic, VecStorage<f32, Const<3>, Dynamic>>,
}

impl ProcrustesAnalyzer {
    /// Creates a new procrustes analyzer that attempts to fit data points to `reference`.
    ///
    /// # Panics
    ///
    /// This panics if the `reference` iterator yields fewer than 2 points.
    pub fn new(reference: impl Iterator<Item = (f32, f32, f32)>) -> Self {
        let reference = reference
            .map(|(x, y, z)| Vector3::new(x, y, z))
            .collect::<Vec<_>>();

        Self::new_impl(reference)
    }

    fn new_impl(mut reference: Vec<Vector3<f32>>) -> Self {
        assert!(
            reference.len() > 1,
            "need at least 2 points for procrustes analysis"
        );

        let centroid = remove_translation(&mut reference);
        let scale = remove_scale(&mut reference);
        log::trace!("ref scale: {scale}, ref centroid: {centroid:?}");

        let ref_to_base_transform = Matrix4::identity()
            .append_translation(&-centroid)
            .append_scaling(1.0 / scale);

        let q = Matrix::from_fn_generic(Dynamic::new(reference.len()), Const, |row, col| {
            reference[row][col]
        });
        let p_t = Matrix::zeros_generic(Const, Dynamic::new(reference.len()));

        Self {
            ref_to_base_transform,
            buf: Vec::new(),
            q,
            p_t,
        }
    }

    /// Performs procrustes analysis on `points`.
    ///
    /// This computes the transformation to apply to the *reference* data to minimize the root-mean
    /// square distance to `points`.
    ///
    /// `points` must yield exactly as many elements as the reference data has, and all points need
    /// to be in the same order (so that the first point in `points` can be correlated to the first
    /// point of the reference data).
    ///
    /// # Limitations
    ///
    /// This function does not work in the presence of reflections. It is also unable to compute a
    /// non-uniform scaling applied to the data (uniform scaling works fine).
    ///
    /// # Panics
    ///
    /// This function will panic if `points` yields a different number of points than contained in
    /// the reference data passed to [`new`][Self::new].
    pub fn analyze(&mut self, points: impl Iterator<Item = (f32, f32, f32)>) -> AnalysisResult {
        self.buf.clear();
        self.buf
            .extend(points.map(|(x, y, z)| Vector3::new(x, y, z)));

        self.analyze_impl()
    }

    fn analyze_impl(&mut self) -> AnalysisResult {
        assert_eq!(
            self.buf.len(),
            self.q.shape().0,
            "`analyze` called on data of different length than the reference"
        );

        let centroid = remove_translation(&mut self.buf);
        let scale = remove_scale(&mut self.buf);
        let mut rotation = self.compute_rotation();

        log::trace!("data scale: {scale}, data centroid: {centroid:?}, rotation: {rotation:?}");

        // Scaling the data with 0.0 collapses it onto a plane, line, or point, in which case the
        // rotation cannot be recovered.
        if scale == 0.0 {
            rotation = Matrix3::identity();
        }

        // For assembling the result, we need to keep in mind that the reference data also had its
        // translation and scaling removed (but has not been rotated).
        let mut rot4 = Matrix4::identity();
        rot4.slice_mut((0, 0), (3, 3)).copy_from(&rotation);

        // NB: `A*B` does B, then A (must be some upstream bug in maths)
        let final_transform = Matrix4::identity()
            .append_scaling(scale)
            .append_translation(&centroid)
            * rot4
            * self.ref_to_base_transform;

        AnalysisResult {
            transform: final_transform,
            rotation: Rotation3::from_matrix(&rotation),
        }
    }

    fn compute_rotation(&mut self) -> Matrix3<f32> {
        for (mut col, pt) in zip_exact(self.p_t.column_iter_mut(), &self.buf) {
            col.x = pt.x;
            col.y = pt.y;
            col.z = pt.z;
        }

        let covariance = &self.p_t * &self.q;

        let svd = covariance.svd(true, true);

        // This does not match the algorithm on Wikipedia's "Kabsch Algorithm" page because that is
        // wrong.
        // I've also watched an MIT lecture according to which the result is just "u * v_t", which
        // is also wrong.
        // Apparently the truth lies somewhere inbetween. Deep.
        let v_t = svd.v_t.unwrap();
        let u = svd.u.unwrap();
        let d = (v_t * u).determinant().signum();

        let d_mat = Matrix3::from_diagonal(&[1.0, 1.0, d].into());

        u * d_mat * v_t
    }
}

/// Removes the uniform translation from `points` and returns it.
fn remove_translation(points: &mut [Vector3<f32>]) -> Vector3<f32> {
    let mut centroid = Vector3::new(0.0, 0.0, 0.0);
    for point in &*points {
        centroid += *point;
    }
    centroid /= points.len() as f32;
    for point in points {
        *point -= centroid;
    }
    centroid
}

/// Normalizes the scale of `points` and returns it.
///
/// Assumes that the centroid of `points` is at the origin.
///
/// Uses root mean square distance to the origin to determine the object's scale.
fn remove_scale(points: &mut [Vector3<f32>]) -> f32 {
    let mut scale = 0.0;
    for point in &*points {
        scale += point.x * point.x + point.y * point.y + point.z * point.z;
    }
    scale /= points.len() as f32;
    let scale = scale.sqrt();

    for point in points {
        *point = *point / scale;
    }

    scale
}

/// Result of procrustes analysis as returned by [`ProcrustesAnalyzer::analyze`].
#[derive(Debug, Clone, Copy)]
pub struct AnalysisResult {
    transform: Matrix4<f32>,
    rotation: Rotation3<f32>,
}

impl AnalysisResult {
    /// Returns the computed transformation matrix.
    #[inline]
    pub fn transformation_matrix(&self) -> Matrix4<f32> {
        self.transform
    }

    #[inline]
    pub fn rotation(&self) -> Rotation3<f32> {
        self.rotation
    }

    pub fn rotation_as_quaternion(&self) -> UnitQuaternion<f32> {
        UnitQuaternion::from_rotation_matrix(&self.rotation)
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use nalgebra::{Point3, Rotation3};

    use super::*;

    const RIGHT_ARROW: &[(f32, f32, f32)] = &[
        (-1.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (1.0, 1.5, 0.0),
        (1.75, 0.0, 0.0),
        (1.0, -1.5, 0.0),
        (1.0, -1.0, 0.0),
        (-1.0, -1.0, 0.0),
    ];

    const LOG: bool = false;
    const MAX_DELTA: f32 = 0.01;

    fn analyze(orig: &[(f32, f32, f32)], transform: Matrix4<f32>) -> AnalysisResult {
        if LOG {
            env_logger::builder()
                .filter_module(env!("CARGO_CRATE_NAME"), log::LevelFilter::Trace)
                .try_init()
                .ok();
        }

        let mut analysis = ProcrustesAnalyzer::new(orig.iter().copied());
        analysis.analyze(orig.iter().map(|&(x, y, z)| {
            let pt = transform.transform_point(&Point3::new(x, y, z));
            (pt.x, pt.y, pt.z)
        }))
    }

    /// Applies `transform` to `orig`, then applies procrustes analysis and checks if we get
    /// approximately `transform` back.
    fn test(orig: &[(f32, f32, f32)], transform: Matrix4<f32>) {
        let recovered_transform = analyze(orig, transform);
        let recovered_transform = recovered_transform.transformation_matrix();

        for (a, b) in zip_exact(transform.iter(), recovered_transform.iter()) {
            assert!(!a.is_nan(), "NaN in reference transform");
            assert!(!b.is_nan(), "NaN in recovered transform");

            if (a - b).abs() > MAX_DELTA {
                panic!(
                    "failed to recover transformation; original transform: {}, recovered transform: {}",
                    transform, recovered_transform
                );
            }
        }
    }

    #[test]
    fn test_identity() {
        test(RIGHT_ARROW, Matrix4::identity());

        let res = analyze(RIGHT_ARROW, Matrix4::identity());
        let rot = res.rotation();
        let rot_id = Rotation3::<f32>::identity();
        for (a, b) in zip_exact(rot.matrix().iter(), rot_id.matrix().iter()) {
            if (a - b).abs() > MAX_DELTA {
                panic!(
                    "failed to recover transformation; original transform: {}, recovered transform: {}",
                    rot.matrix(), rot_id.matrix()
                );
            }
        }

        let quat = res.rotation_as_quaternion();
        let quat_id = UnitQuaternion::identity();
        assert!(
            quat.angle_to(&quat_id) <= MAX_DELTA,
            "{}",
            quat.angle_to(&quat_id)
        );
    }

    #[test]
    fn test_translate() {
        // First, along the unrelated Z axis (all points are in the XY plane).
        test(
            RIGHT_ARROW,
            Matrix4::identity().append_translation(&Vector3::new(0.0, 0.0, 1.0)),
        );

        test(
            RIGHT_ARROW,
            Matrix4::identity().append_translation(&Vector3::new(0.0, -4.0, 0.0)),
        );

        test(
            RIGHT_ARROW,
            Matrix4::identity().append_translation(&Vector3::new(2.0, -4.0, -0.5)),
        );
    }

    #[test]
    fn test_uniform_scaling() {
        test(RIGHT_ARROW, Matrix4::identity().append_scaling(4.0));
        test(RIGHT_ARROW, Matrix4::identity().append_scaling(0.2));
        // Scaling by 0 collapses all points into the origin. This makes deriving any rotational
        // component impossible, but the rest of the algorithm should still work.
        test(RIGHT_ARROW, Matrix4::identity().append_scaling(0.0));
    }

    #[test]
    fn test_rotation() {
        test(
            RIGHT_ARROW,
            Rotation3::new(Vector3::new(0.0, 0.0, 1.0) * 2.0).into(),
        );
        test(
            RIGHT_ARROW,
            Rotation3::new(Vector3::new(0.0, 0.0, -1.0) * 2.0).into(),
        );
        test(
            RIGHT_ARROW,
            Rotation3::new(Vector3::new(0.5, 1.0, -1.0) * PI).into(),
        );
        test(
            RIGHT_ARROW,
            Rotation3::new(Vector3::new(0.5, 1.0, -1.0) * -PI).into(),
        );
    }

    #[test]
    fn test_combinations() {
        test(
            RIGHT_ARROW,
            Matrix4::from(Rotation3::new(Vector3::new(0.5, 1.0, -1.0) * PI))
                .append_translation(&Vector3::new(1.0, 0.5, 2.0))
                .append_scaling(2.0),
        );
        test(
            RIGHT_ARROW,
            Matrix4::from(Rotation3::new(Vector3::new(0.5, 1.0, -1.0) * PI))
                * Matrix4::identity()
                    .append_translation(&Vector3::new(1.0, 0.5, 2.0))
                    .append_scaling(2.0),
        );
        test(
            RIGHT_ARROW,
            Matrix4::from(Rotation3::new(Vector3::new(0.5, 1.0, -1.0) * PI))
                * Matrix4::identity()
                    .append_scaling(0.3)
                    .append_translation(&Vector3::new(-1.0, -0.5, 2.0))
                    .append_scaling(2.1),
        );
    }
}
