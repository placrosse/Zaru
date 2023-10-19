//! Implements [Procrustes analysis].
//!
//! [Procrustes analysis]: https://en.wikipedia.org/wiki/Procrustes_analysis

use nalgebra::{Const, Dyn, Matrix, Matrix3, Matrix4, OMatrix, Rotation3, UnitQuaternion, Vector3};
use zaru_linalg::{vec3, Vec3f};

use crate::iter::zip_exact;

/// Performs procrustes analysis.
///
/// This type is created with a set of reference points passed to [`ProcrustesAnalyzer::new`], and
/// can then compute the linear transformation needed to turn these reference points into a new set
/// of points passed to [`ProcrustesAnalyzer::analyze`].
#[derive(Clone)]
pub struct ProcrustesAnalyzer {
    ref_centroid: Vector3<f32>,
    ref_scale: f32,

    buf: Vec<Vector3<f32>>,
    /// `Q` matrix for Kabsch algorithm, computed from reference points.
    q: OMatrix<f32, Dyn, Const<3>>,
    /// Transposed `P` matrix for Kabsch algorithm.
    p_t: OMatrix<f32, Const<3>, Dyn>,
}

impl ProcrustesAnalyzer {
    /// Creates a new procrustes analyzer that attempts to fit data points to `reference`.
    ///
    /// # Panics
    ///
    /// This panics if the `reference` iterator yields fewer than 2 points.
    pub fn new(reference: impl Iterator<Item = Vec3f>) -> Self {
        let reference = reference
            .map(|v| Vector3::new(v.x, v.y, v.z))
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

        let q =
            Matrix::from_fn_generic(Dyn(reference.len()), Const, |row, col| reference[row][col]);
        let p_t = Matrix::zeros_generic(Const, Dyn(reference.len()));

        Self {
            ref_centroid: centroid,
            ref_scale: scale,
            buf: Vec::new(),
            q,
            p_t,
        }
    }

    /// Returns the centroid of the reference data.
    #[inline]
    pub fn reference_centroid(&self) -> Vector3<f32> {
        self.ref_centroid
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
    pub fn analyze(&mut self, points: impl Iterator<Item = Vec3f>) -> AnalysisResult {
        self.buf.clear();
        self.buf.extend(points.map(|v| Vector3::new(v.x, v.y, v.z)));

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

        // Scaling the data with 0.0 collapses it onto a point, in which case the rotation cannot be
        // recovered.
        if scale == 0.0 {
            rotation = Matrix3::identity();
        }

        debug_assert!(rotation.is_special_orthogonal(0.001));

        let rotation =
            UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(rotation));

        // Scale of the analyzed data compared to the reference data.
        let scale = scale / self.ref_scale;

        // Translation from reference data to analyzed data.
        // Since the centroid of the reference data can be offset from the origin, it can be moved
        // around by rotation and scaling, so those have to be applied to the centroids to get the
        // "real" translation.
        let centroid_offset = rotation * self.ref_centroid * scale;
        let translation = centroid - centroid_offset;

        AnalysisResult {
            centroid,
            ref_centroid: self.ref_centroid,
            translation,
            scale,
            rotation,
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
        *point /= scale;
    }

    scale
}

/// Result of procrustes analysis as returned by [`ProcrustesAnalyzer::analyze`].
#[derive(Debug, Clone, Copy)]
pub struct AnalysisResult {
    /// Centroid of the analyzed data. Rotation and scaling happens around this point.
    centroid: Vector3<f32>,
    ref_centroid: Vector3<f32>,
    translation: Vector3<f32>,
    scale: f32,
    rotation: UnitQuaternion<f32>,
}

impl AnalysisResult {
    /// Returns the centroid of the analyzed data.
    ///
    /// Scaling and rotation happens around this point.
    #[inline]
    pub fn centroid(&self) -> Vec3f {
        vec3(self.centroid.x, self.centroid.y, self.centroid.z)
    }

    /// Returns the translation that was applied to the analyzed data, relative to the reference
    /// data.
    ///
    /// Unlike [`AnalysisResult::centroid`], this translation value is computed while taking
    /// rotation and scaling already into account. That means, if a rotation or scaling offsets the
    /// centroid of the reference data, that offset *will* be reflected in the centroid, but *not*
    /// in the translation returned by this method. Only true translation is returned.
    #[inline]
    pub fn translation(&self) -> Vec3f {
        vec3(self.translation.x, self.translation.y, self.translation.z)
    }

    /// Returns the computed rotation as a unit quaternion.
    ///
    /// This describes how the reference data was rotated around its centroid to reach the analyzed
    /// data.
    #[inline]
    pub fn rotation(&self) -> UnitQuaternion<f32> {
        self.rotation
    }

    /// Returns the uniform scaling of the analyzed data compared to the reference data.
    #[inline]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Computes the recovered transformation that was applied to the reference data.
    pub fn transform(&self) -> Matrix4<f32> {
        // Move reference data to the origin, rotate and scale it, then move it to `self.centroid`.
        let ref_to_origin = Matrix4::new_translation(&-self.ref_centroid);
        let rot = self.rotation.to_homogeneous();
        Matrix4::new_translation(&self.centroid) * rot * ref_to_origin.append_scaling(self.scale)
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use approx::assert_relative_eq;
    use nalgebra::{Point3, Rotation3};
    use once_cell::sync::Lazy;
    use zaru_linalg::{assert_approx_eq, vec3, Vec3};

    use super::*;

    const REFERENCE_POINTS: &[Vec3f] = &[
        vec3(-1.0, 1.0, 0.0),
        vec3(1.0, 1.0, 0.0),
        vec3(1.0, 1.5, 0.0),
        vec3(1.75, 0.0, 0.0),
        vec3(1.0, -1.5, 0.0),
        vec3(1.0, -1.0, 0.0),
        vec3(-1.0, -1.0, 0.0),
        vec3(1.0, 1.0, 5.0),
    ];
    static ANALYZER: Lazy<ProcrustesAnalyzer> =
        Lazy::new(|| ProcrustesAnalyzer::new(REFERENCE_POINTS.iter().copied()));

    const LOG: bool = false;
    const MAX_DELTA: f32 = 0.00001;
    const MAX_TRANSLATION_DELTA: f32 = 0.001;

    fn analyze(transform: Matrix4<f32>) -> AnalysisResult {
        if LOG {
            env_logger::builder()
                .filter_module(env!("CARGO_CRATE_NAME"), log::LevelFilter::Trace)
                .try_init()
                .ok();
        }

        ANALYZER.clone().analyze(REFERENCE_POINTS.iter().map(|&v| {
            let pt = transform.transform_point(&Point3::new(v.x, v.y, v.z));
            vec3(pt.x, pt.y, pt.z)
        }))
    }

    /// Applies `transform` to `orig`, then applies procrustes analysis and checks if we get
    /// approximately `transform` back.
    fn test_recover_transform(transform: Matrix4<f32>) {
        let recovered_transform = analyze(transform);
        let recovered_transform = recovered_transform.transform();

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

    fn test_z_rotation(scaling: f32, translation: Vector3<f32>) {
        fn clamp_degrees(degrees: i32) -> i32 {
            ((degrees + 180) % 360) - 180
        }

        let range = -180..=360;
        let expected = range.clone().map(clamp_degrees).collect::<Vec<_>>();
        let recovered = range
            .clone()
            .map(|deg| {
                let angle = (deg as f32).to_radians();
                let result = analyze(
                    Matrix4::new_translation(&translation).prepend_scaling(scaling)
                        * Matrix4::from(Rotation3::new(Vector3::new(0.0, 0.0, 1.0) * angle)),
                );

                let translation = vec3(translation.x, translation.y, translation.z);
                assert_approx_eq!(result.scale(), scaling).rel(MAX_DELTA);
                assert_approx_eq!(result.translation(), translation).abs(MAX_TRANSLATION_DELTA);

                let (roll, pitch, yaw) = result.rotation().euler_angles();
                assert_approx_eq!(roll, 0.0).abs(MAX_DELTA);
                assert_approx_eq!(pitch, 0.0).abs(MAX_DELTA);
                clamp_degrees(yaw.to_degrees().round() as i32)
            })
            .collect::<Vec<_>>();

        assert_eq!(expected, recovered);
    }

    #[test]
    fn test_identity() {
        test_recover_transform(Matrix4::identity());

        let res = analyze(Matrix4::identity());

        let quat = res.rotation();
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
        test_recover_transform(
            Matrix4::identity().append_translation(&Vector3::new(0.0, 0.0, 1.0)),
        );

        test_recover_transform(
            Matrix4::identity().append_translation(&Vector3::new(0.0, -4.0, 0.0)),
        );

        test_recover_transform(
            Matrix4::identity().append_translation(&Vector3::new(2.0, -4.0, -0.5)),
        );
    }

    #[test]
    fn test_uniform_scaling() {
        test_recover_transform(Matrix4::identity().append_scaling(4.0));
        test_recover_transform(Matrix4::identity().append_scaling(0.2));
        // Scaling by 0 collapses all points into the origin. This makes deriving any rotational
        // component impossible, but the rest of the algorithm should still work.
        test_recover_transform(Matrix4::identity().append_scaling(0.0));

        let res = analyze(Matrix4::identity().append_scaling(2.0));
        assert_eq!(res.scale(), 2.0);
        assert_relative_eq!(
            res.rotation(),
            UnitQuaternion::identity(),
            epsilon = MAX_DELTA
        );
        assert_approx_eq!(res.translation(), Vec3::ZERO).abs(MAX_DELTA);

        let res = analyze(Matrix4::identity().append_scaling(0.5));
        assert_eq!(res.scale(), 0.5);
        assert_relative_eq!(
            res.rotation(),
            UnitQuaternion::identity(),
            epsilon = MAX_DELTA
        );
        assert_approx_eq!(res.translation(), Vec3::ZERO).abs(MAX_DELTA);
    }

    #[test]
    fn test_rotation() {
        test_recover_transform(Rotation3::new(Vector3::new(0.0, 0.0, 1.0) * 2.0).into());
        test_recover_transform(Rotation3::new(Vector3::new(0.0, 0.0, -1.0) * 2.0).into());
        test_recover_transform(Rotation3::new(Vector3::new(0.5, 1.0, -1.0) * PI).into());
        test_recover_transform(Rotation3::new(Vector3::new(0.5, 1.0, -1.0) * -PI).into());
    }

    #[test]
    fn test_combinations() {
        test_recover_transform(
            Matrix4::from(Rotation3::new(Vector3::new(0.5, 1.0, -1.0) * PI))
                .append_translation(&Vector3::new(1.0, 0.5, 2.0))
                .append_scaling(2.0),
        );
        test_recover_transform(
            Matrix4::from(Rotation3::new(Vector3::new(0.5, 1.0, -1.0) * PI))
                * Matrix4::identity()
                    .append_translation(&Vector3::new(1.0, 0.5, 2.0))
                    .append_scaling(2.0),
        );
        test_recover_transform(
            Matrix4::from(Rotation3::new(Vector3::new(0.5, 1.0, -1.0) * PI))
                * Matrix4::identity()
                    .append_scaling(0.3)
                    .append_translation(&Vector3::new(-1.0, -0.5, 2.0))
                    .append_scaling(2.1),
        );
        test_recover_transform(
            Matrix4::from(Rotation3::new(Vector3::new(0.5, 1.0, -1.0) * 2.3))
                * Matrix4::identity()
                    .append_scaling(0.3)
                    .append_translation(&Vector3::new(-1.0, -0.5, 2.0))
                    .append_scaling(2.1),
        );
    }

    #[test]
    fn test_rot_around_z() {
        test_z_rotation(1.0, Vector3::zeros());
        test_z_rotation(2.0, Vector3::zeros());
        test_z_rotation(0.5, Vector3::zeros());

        test_z_rotation(1.0, Vector3::new(10.0, 0.0, 0.0));
        test_z_rotation(1.0, Vector3::new(-10.0, 7.0, 3.0));

        test_z_rotation(0.5, Vector3::new(10.0, 0.0, 0.0));
        test_z_rotation(2.0, Vector3::new(-10.0, 7.0, 3.0));
    }

    #[test]
    fn large_scale_and_move() {
        test_z_rotation(500.0, Vector3::new(0.5, 300.0, -1.0));
    }

    #[test]
    fn jitter() {
        let (expected_roll, expected_pitch, expected_yaw) = (3.0, -1.0, 2.0);
        let rot = UnitQuaternion::from_euler_angles(expected_roll, expected_pitch, expected_yaw);
        let offset = vec3(50.0, 200.0, -20.0);
        let mut rng = fastrand::Rng::with_seed(0x3024b6663d843ca2);
        let res = ANALYZER.clone().analyze(REFERENCE_POINTS.iter().map(|&v| {
            let rotated = rot * Vector3::new(v.x, v.y, v.z);
            vec3(
                rotated.x * 100.0 + offset.x + rng.f32() - 0.5,
                rotated.y * 100.0 + offset.y + rng.f32() - 0.5,
                rotated.z * 100.0 + offset.z + rng.f32() - 0.5,
            )
        }));
        assert_approx_eq!(res.scale(), 100.0).abs(0.1);
        assert_approx_eq!(res.translation(), offset).abs(0.5);
        let (roll, pitch, yaw) = res.rotation().euler_angles();
        assert_approx_eq!(roll.to_degrees(), expected_roll.to_degrees()).abs(0.2);
        assert_approx_eq!(pitch.to_degrees(), expected_pitch.to_degrees(),).abs(0.2);
        assert_approx_eq!(yaw.to_degrees(), expected_yaw.to_degrees()).abs(0.2);
    }
}
