//! Perspective-N-Point solving.
//!
//! (experimental, might not work)

use nalgebra::{Const, Dynamic, Matrix, Matrix3x4, OMatrix, Rotation3, Vector3};

use crate::iter::zip_exact;

/// Intrinsic parameters of a pinhole camera.
#[derive(Debug, Clone, Copy)]
pub struct IntrinsicParams {
    focal_length: f32,
    pixel_size: [f32; 2],
    principal_point: [f32; 2],
}

impl IntrinsicParams {
    /// Creates a new set of intrinsic parameters for a pinhole camera.
    ///
    /// # Parameters
    ///
    /// - `focal_length`: the distance of the projection plane from the camera's pinhole.
    /// - `pixel_size`: the width and height of each pixel in world coordinates.
    pub fn new(focal_length: f32, pixel_size: [f32; 2]) -> Self {
        Self {
            focal_length,
            pixel_size,
            principal_point: [0.0, 0.0],
        }
    }

    /// Sets the camera's principal point.
    ///
    /// The principal point is the point on the projection plane where points on the Z axis get
    /// projected onto.
    ///
    /// By default, this point is the origin (`[0.0, 0.0]`).
    pub fn set_principal_point(&mut self, principal_point: [f32; 2]) {
        self.principal_point = principal_point;
    }

    /// Returns a 3x4 matrix containing the intrinsic parameters.
    ///
    /// A point (in 4D homogeneous coordinates) can be projected through the camera described by the
    /// [`IntrinsicParams`] by multiplying it with this matrix.
    #[rustfmt::skip]
    pub fn to_matrix(&self) -> Matrix3x4<f32> {
        let ax = self.focal_length / self.pixel_size[0];
        let ay = self.focal_length / self.pixel_size[1];
        let [u0, v0] = self.principal_point;
        Matrix3x4::new(
            ax, 0.0, u0, 0.0,
            0.0, ay, v0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        )
    }
}

/// A Direct Linear Transform (DLT) solver for the Perspective-N-Point problem.
///
/// DLT computes both the intrinsic and extrinsic parameters of the camera, from a minimum of 6
/// 3D-to-2D point correspondences.
pub struct Dlt {
    reference: Vec<[f32; 3]>,
    matrix: OMatrix<f32, Dynamic, Const<12>>,
}

impl Dlt {
    pub fn new(reference: impl Iterator<Item = [f32; 3]>) -> Self {
        let reference = reference.collect::<Vec<_>>();
        assert!(
            reference.len() >= 6,
            "DLT needs at least 6 point correspondences"
        );

        let matrix = Matrix::zeros_generic(Dynamic::new(reference.len() * 2), Const);

        Self { reference, matrix }
    }

    pub fn solve(&mut self, projected: impl ExactSizeIterator<Item = [f32; 2]>) -> DltOutput {
        // https://files.ifi.uzh.ch/rpg/teaching/2016/03_image_formation_2.pdf

        for (i, (&[x, y, z], [u, v])) in zip_exact(&self.reference, projected).enumerate() {
            let mut rows = self.matrix.rows_mut(i * 2, 2);

            let mut row = rows.row_mut(0);
            row[0] = x;
            row[1] = y;
            row[2] = z;
            row[3] = 1.0;
            row[4] = 0.0;
            row[5] = 0.0;
            row[6] = 0.0;
            row[7] = 0.0;
            row[8] = -u * x;
            row[9] = -u * y;
            row[10] = -u * z;
            row[11] = -u;

            let mut row = rows.row_mut(1);
            row[0] = 0.0;
            row[1] = 0.0;
            row[2] = 0.0;
            row[3] = 0.0;
            row[4] = x;
            row[5] = y;
            row[6] = z;
            row[7] = 1.0;
            row[8] = -v * x;
            row[9] = -v * y;
            row[10] = -v * z;
            row[11] = -v;
        }

        log::trace!("self.matrix={}", self.matrix);
        let svd = self.matrix.slice_range(.., ..).svd(false, true);
        log::trace!("v_t={}", svd.v_t.as_ref().unwrap());
        log::trace!("sigma={}", svd.singular_values);
        let p = svd.v_t.as_ref().unwrap().row(11);
        let p = Matrix3x4::new(
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11],
        );
        log::trace!("p={p}");

        // Extract the rotation.
        let svd = p.fixed_columns::<3>(0).svd(true, true);
        let rot = svd.u.unwrap() * svd.v_t.unwrap();
        // Flip the sign if the determinant is negative; removes mirroring from the rotation matrix.
        let rot = rot.determinant().signum() * rot;
        assert!(
            rot.is_special_orthogonal(0.001),
            "not special orthogonal; det={}, rot={rot}",
            rot.determinant()
        );
        let rot = Rotation3::from_matrix_unchecked(rot);
        log::trace!("rot={rot}");

        let t = p.column(3).into_owned() / svd.singular_values[0];
        log::trace!("t={t}");

        DltOutput {
            rotation: rot,
            translation: t,
        }
    }
}

/// The type returned by [`Dlt::solve`].
pub struct DltOutput {
    rotation: Rotation3<f32>,
    translation: Vector3<f32>,
}

impl DltOutput {
    /// Returns the recovered rotation of the camera.
    #[inline]
    pub fn rotation(&self) -> &Rotation3<f32> {
        &self.rotation
    }

    /// Returns the recovered translation of the camera.
    #[inline]
    pub fn translation(&self) -> Vector3<f32> {
        self.translation
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::hash_map::DefaultHasher,
        hash::{Hash, Hasher},
        thread,
    };

    use approx::assert_relative_eq;
    use nalgebra::{Matrix4, Vector4};

    use super::*;

    const POS_EPSILON: f32 = 0.01;
    const ANGLE_EPSILON: f32 = 0.05;

    fn project(intrinsic: &Matrix3x4<f32>, p: Vector3<f32>) -> [f32; 2] {
        let p = intrinsic * p.to_homogeneous();
        [p.x / p.z, p.y / p.z]
    }

    fn check(
        n: usize,
        tf: Matrix4<f32>,
        translation: Vector3<f32>,
        roll: f32,
        pitch: f32,
        yaw: f32,
    ) {
        let mut hasher = DefaultHasher::new();
        thread::current().name().unwrap().hash(&mut hasher);
        let seed = hasher.finish();
        let rng = fastrand::Rng::with_seed(seed);

        let gen = || rng.f32() - 0.5;
        let points = std::iter::from_fn(|| Some([gen(), gen(), gen()]))
            .take(n)
            .collect::<Vec<_>>();
        let intrinsic = IntrinsicParams::new(3.0, [0.5, 0.5]).to_matrix();
        let mut dlt = Dlt::new(points.iter().copied());
        let out = dlt.solve(points.iter().map(|&[x, y, z]| {
            let p = tf * Vector4::new(x, y, z, 1.0);
            project(&intrinsic, (p / p.w).xyz())
        }));
        assert_relative_eq!(out.translation(), translation, epsilon = POS_EPSILON);
        let (r, p, y) = out.rotation().euler_angles();
        assert_relative_eq!(roll, r, epsilon = ANGLE_EPSILON);
        assert_relative_eq!(pitch, p, epsilon = ANGLE_EPSILON);
        assert_relative_eq!(yaw, y, epsilon = ANGLE_EPSILON);
    }

    #[test]
    fn test_identity() {
        check(6, Matrix4::identity(), Vector3::zeros(), 0.0, 0.0, 0.0);
        check(7, Matrix4::identity(), Vector3::zeros(), 0.0, 0.0, 0.0);
        check(60, Matrix4::identity(), Vector3::zeros(), 0.0, 0.0, 0.0);
    }

    #[test]
    fn test_translate() {
        // FIXME: translation along Z is not recovered correctly
        check(
            6,
            Matrix4::new_translation(&Vector3::new(1.0, 5.0, 0.0)),
            -Vector3::new(1.0, 5.0, 0.0),
            0.0,
            0.0,
            0.0,
        );
    }

    #[test]
    fn test_rotate() {
        check(
            6,
            Rotation3::from_euler_angles(45.0f32.to_radians(), 0.0, 0.0).into(),
            Vector3::zeros(),
            45.0f32.to_radians(),
            0.0,
            0.0,
        );
        check(
            60,
            Rotation3::from_euler_angles(45.0f32.to_radians(), 0.0, 0.0).into(),
            Vector3::zeros(),
            45.0f32.to_radians(),
            0.0,
            0.0,
        );

        check(
            6,
            Rotation3::from_euler_angles(90.0f32.to_radians(), 0.0, 0.0).into(),
            Vector3::zeros(),
            90.0f32.to_radians(),
            0.0,
            0.0,
        );
        check(
            6,
            Rotation3::from_euler_angles(90.0f32.to_radians(), 45.0f32.to_radians(), 0.0).into(),
            Vector3::zeros(),
            90.0f32.to_radians(),
            45.0f32.to_radians(),
            0.0,
        );
    }
}
