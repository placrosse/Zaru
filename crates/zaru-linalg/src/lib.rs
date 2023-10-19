//! A simple linear algebra library for Zaru.
//!
//! # Motivation
//!
//! The Zaru computer perception libraries sometimes need to expose linear algebra types in their
//! public APIs. This library was created to accomodate that use case.
//!
//! Existing Rust libraries have problems and limitations that make them unsuitable for this use
//! case:
//!
//! - Some of them aim for maximum flexibility, and pay the complexity cost associated with that.
//!   Exposing types from such a library in the public Zaru API's makes Zaru unnecessarily
//!   difficult to use (ease of use is one of Zaru's design goals).
//! - Many libraries still see many breaking changes. Exposing types from such a library in public
//!   APIs would cause unnecessary churn for dependants.
//! - Some libraries are designed exclusively for computer graphics applications. While Zaru does
//!   and/or will use the GPU, it is primarily a computer *vision* library, and as such might have
//!   requirements that are out of scope for a computer graphics focused library.
//!
//! # Goals & Non-Goals
//!
//! - Don't support dynamically-sized vectors and matrices for now. The API can be significantly
//!   simplified by relying on const generics to specify vector and matrix dimensions. If
//!   dynamically-sized objects are needed in the future, they will be added as separate types.
//! - Support only a single, column-major, unpadded data layout for matrices and vectors, further
//!   simplifying their API.
//! - Be generic over the element type, but don't try to support non-[`Copy`] numeric types (eg.
//!   "big decimals").
//! - Don't have any unstable public dependencies. "Unstable" includes everything pre-1.0, as well
//!   as libraries that violate semver, as well as libraries that regularly do breaking post-1.0
//!   releases.
//! - Put at least some effort into designing an ergonomic API that adheres to the
//!   [Rust API Guidelines].
//!
//! [Rust API Guidelines]: https://rust-lang.github.io/api-guidelines/

pub mod approx;
mod matrix;
mod traits;
mod vector;

pub use matrix::*;
pub use traits::*;
pub use vector::*;
