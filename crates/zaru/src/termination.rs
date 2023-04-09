//! Defines the [`Termination`] trait.

use std::{convert::Infallible, fmt::Debug, process};

/// This trait extends the [`std::process::Termination`] trait for use in Zaru.
///
/// The purpose of this trait is to allow Zaru to introspect the termination status. This is
/// required because not all platforms allow returning from the event loop handler, so Zaru will
/// exit the process itself depending on the [`Termination`] value returned by the user code.
///
/// Eventually, this trait should probably be replaced with something useful in a GUI environment
/// (eg. that returns the error as a [`String`] so that the GUI can show an error popup before
/// exiting).
pub trait Termination: process::Termination {
    fn is_success(&self) -> bool;
}

impl Termination for Infallible {
    fn is_success(&self) -> bool {
        match *self {}
    }
}

impl Termination for () {
    fn is_success(&self) -> bool {
        true
    }
}

impl<T: Termination, E: Debug> Termination for Result<T, E> {
    fn is_success(&self) -> bool {
        match self {
            Ok(term) => term.is_success(),
            Err(_) => false,
        }
    }
}
