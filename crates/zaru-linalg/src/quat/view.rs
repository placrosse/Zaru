use std::{
    mem,
    ops::{Deref, DerefMut},
};

use crate::Quat;

#[repr(C)]
pub struct IJKW<T> {
    pub i: T,
    pub j: T,
    pub k: T,
    pub w: T,
}

impl<T> Deref for Quat<T> {
    type Target = IJKW<T>;

    fn deref(&self) -> &Self::Target {
        unsafe { mem::transmute(self) }
    }
}

impl<T> DerefMut for Quat<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { mem::transmute(self) }
    }
}
