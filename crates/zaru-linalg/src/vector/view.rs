use std::{
    mem,
    ops::{Deref, DerefMut},
};

use crate::Vector;

// The reasonable part:

#[repr(C)]
pub struct X<T> {
    pub x: T,
    _priv: (), // prevent external construction
}

#[repr(C)]
pub struct XY<T> {
    pub x: T,
    pub y: T,
    _priv: (), // prevent external construction
}

#[repr(C)]
pub struct XYZ<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    _priv: (), // prevent external construction
}

#[repr(C)]
pub struct XYZW<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
    _priv: (), // prevent external construction
}

// The funny part:

#[repr(C)]
pub struct R<T> {
    pub r: T,
    _priv: (), // prevent external construction
}

#[repr(C)]
pub struct RG<T> {
    pub r: T,
    pub g: T,
    _priv: (), // prevent external construction
}

#[repr(C)]
pub struct RGB<T> {
    pub r: T,
    pub g: T,
    pub b: T,
    _priv: (), // prevent external construction
}

#[repr(C)]
pub struct RGBA<T> {
    pub r: T,
    pub g: T,
    pub b: T,
    pub a: T,
    _priv: (), // prevent external construction
}

// The "taking it too far" part:

#[repr(C)]
pub struct WH<T> {
    pub w: T,
    pub h: T,
    _priv: (), // prevent external construction
}

impl<T> Deref for Vector<T, 1> {
    type Target = X<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { mem::transmute(self) }
    }
}

impl<T> DerefMut for Vector<T, 1> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { mem::transmute(self) }
    }
}

impl<T> Deref for Vector<T, 2> {
    type Target = XY<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { mem::transmute(self) }
    }
}

impl<T> DerefMut for Vector<T, 2> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { mem::transmute(self) }
    }
}

impl<T> Deref for Vector<T, 3> {
    type Target = XYZ<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { mem::transmute(self) }
    }
}

impl<T> DerefMut for Vector<T, 3> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { mem::transmute(self) }
    }
}

impl<T> Deref for Vector<T, 4> {
    type Target = XYZW<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { mem::transmute(self) }
    }
}

impl<T> DerefMut for Vector<T, 4> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { mem::transmute(self) }
    }
}

impl<T> Deref for X<T> {
    type Target = R<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { mem::transmute(self) }
    }
}

impl<T> DerefMut for X<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { mem::transmute(self) }
    }
}

impl<T> Deref for XY<T> {
    type Target = RG<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { mem::transmute(self) }
    }
}

impl<T> DerefMut for XY<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { mem::transmute(self) }
    }
}

impl<T> Deref for XYZ<T> {
    type Target = RGB<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { mem::transmute(self) }
    }
}

impl<T> DerefMut for XYZ<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { mem::transmute(self) }
    }
}

impl<T> Deref for XYZW<T> {
    type Target = RGBA<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { mem::transmute(self) }
    }
}

impl<T> DerefMut for XYZW<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { mem::transmute(self) }
    }
}

impl<T> Deref for RG<T> {
    type Target = WH<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { mem::transmute(self) }
    }
}

impl<T> DerefMut for RG<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { mem::transmute(self) }
    }
}
