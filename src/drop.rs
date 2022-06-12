//! Utilities related to destructors and drop.

/// Drop guard returned by [`defer`].
#[must_use = "`Defer` should be assigned to a variable, or it will be dropped immediately"]
pub struct Defer<F: FnOnce()>(Option<F>);

impl<F: FnOnce()> Drop for Defer<F> {
    fn drop(&mut self) {
        (self.0.take().unwrap())();
    }
}

/// Returns a value that runs `cb` when dropped.
pub fn defer<F: FnOnce()>(cb: F) -> Defer<F> {
    Defer(Some(cb))
}

/// A type that panics when dropped, unless [`DropBomb::defuse`] is called first.
pub struct DropBomb {
    msg: &'static str,
    defused: bool,
}

impl DropBomb {
    /// Creates a new drop bomb that will panic with `msg` when dropped.
    pub fn new(msg: &'static str) -> Self {
        Self {
            msg,
            defused: false,
        }
    }

    /// Defuses the bomb, making it no longer panic when dropped.
    pub fn defuse(&mut self) {
        self.defused = true;
    }
}

impl Drop for DropBomb {
    fn drop(&mut self) {
        if !std::thread::panicking() && !self.defused {
            panic!("{}", self.msg);
        }
    }
}
