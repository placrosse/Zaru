#[must_use = "`OnDrop` should be assigned to a variable, or it will be dropped immediately"]
pub struct OnDrop<F: FnOnce()>(Option<F>);

impl<F: FnOnce()> Drop for OnDrop<F> {
    fn drop(&mut self) {
        (self.0.take().unwrap())();
    }
}

/// Returns a value that runs `cb` when dropped.
pub fn on_drop<F: FnOnce()>(cb: F) -> OnDrop<F> {
    OnDrop(Some(cb))
}
