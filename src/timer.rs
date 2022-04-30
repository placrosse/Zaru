//! Performance measurement tools.

use std::{
    cell::{Cell, RefCell},
    fmt::{self, Arguments},
    time::{Duration, Instant},
};

const MAX_DURATIONS: usize = 250;

/// A timer that can measure and average the time an operation takes.
///
/// Collected timings are averaged and reset when the timer is displayed using `{}`
/// ([`std::fmt::Display`]).
pub struct Timer {
    name: &'static str,
    durations: RefCell<Vec<Duration>>,
    forgotten: Cell<bool>,
}

impl Timer {
    /// Creates a new timer.
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            durations: Default::default(),
            forgotten: Cell::new(false),
        }
    }

    /// Invokes a closure, measuring and recording the time it takes.
    pub fn time<T>(&mut self, timee: impl FnOnce() -> T) -> T {
        let _guard = self.start();
        timee()
    }

    /// Starts timing an operation using a drop guard.
    ///
    /// When the returned [`TimerGuard`] is dropped, the time between the call to `start` and the
    /// drop is measured and recorded.
    pub fn start(&mut self) -> TimerGuard<'_> {
        TimerGuard {
            start: Instant::now(),
            timer: self,
        }
    }

    fn stop(&mut self, start: Instant) {
        if self.forgotten.get() {
            return;
        }

        let duration = start.elapsed();
        let durations = &mut self.durations.get_mut();
        if durations.len() < MAX_DURATIONS {
            durations.push(duration);
        } else {
            // FIXME use a better strategy
            self.forgotten.set(true);
            durations.clear();
        }
    }
}

/// Displays the average recorded time and resets it.
impl fmt::Display for Timer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.forgotten.get() {
            write!(f, "{}: <forgotten>", self.name)
        } else {
            // (this can't actually fail, `time` takes `&mut self` and this function can't be
            // invoked more than once at the same time because `Timer` isn't `Sync`)
            let mut durations = self.durations.borrow_mut();
            let len = durations.len();
            let num = durations.len() as f32;
            let avg_ms = durations
                .iter()
                .fold(0.0, |prev, new| prev + new.as_secs_f32() * 1000.0 / num);
            durations.clear();

            write!(f, "{}: {len}x{avg_ms:.01}ms", self.name)
        }
    }
}

/// Guard returned by [`Timer::start`]. Stops timing the operation when dropped.
pub struct TimerGuard<'a> {
    start: Instant,
    timer: &'a mut Timer,
}

impl Drop for TimerGuard<'_> {
    fn drop(&mut self) {
        self.timer.stop(self.start);
    }
}

/// Logs frames per second with optional extra data.
pub struct FpsCounter {
    name: String,
    frames: u32,
    start: Instant,
}

impl FpsCounter {
    pub fn new<N: Into<String>>(name: N) -> Self {
        Self {
            name: name.into(),
            frames: 0,
            start: Instant::now(),
        }
    }

    /// Advances the frame counter by 1 and logs FPS if one second has passed.
    ///
    /// The logged string will also include the timer's name passed to [`Timer::new`].
    pub fn tick(&mut self) {
        self.tick_impl(format_args!(""));
    }

    /// Advances the frame counter by 1 and logs FPS and `extra` data if one second has passed.
    ///
    /// The logged string will also include the timer's name passed to [`Timer::new`].
    pub fn tick_with<D: fmt::Display, I: IntoIterator<Item = D>>(&mut self, extra: I) {
        struct DisplayExtra<D: fmt::Display, I: Iterator<Item = D>>(Cell<Option<I>>);

        impl<D: fmt::Display, I: Iterator<Item = D>> fmt::Display for DisplayExtra<D, I> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let mut iter = self.0.take().unwrap();
                let item = iter.next();
                match item {
                    Some(item) => {
                        f.write_str(" (")?;
                        write!(f, "{}", item)?;
                        for item in iter {
                            f.write_str(", ")?;
                            write!(f, "{}", item)?;
                        }
                        f.write_str(")")?;
                        Ok(())
                    }
                    None => Ok(()),
                }
            }
        }

        self.tick_impl(format_args!(
            "{}",
            DisplayExtra(Cell::new(Some(extra.into_iter())))
        ));
    }

    fn tick_impl(&mut self, args: Arguments<'_>) {
        self.frames += 1;
        if self.start.elapsed() > Duration::from_secs(1) {
            log::debug!("{}: {} FPS{}", self.name, self.frames, args);

            self.frames = 0;
            self.start = Instant::now();
        }
    }
}
