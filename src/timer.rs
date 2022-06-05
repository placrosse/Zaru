//! Performance measurement tools.

use std::{
    cell::Cell,
    fmt::{self, Arguments},
    time::{Duration, Instant},
};

use crate::filter::{ema, Filter};

const EMA_ALPHA: f32 = 0.3;

/// A timer that can measure and average the time an operation takes.
///
/// Collected timings are averaged and reset when the timer is displayed using `{}`
/// ([`std::fmt::Display`]).
pub struct Timer {
    name: &'static str,
    ema: ema::Ema,
    ema_state: Cell<ema::State>,
    /// The current average time.
    avg: Cell<f32>,
    /// The number of time measurements that contributed to the current `avg`.
    count: Cell<usize>,
}

impl Timer {
    /// Creates a new timer.
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            ema: ema::Ema::new(EMA_ALPHA),
            ema_state: Default::default(),
            avg: Cell::new(0.0),
            count: Cell::new(0),
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
        let duration = start.elapsed();
        let state = self.ema_state.get_mut();
        self.avg.set(self.ema.filter(state, duration.as_secs_f32()));
        *self.count.get_mut() += 1;
    }
}

/// Displays the average recorded time and resets it.
impl fmt::Display for Timer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // (this can't actually fail, `time` takes `&mut self` and this function can't be
        // invoked more than once at the same time because `Timer` isn't `Sync`)
        self.ema_state.replace(Default::default());

        let avg = self.avg.replace(0.0);
        let len = self.count.replace(0);
        let avg_ms = avg * 1000.0;

        write!(f, "{}: {len}x{avg_ms:.01}ms", self.name)
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
