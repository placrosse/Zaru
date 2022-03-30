//! Thread pipelining utilities.
//!
//! "Thread pipelining" is at the software level what CPU pipelining is at the hardware level – a
//! processing pipeline can be split up and distributed across threads to improve throughput without
//! sacrificing much latency or introducing the typical multithreading-related issues.
//!
//! The written code is typically very close to single-threaded code, with channel transfers
//! inserted when threads are crossed.

use std::marker::PhantomData;

use crossbeam::channel::{RecvError, SendError};

/// Creates a channel suitable for sending data between pipeline stages.
///
/// Values sent across the channel are not buffered. A call to [`Sender::send`] will block until
/// another thread is at a matching [`Receiver::recv`] call.
pub fn channel<T>() -> (InactiveSender<T>, InactiveReceiver<T>) {
    let (sender, recv) = crossbeam::channel::bounded(0);

    (
        InactiveSender { inner: sender },
        InactiveReceiver { inner: recv },
    )
}
/*
/// Creates a "buffer" channel suitable for data that bypasses pipeline stages.
///
/// This functions almost the same as [`channel`], but will buffer exactly 1 value without blocking.
///
/// To understand what this is useful for, consider the following setup:
///
/// ```text
///                         +----------+
///          /------------> | Thread 2 | ---\
/// +----------+            +----------+     \     +----------+
/// | Thread 1 |                              +--> | Thread 4 |
/// +----------+            +----------+     /     +----------+
///        | \------------> | Thread 3 | ---/        ^
///        |                +----------+             |
///        |                                         |
///        |                 buffer                  |
///        +-----------------------------------------+
/// ```
///
/// Here, thread 1 sends some data to threads 2 and 3, which then perform some (presumably
/// expensive) computations on it and send the result to thread 4. However, thread 1 might *also*
/// produce some data that is directly of interest to thread 4 and isn't needed by threads 2 and 3.
///
/// It could just send this data along to thread 2 or 3 and have that forward it to thread 4, but
/// that introduces some complexity in thread 2 or 3, and also requires picking one of them to
/// forward the extra data.
///
/// Using an additional channel to send the data directly to thread 4 does not work because that
/// blocks until thread 4
pub fn buffer<T>() -> (InactiveSender<T>, InactiveReceiver<T>) {
    let (sender, recv) = crossbeam::channel::bounded(1);

    (
        InactiveSender { inner: sender },
        InactiveReceiver { inner: recv },
    )
}*/

/// The inactive sending half of a channel.
pub struct InactiveSender<T> {
    inner: crossbeam::channel::Sender<T>,
}

impl<T> InactiveSender<T> {
    /// Activates the sender, locking it to the calling thread.
    ///
    /// The purpose of this "activation dance" is to ensure that pipeline threads *always* own the
    /// channels they use to communicate. This is important because that means the channel half will
    /// be dropped when the thread dies, notifying the other thread connected to the channel.
    ///
    /// Normal channels only require `&self` or `&mut self` to perform operations, which means that
    /// the using thread does not have to own the channel half. When using scoped threads, this
    /// might lead to a situation where a thread has died (possibly due to a panic), but the threads
    /// it was communicating with never finding out, because it doesn't drop its channel halves.
    pub fn activate(self) -> Sender<T> {
        Sender {
            inner: self.inner,
            _lock: PhantomData,
        }
    }
}

/// The inactive receiving half of a channel.
pub struct InactiveReceiver<T> {
    inner: crossbeam::channel::Receiver<T>,
}

impl<T> InactiveReceiver<T> {
    /// Activates the receiver, locking it to the calling thread.
    ///
    /// The purpose of this "activation dance" is to ensure that pipeline threads *always* own the
    /// channels they use to communicate. This is important because that means the channel half will
    /// be dropped when the thread dies, notifying the other thread connected to the channel.
    ///
    /// Normal channels only require `&self` or `&mut self` to perform operations, which means that
    /// the using thread does not have to own the channel half. When using scoped threads, this
    /// might lead to a situation where a thread has died (possibly due to a panic), but the threads
    /// it was communicating with never finding out, because it doesn't drop its channel halves.
    pub fn activate(self) -> Receiver<T> {
        Receiver {
            inner: self.inner,
            _lock: PhantomData,
        }
    }
}

/// The sending half of a channel.
pub struct Sender<T> {
    inner: crossbeam::channel::Sender<T>,
    /// Restricts the value to the owning thread.
    _lock: PhantomData<*const ()>,
}

impl<T> Sender<T> {
    /// Sends a value across the channel.
    ///
    /// This will block until the receiving thread calls [`Receiver::recv`].
    ///
    /// If the receiving thread dies or the receiver has been dropped, an error will be returned.
    /// If that happens, the caller should exit.
    pub fn send(&self, value: T) -> Result<(), SendError<T>> {
        self.inner.send(value)
    }
}

/// The receiving half of a channel.
pub struct Receiver<T> {
    inner: crossbeam::channel::Receiver<T>,
    /// Restricts the value to the owning thread.
    _lock: PhantomData<*const ()>,
}

impl<T> Receiver<T> {
    /// Receives a value from the channel.
    pub fn recv(&self) -> Result<T, RecvError> {
        self.inner.recv()
    }
}

impl<T> IntoIterator for Receiver<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            inner: self.inner.into_iter(),
            _lock: PhantomData,
        }
    }
}

/// Iterator over received messages.
pub struct IntoIter<T> {
    inner: crossbeam::channel::IntoIter<T>,
    /// Restricts the value to the owning thread.
    _lock: PhantomData<*const ()>,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}
