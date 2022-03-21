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
pub fn channel<T>() -> (Sender<T>, Receiver<T>) {
    let (sender, recv) = crossbeam::channel::bounded(0);

    (Sender { inner: sender }, Receiver { inner: recv })
}

/// The inactive sending half of a channel.
pub struct Sender<T> {
    inner: crossbeam::channel::Sender<T>,
}

impl<T> Sender<T> {
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
    pub fn activate(self) -> ActiveSender<T> {
        ActiveSender {
            inner: self.inner,
            _lock: PhantomData,
        }
    }
}

/// The inactive receiving half of a channel.
pub struct Receiver<T> {
    inner: crossbeam::channel::Receiver<T>,
}

impl<T> Receiver<T> {
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
    pub fn activate(self) -> ActiveReceiver<T> {
        ActiveReceiver {
            inner: self.inner,
            _lock: PhantomData,
        }
    }
}

/// The sending half of a channel.
pub struct ActiveSender<T> {
    inner: crossbeam::channel::Sender<T>,
    /// Restricts the value to the owning thread.
    _lock: PhantomData<*const ()>,
}

impl<T> ActiveSender<T> {
    /// Sends a value across the channel.
    ///
    /// This will block until the receiving thread calls [`ActiveReceiver::recv`].
    ///
    /// If the receiving thread dies or the receiver has been dropped, an error will be returned.
    /// If that happens, the caller should exit.
    pub fn send(&mut self, value: T) -> Result<(), SendError<T>> {
        self.inner.send(value)
    }
}

/// The receiving half of a channel.
pub struct ActiveReceiver<T> {
    inner: crossbeam::channel::Receiver<T>,
    /// Restricts the value to the owning thread.
    _lock: PhantomData<*const ()>,
}

impl<T> ActiveReceiver<T> {
    /// Receives a value from the channel.
    pub fn recv(&mut self) -> Result<T, RecvError> {
        self.inner.recv()
    }
}

impl<T> IntoIterator for ActiveReceiver<T> {
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
