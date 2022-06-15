//! Thread pipelining utilities.
//!
//! "Thread pipelining" is at the software level what CPU pipelining is at the hardware level – a
//! processing pipeline can be split up and distributed across threads to improve throughput without
//! sacrificing much latency or introducing the typical multithreading-related issues.
//!
//! The written code is typically very close to single-threaded code, with channel transfers
//! inserted when threads are crossed.

use std::{
    io,
    marker::PhantomData,
    panic::resume_unwind,
    thread::{self, JoinHandle},
};

use crossbeam::channel::{RecvError, SendError};

use crate::drop::DropBomb;

/// Creates a channel suitable for sending data between pipeline stages.
///
/// Values sent across the channel are not buffered. A call to [`Sender::send`] will block until
/// another thread is at a matching [`Receiver::recv`] call.
///
/// Therefore, this type of channel is suitable only for data moving *forwards* in a processing
/// pipeline, or *into* a [`Worker`]. If data is sent backwards, a [`promise`] should be used
/// instead.
pub fn channel<T>() -> (InactiveSender<T>, InactiveReceiver<T>) {
    let (sender, recv) = crossbeam::channel::bounded(0);

    (
        InactiveSender { inner: sender },
        InactiveReceiver { inner: recv },
    )
}

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

/// Creates a connected pair of [`Promise`] and [`PromiseHandle`].
pub fn promise<T>() -> (Promise<T>, PromiseHandle<T>) {
    // Capacity of 1 means that `Promise::fulfill` will never block, which is the property we want.
    let (sender, recv) = crossbeam::channel::bounded(1);
    let bomb = DropBomb::new("`Promise` dropped without being fulfilled");

    (
        Promise {
            inner: sender,
            bomb,
        },
        PromiseHandle { recv },
    )
}

/// An empty slot that can be filled with a `T`, fulfilling the promise.
///
/// This is a *linear type*: if it gets dropped without being fulfilled, the thread will panic.
///
/// This API is vaguely inspired by C++'s `std::promise`.
pub struct Promise<T> {
    inner: crossbeam::channel::Sender<T>,
    bomb: DropBomb,
}

impl<T> Promise<T> {
    /// Fulfills the promise with a value, consuming it.
    ///
    /// If a thread is currently waiting at [`PromiseHandle::block`], it will be woken up.
    ///
    /// This method does not block or fail. If the connected [`PromiseHandle`] was dropped, `value`
    /// will be dropped and nothing happens. The calling thread is expected to exit when it attempts
    /// to obtain a new [`Promise`] to fulfill.
    pub fn fulfill(mut self, value: T) {
        // This ignores errors. The assumption is that the thread will exit once it tries to obtain
        // a new `Promise` to fulfill.
        self.inner.send(value).ok();
        self.bomb.defuse();
    }
}

/// A handle connected to a [`Promise`] that will eventually resolve to a value of type `T`.
///
/// C++ calls this a "future", but that means something pretty different in Rust.
pub struct PromiseHandle<T> {
    recv: crossbeam::channel::Receiver<T>,
}

impl<T> PromiseHandle<T> {
    /// Blocks the calling thread until the [`Promise`] is fulfilled.
    pub fn block(self) -> Result<T, PromiseDropped> {
        /*
        Problem: when this fails, we know that the other thread has panicked, but we have no access
        to its panic payload, so we can't propagate it. If the other thread is a `Worker`, it will
        propagate automatically when dropped, but we cannot trigger that from this method: if we
        unwind with a different payload, `Worker`s destructor cannot resume unwinding with the
        correct payload later.
        So we make the caller handle this.
        */
        self.recv.recv().map_err(|_| PromiseDropped)
    }

    /// Returns whether the associated [`Promise`] has been fulfilled.
    ///
    /// If this returns `true`, calling [`PromiseHandle::block`] on `self` will return immediately,
    /// without blocking.
    pub fn is_fulfilled(&self) -> bool {
        // FIXME: this returns `false` when the promise is dropped, should really return `true` instead!
        !self.recv.is_empty()
    }
}

/// An error indicating that the connected [`Promise`] object was dropped without being fulfilled.
///
/// Since [`Promise`] panics on drop, this indicates that the thread holding it is panicking. Thus,
/// when this error is received, the other thread should be joined and its panic payload propagated
/// (if it's a [`Worker`], simply dropping it is enough).
#[derive(Debug, Clone, Copy)]
pub struct PromiseDropped;

/// A handle to a worker thread that processes messages of type `I`.
///
/// When dropped, the channel to the thread will be dropped and the thread will be joined. If the
/// thread has panicked, the panic will be forwarded to the thread dropping the `Worker`.
pub struct Worker<I: Send + 'static> {
    sender: Option<Sender<I>>,
    handle: Option<JoinHandle<()>>,
}

impl<I: Send + 'static> Drop for Worker<I> {
    fn drop(&mut self) {
        // Close the channel to signal the thread to exit.
        drop(self.sender.take());

        self.wait_for_exit();
    }
}

impl<I: Send + 'static> Worker<I> {
    pub fn spawn<N, F>(name: N, handler: F) -> io::Result<Self>
    where
        N: Into<String>,
        F: FnOnce(Receiver<I>) + Send + 'static,
    {
        let (sender, recv) = channel();
        let handle = thread::Builder::new()
            .name(name.into())
            .spawn(|| handler(recv.activate()))?;

        Ok(Self {
            sender: Some(sender.activate()),
            handle: Some(handle),
        })
    }

    fn wait_for_exit(&mut self) {
        // Wait for it to exit and propagate its panic if it panicked.
        if let Some(handle) = self.handle.take() {
            match handle.join() {
                Ok(()) => {}
                Err(payload) => {
                    if !thread::panicking() {
                        resume_unwind(payload);
                    }
                }
            }
        }
    }

    /// Sends a message to the worker thread.
    ///
    /// This will block until the thread is available to accept the message.
    pub fn send(&mut self, msg: I) -> Result<(), SendError<I>> {
        self.sender.as_mut().unwrap().send(msg)
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use super::*;

    fn silent_panic(payload: String) {
        resume_unwind(Box::new(payload));
    }

    #[test]
    fn worker_propagates_panic_on_drop() {
        let worker = Worker::spawn("panic", |_: Receiver<()>| {
            silent_panic("worker panic".into())
        })
        .unwrap();
        catch_unwind(AssertUnwindSafe(|| drop(worker))).unwrap_err();
    }

    #[test]
    fn worker_does_not_propagate_panic_on_send() {
        let mut worker = Worker::spawn("panic", |_| silent_panic("worker panic".into())).unwrap();
        catch_unwind(AssertUnwindSafe(|| worker.send(())))
            .unwrap()
            .unwrap_err();
        catch_unwind(AssertUnwindSafe(|| drop(worker))).unwrap_err();
    }
}
