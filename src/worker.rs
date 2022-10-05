use std::{
    io,
    panic::resume_unwind,
    thread::{self, JoinHandle},
};

use flume::{Receiver, Sender};

use crate::drop::defer;

/// Creates a connected pair of [`Promise`] and [`PromiseHandle`].
pub fn promise<T>() -> (Promise<T>, PromiseHandle<T>) {
    // Capacity of 1 means that `Promise::fulfill` will never block, which is the property we want.
    let (sender, recv) = flume::bounded(1);
    (Promise { inner: sender }, PromiseHandle { recv })
}

/// An empty slot that can be filled with a `T`, fulfilling the promise.
///
/// Fulfilling a [`Promise`] lets the connected [`PromiseHandle`] retrieve the value. A connected
/// pair of [`Promise`] and [`PromiseHandle`] can be created by calling [`promise`].
pub struct Promise<T> {
    inner: Sender<T>,
}

impl<T> Promise<T> {
    /// Fulfills the promise with a value, consuming it.
    ///
    /// If a thread is currently waiting at [`PromiseHandle::block`], it will be woken up.
    ///
    /// This method does not block or fail. If the connected [`PromiseHandle`] was dropped, `value`
    /// will be dropped and nothing happens. The calling thread is expected to exit when it attempts
    /// to obtain a new [`Promise`] to fulfill.
    pub fn fulfill(self, value: T) {
        // This ignores errors. The assumption is that the thread will exit once it tries to obtain
        // a new `Promise` to fulfill.
        self.inner.send(value).ok();
    }
}

/// A handle connected to a [`Promise`] that will eventually resolve to a value of type `T`.
///
/// A connected pair of [`Promise`] and [`PromiseHandle`] can be created by calling [`promise`].
pub struct PromiseHandle<T> {
    recv: Receiver<T>,
}

impl<T> PromiseHandle<T> {
    /// Blocks the calling thread until the connected [`Promise`] is fulfilled.
    ///
    /// If the [`Promise`] is dropped without being fulfilled, a [`PromiseDropped`] error is
    /// returned instead. This typically means one of two things:
    ///
    /// - The thread holding the promise has deliberately decided not to fulfill it (for
    ///   example, because it has skipped processing an item).
    /// - The thread holding the promise has panicked.
    ///
    /// Usually, the correct way to handle this is to just skip the item expected from the
    /// [`Promise`]. If the thread has panicked, and it's a [`Worker`] thread, then the next
    /// attempt to send a message to it will propagate the panic to the owning thread, and tear
    /// down the process as usual.
    pub fn block(self) -> Result<T, PromiseDropped> {
        self.recv.recv().map_err(|_| PromiseDropped { _priv: () })
    }

    /// Tests whether a call to [`PromiseHandle::block`] will block or return immediately.
    ///
    /// If this returns `false`, calling [`PromiseHandle::block`] on `self` will return immediately,
    /// without blocking.
    pub fn will_block(&self) -> bool {
        // `Promise::fulfill` drops the sender and disconnects the channel, and so does dropping a
        // `Promise` without fulfilling it.
        !self.recv.is_disconnected()
    }
}

/// An error returned by [`PromiseHandle::block`] indicating that the connected [`Promise`] object
/// was dropped without being fulfilled.
#[derive(Debug, Clone)]
pub struct PromiseDropped {
    _priv: (),
}

/// A builder object that can be used to configure and spawn a [`Worker`].
#[derive(Clone)]
pub struct WorkerBuilder {
    name: Option<String>,
    capacity: usize,
}

impl WorkerBuilder {
    /// Sets the name of the [`Worker`] thread.
    pub fn name<N: Into<String>>(self, name: N) -> Self {
        Self {
            name: Some(name.into()),
            ..self
        }
    }

    /// Sets the channel capacity of the [`Worker`].
    ///
    /// By default, a capacity of 0 is used, which means that [`Worker::send`] will block until the
    /// worker has finished processing any preceding message.
    ///
    /// When a pipeline of [`Worker`]s is used, the capacity of later [`Worker`]s may be increased
    /// to allow the processing of multiple input messages at once.
    pub fn capacity(self, capacity: usize) -> Self {
        Self { capacity, ..self }
    }

    /// Spawns a [`Worker`] thread that uses `handler` to process incoming messages.
    pub fn spawn<I, F>(self, mut handler: F) -> io::Result<Worker<I>>
    where
        I: Send + 'static,
        F: FnMut(I) + Send + 'static,
    {
        let (sender, recv) = flume::bounded(self.capacity);
        let mut builder = thread::Builder::new();
        if let Some(name) = self.name.clone() {
            builder = builder.name(name);
        }
        let handle = builder.spawn(move || {
            let _guard;
            if let Some(name) = self.name {
                log::trace!("worker '{name}' starting");
                _guard = defer(move || log::trace!("worker '{name}' exiting"));
            }
            for message in recv {
                handler(message);
            }
        })?;

        Ok(Worker {
            sender: Some(sender),
            handle: Some(handle),
        })
    }
}

/// A handle to a worker thread that processes messages of type `I`.
///
/// This type enforces structured concurrency: When it's dropped, the thread will be signaled to
/// exit and the thread will be joined. If the thread has panicked, the panic will be forwarded
/// to the thread dropping the [`Worker`].
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

impl Worker<()> {
    /// Returns a builder that can be used to configure and spawn a [`Worker`].
    #[inline]
    pub fn builder() -> WorkerBuilder {
        WorkerBuilder {
            name: None,
            capacity: 0,
        }
    }
}

impl<I: Send + 'static> Worker<I> {
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
    /// If the worker's channel capacity is exhausted, this will block until the worker is available
    /// to accept the message.
    ///
    /// If the worker has panicked, this will propagate the panic to the calling thread.
    pub fn send(&mut self, msg: I) {
        match self.sender.as_ref().unwrap().send(msg) {
            Ok(()) => {}
            Err(_) => {
                self.wait_for_exit();
            }
        }
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
        let mut worker = Worker::builder()
            .spawn(|_: ()| silent_panic("worker panic".into()))
            .unwrap();
        worker.send(());
        catch_unwind(AssertUnwindSafe(|| drop(worker))).unwrap_err();
    }

    #[test]
    fn worker_propagates_panic_on_send() {
        let mut worker = Worker::builder()
            .spawn(|_| silent_panic("worker panic".into()))
            .unwrap();
        worker.send(());
        catch_unwind(AssertUnwindSafe(|| worker.send(()))).unwrap_err();
        catch_unwind(AssertUnwindSafe(|| drop(worker))).unwrap();
    }

    #[test]
    fn promise_fulfillment() {
        let (promise, handle) = promise();
        assert!(handle.will_block());
        promise.fulfill(());
        assert!(!handle.will_block());
        handle.block().unwrap();
    }

    #[test]
    fn promise_drop() {
        let (promise, handle) = promise::<()>();
        assert!(handle.will_block());
        drop(promise);
        assert!(!handle.will_block());
        handle.block().unwrap_err();
    }
}
