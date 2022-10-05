use std::{
    io,
    panic::resume_unwind,
    thread::{self, JoinHandle},
};

use crossbeam::channel::Sender;

use crate::drop::defer;

/// Creates a connected pair of [`Promise`] and [`PromiseHandle`].
pub fn promise<T>() -> (Promise<T>, PromiseHandle<T>) {
    // Capacity of 1 means that `Promise::fulfill` will never block, which is the property we want.
    let (sender, recv) = crossbeam::channel::bounded(1);
    (Promise { inner: sender }, PromiseHandle { recv })
}

/// An empty slot that can be filled with a `T`, fulfilling the promise.
///
/// Fulfilling a [`Promise`] lets the connected [`PromiseHandle`] retrieve the value. A connected
/// pair of [`Promise`] and [`PromiseHandle`] can be created by calling [`promise`].
pub struct Promise<T> {
    inner: crossbeam::channel::Sender<T>,
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
        self.recv.recv().map_err(|_| PromiseDropped { _priv: () })
    }

    /// Returns whether the associated [`Promise`] has been fulfilled.
    ///
    /// If this returns `true`, calling [`PromiseHandle::block`] on `self` will return immediately,
    /// without blocking.
    pub fn is_fulfilled(&self) -> bool {
        // FIXME: this returns `false` when the promise is dropped, should really return `true` instead!
        // (rename to `will_block`?)
        !self.recv.is_empty()
    }
}

/// An error returned by [`PromiseHandle::block`] indicating that the connected [`Promise`] object
/// was dropped without being fulfilled.
#[derive(Debug, Clone, Copy)]
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
        let (sender, recv) = crossbeam::channel::bounded(self.capacity);
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
    /// This will block until the thread is available to accept the message.
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
    fn promise_is_fulfilled() {
        let (promise, handle) = promise();
        assert!(!handle.is_fulfilled());
        promise.fulfill(());
        assert!(handle.is_fulfilled());
        handle.block().unwrap();
    }
}
