//! A work-sharing thread pool sized for weft's parallel shape.
//!
//! Every parallel region in weft has the same form: `n` independent items,
//! each already sized to be worth a task, run to completion before the
//! caller continues. There is no recursive subdivision, no nested
//! parallelism, and no task that outlives its region. That is a much
//! narrower contract than a general work-stealing runtime implements, and
//! it buys three things this pool is built around:
//!
//! - **No per-region allocation and no deques.** A region publishes one
//!   pointer to a closure living on the caller's stack; workers claim item
//!   ranges out of a single atomic cursor. Claiming work is one CAS (or one
//!   `fetch_add`), not a push/pop/steal protocol.
//! - **Stable, dense worker indices.** Every closure receives its worker
//!   index directly, so per-worker scratch (nearest-color caches, LZW
//!   encoders) is a plain array lookup instead of a thread-local probe, and
//!   the index is always valid — there is no "called from outside the pool"
//!   case to reserve a spare slot for.
//! - **The submitting thread is a worker.** `Pool::new(n)` spawns `n - 1`
//!   threads; the caller runs share `n - 1`. `--threads N` therefore means
//!   N runnable threads, not N plus a blocked one, and `Pool::new(1)` is
//!   pure inline execution with no threads and no atomics on the fast path.
//! - **Joining a region is optional.** A region is open (even epoch) or
//!   closed (odd). A worker joins by incrementing the active count and
//!   re-reading the epoch; if the region closed in between it backs out
//!   without touching the job. The caller closes the region once it has
//!   drained the cursor itself and waits only for workers that joined. So
//!   a worker that the OS descheduled, or that is still waking from a
//!   park, simply misses the region instead of stalling it — the first
//!   version of this pool made every worker acknowledge every region, and
//!   at 16+ threads one late worker per region was enough to push each
//!   barrier past the caller's spin budget and collapse the whole pool
//!   into a futex round trip per region (52 us at 22 threads, against
//!   3 us now; see the `bench` module).
//!
//! Item granularity follows the two shapes weft actually has. A region of
//! few, large, unevenly sized items — frames, row strips, palette buckets —
//! hands out one item per claim, which is the best balance available and
//! whose atomic is invisible next to the item. A region of many tiny items
//! — one histogram entry, one grid cell — would drown in that atomic, so
//! above 32 items per participant claims are guided: each takes
//! `remaining / (4 * threads)` items, coarse at the start and decaying to
//! single items over the tail, so the region ends evenly.
//!
//! Results are always written to their own index, so every `map` here
//! preserves input order exactly: float reductions downstream see the same
//! association at any thread count, which is what keeps weft's output
//! byte-identical across `--threads`.
//!
//! Between regions workers spin briefly, then poll on `yield_now`, then
//! park. Back-to-back regions (the histogram batch loop, median cut's box
//! scans) find them spinning and dispatch in tens of nanoseconds; the
//! yield phase covers the serial gaps between phases at no cost on an idle
//! core and hands the core to the reader or hold threads on a busy one; a
//! gap longer than that parks them, and the next region unparks only as
//! many as it has items for, then starts without waiting for them. A
//! worker whose last idle ended in a park skips the spin and yield the
//! next time and parks at once (adaptive spinning, after filament): on a
//! slow input the pool is idle between batches, and burning the yield
//! window on every worker per batch is a few percent of a core taken from
//! the decoder feeding us. It warms back up when a park turns out short
//! or when it wakes into a region that has already closed.

use std::any::Any;
use std::cell::Cell;
use std::hint;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use Ordering::{Acquire, Relaxed, Release, SeqCst};

/// Spin iterations before a waiter looks at the clock. Long enough to
/// cover a dispatch that is already in flight without paying for a
/// `Instant::now` in the common case.
const SPIN_QUICK: u32 = 64;
/// How long a waiter spins on `pause` before it starts yielding. Bounded
/// in time rather than in iterations because a `pause` costs anywhere from
/// ~10 cycles (Broadwell) to ~140 (Skylake and later) to ~1 (a plain yield
/// hint on aarch64), so an iteration count that idles for 15 us on one
/// machine idles for 200 us on the next.
const SPIN_TIME: Duration = Duration::from_micros(10);
/// How long a waiter keeps polling with `yield_now` between polls before
/// it parks. A yield returns at once on an idle core and hands the core
/// over on a busy one, so this phase is nearly free either way, and it is
/// what keeps the pool out of the park/wake regime: a futex wake costs
/// tens of microseconds on a virtual machine, more than most of the
/// serial gaps between weft's regions, so a worker that parks at 10 us
/// would be asleep for exactly the gaps it should have covered.
#[cfg(not(all(target_arch = "aarch64", target_vendor = "apple")))]
const YIELD_TIME: Duration = Duration::from_micros(200);
// On the M1 Max, parking after the initial spin beat repeated scheduler
// yields in both the subset screen and the full RGBA corpus comparison.
#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
const YIELD_TIME: Duration = Duration::ZERO;
/// A park shorter than this means the next region arrived faster than the
/// park/unpark path can deliver it, so the worker should have stayed
/// awake: it goes back to spinning. Longer parks are the pauses between
/// pipeline stages or the gaps of a slow input, where spinning first would
/// only burn the yield window for nothing (see `worker`).
const HOT_GAP: Duration = Duration::from_millis(1);

// ---------------------------------------------------------------------------
// Global pool

static GLOBAL: OnceLock<Pool> = OnceLock::new();

/// Build the process-wide pool with `threads` participants. Called once
/// from `main`, before anything can reach [`global`] and settle the
/// default; a second call would mean `--threads` had been silently
/// ignored, so it fails loudly instead.
pub fn init(threads: usize) {
    assert!(
        GLOBAL.set(Pool::new(threads)).is_ok(),
        "the pool was already running before --threads was applied"
    );
}

/// The process-wide pool, defaulting to one participant per core.
pub fn global() -> &'static Pool {
    GLOBAL.get_or_init(|| Pool::new(default_threads()))
}

fn default_threads() -> usize {
    thread::available_parallelism().map_or(1, |n| n.get())
}

// ---------------------------------------------------------------------------
// Dispatch

/// The closure for the region in flight, as seen by a worker. Published as
/// a raw pointer to the caller's stack; the caller does not return until
/// every worker that joined the region has left it, and a worker only
/// reads the pointer after a join that succeeded, so the referent outlives
/// every read.
struct Job<'a> {
    run: &'a (dyn Fn(usize) + Sync),
}

/// Pads a field out to its own pair of cache lines (Intel's adjacent-line
/// prefetcher pairs 64-byte lines, so 128 is the unit that keeps two
/// hot fields from ever sharing traffic).
#[repr(align(128))]
struct Pad<T>(T);

impl<T> std::ops::Deref for Pad<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        &self.0
    }
}

/// What the publisher writes once per region and every worker polls:
/// kept on its own line, away from anything workers write.
///
/// `epoch` is even while a region is open and odd while none is. Opening
/// a region bumps it to the next even value; closing it bumps it again.
struct Region {
    epoch: AtomicU64,
    /// `*const Job`, published before the epoch bump that opens it.
    job: AtomicPtr<()>,
}

/// The completion barrier: workers that have joined the open region and
/// not yet left it. Nothing else lives on this line, so their RMWs
/// contend only with each other and never with the spinners polling
/// `Region`.
struct Barrier {
    active: AtomicUsize,
    /// Set while the caller is blocked on `done`.
    waiting: AtomicBool,
}

struct Shared {
    threads: usize,
    /// Held for the duration of a region. weft submits every region of a
    /// given pool from one thread, so this is uncontended there; it is what
    /// makes a pool shared by several submitting threads (the unit tests
    /// share the global pool) queue instead of corrupting each other's job
    /// slot.
    submit: Mutex<()>,
    region: Pad<Region>,
    barrier: Pad<Barrier>,
    /// One flag per worker, set while it is parked (or committing to park),
    /// so the publisher unparks only the threads that need it.
    parked: Vec<Pad<AtomicBool>>,
    shutdown: Pad<AtomicBool>,
    /// Set by a worker that stored a panic, so the per-region check is a
    /// load rather than a mutex.
    panicked: Pad<AtomicBool>,
    lock: Mutex<()>,
    done: Condvar,
    /// First panic escaping a worker in the current region.
    panic: Mutex<Option<Box<dyn Any + Send>>>,
    /// Worker thread handles, for `unpark`; filled in by `Pool::new`
    /// before any region can run.
    handles: OnceLock<Vec<thread::Thread>>,
}

thread_local! {
    /// The pool this thread is currently executing a region for, and its
    /// worker index in it. Lets a (never intended, but not deadlocking)
    /// nested region on the same pool run inline under the right index.
    static CURRENT: Cell<(*const Shared, usize)> = const { Cell::new((std::ptr::null(), 0)) };
}

pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

impl Pool {
    /// A pool with `threads` participants: `threads - 1` spawned workers
    /// plus whichever thread submits the region. `threads` is clamped to at
    /// least 1, where every region runs inline.
    pub fn new(threads: usize) -> Pool {
        let threads = threads.max(1);
        let shared = Arc::new(Shared {
            threads,
            submit: Mutex::new(()),
            region: Pad(Region {
                // odd: nothing in flight
                epoch: AtomicU64::new(1),
                job: AtomicPtr::new(std::ptr::null_mut()),
            }),
            barrier: Pad(Barrier {
                active: AtomicUsize::new(0),
                waiting: AtomicBool::new(false),
            }),
            parked: (0..threads - 1)
                .map(|_| Pad(AtomicBool::new(false)))
                .collect(),
            shutdown: Pad(AtomicBool::new(false)),
            panicked: Pad(AtomicBool::new(false)),
            lock: Mutex::new(()),
            done: Condvar::new(),
            panic: Mutex::new(None),
            handles: OnceLock::new(),
        });
        let workers: Vec<JoinHandle<()>> = (0..threads - 1)
            .map(|i| {
                let shared = shared.clone();
                thread::Builder::new()
                    .name(format!("weft-{i}"))
                    .spawn(move || worker(&shared, i))
                    .expect("spawn pool worker")
            })
            .collect();
        let handles: Vec<thread::Thread> = workers.iter().map(|w| w.thread().clone()).collect();
        let _ = shared.handles.set(handles);
        Pool { shared, workers }
    }

    /// Participants, including the submitting thread. Worker indices handed
    /// to region closures are `0..threads()`.
    #[inline]
    pub fn threads(&self) -> usize {
        self.shared.threads
    }

    /// Run `f(worker_index)` on this thread and on every worker that joins
    /// the region before this thread has exhausted it, and return once all
    /// of them have left it. `f` is expected to pull work from a shared
    /// cursor and return when it is empty. `useful` is how many workers
    /// could possibly find work — parked workers beyond that are left
    /// asleep.
    ///
    /// Joining is optional by design: a worker that is descheduled, or
    /// still waking, simply misses the region (or joins in time to find
    /// the cursor empty), so no region ever waits on a thread that has
    /// nothing to contribute.
    fn dispatch(&self, useful: usize, f: &(dyn Fn(usize) + Sync)) {
        let s = &*self.shared;
        let nested = CURRENT.with(|c| {
            let (p, w) = c.get();
            (std::ptr::eq(p, s)).then_some(w)
        });
        // Already inside a region of this pool: the other participants are
        // busy in the outer region, so run inline under this thread's index
        // rather than deadlocking on a barrier that can never fill.
        if let Some(w) = nested {
            f(w);
            return;
        }
        let me = s.threads - 1;
        if s.threads == 1 {
            f(me);
            return;
        }
        // Held until the region is closed and drained.
        let _submit = s.submit.lock().unwrap_or_else(|e| e.into_inner());
        let job = Job { run: f };
        // SAFETY: the region is closed, and every worker that joined it has
        // left, before this function returns, so the borrow this erases
        // cannot outlive `job`.
        let ptr = &job as *const Job as *mut ();
        s.region.job.store(ptr, Release);
        let open = s.region.epoch.fetch_add(1, SeqCst) + 1;
        debug_assert!(open.is_multiple_of(2));
        s.wake(useful.min(s.threads - 1));
        let mine = CURRENT.with(|c| c.replace((s as *const Shared, me)));
        let caught = panic::catch_unwind(AssertUnwindSafe(|| f(me)));
        CURRENT.with(|c| c.set(mine));
        // Close, then wait for whoever got in. A worker's join is
        // `active += 1` followed by a re-read of the epoch: if that re-read
        // sees the close, it backs out without touching the job; if it
        // does not, its increment preceded the close in the SeqCst order
        // and the load below sees it.
        s.region.epoch.fetch_add(1, SeqCst);
        s.wait_done();
        // Panics are reported only after the barrier: the region's closure
        // and outputs live on this stack. Drain the workers' slot before
        // re-raising anything, both so no lock is held across the unwind
        // (that would poison it for every later region) and so a worker
        // panic cannot survive into the next region when this thread has
        // its own to raise.
        let from_worker = if s.panicked.swap(false, SeqCst) {
            s.panic.lock().unwrap_or_else(|e| e.into_inner()).take()
        } else {
            None
        };
        if let Err(p) = caught {
            panic::resume_unwind(p);
        }
        if let Some(p) = from_worker {
            panic::resume_unwind(p);
        }
    }
}

impl Shared {
    /// Unpark up to `n` parked workers. Spinning workers see the epoch on
    /// their own; a parked one either sees it in its final check before
    /// parking or has its flag seen here — the epoch bump and the flag
    /// store are both SeqCst, so one of the two orders wins.
    fn wake(&self, n: usize) {
        let mut left = n;
        if left == 0 {
            return;
        }
        for (i, flag) in self.parked.iter().enumerate() {
            if flag.load(SeqCst) {
                self.handles.get().expect("handles set in Pool::new")[i].unpark();
                left -= 1;
                if left == 0 {
                    return;
                }
            }
        }
    }

    fn wait_done(&self) {
        // SeqCst rather than Acquire: the argument in `dispatch` is about
        // the total order of SeqCst operations, and this load has to be in
        // it to be placed after the close.
        if spin_until(|| (self.barrier.active.load(SeqCst) == 0).then_some(())).is_some() {
            return;
        }
        self.barrier.waiting.store(true, SeqCst);
        if self.barrier.active.load(SeqCst) != 0 {
            // Poison-tolerant throughout: a panicking region closure leaves
            // none of the pool's own state inconsistent, so honouring a
            // poison flag here would only replace a propagating panic with
            // a confusing second one.
            let mut g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
            while self.barrier.active.load(SeqCst) != 0 {
                g = self.done.wait(g).unwrap_or_else(|e| e.into_inner());
            }
        }
        self.barrier.waiting.store(false, SeqCst);
    }

    /// Leave the region. SeqCst on both sides of the handshake: the caller
    /// stores `waiting` then loads `active`, this stores `active` then
    /// loads `waiting`, so one of the two always observes the other.
    #[inline]
    fn leave(&self) {
        if self.barrier.active.fetch_sub(1, SeqCst) == 1 && self.barrier.waiting.load(SeqCst) {
            let _g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
            self.done.notify_one();
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, SeqCst);
        for w in &self.workers {
            w.thread().unpark();
        }
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

fn worker(shared: &Shared, index: usize) {
    // The last region this worker joined; it never joins one twice.
    let mut last = 0u64;
    // Adaptive spinning. Spinning and yielding before a park only pays
    // when the next region arrives inside that window; when the previous
    // idle episode ended in a park it usually will not (a slow input
    // delivers a batch every few tens of milliseconds; pipeline stages
    // are separated by milliseconds of serial work), so a worker that
    // went cold parks at once instead of burning the yield window every
    // time. It warms back up on evidence that regions are arriving
    // faster than the park/unpark path can catch them: a park that was
    // over within `HOT_GAP`, or waking into a region already closed.
    let mut hot = true;
    // Whether the current episode's park ended before the region had
    // closed (only meaningful when the episode parked at all).
    let mut parked_for: Option<Duration> = None;
    loop {
        let epoch = loop {
            let open = || {
                let e = shared.region.epoch.load(SeqCst);
                (e.is_multiple_of(2) && e != last).then_some(e)
            };
            let seen = if hot { spin_until(open) } else { open() };
            if let Some(e) = seen {
                break e;
            }
            if shared.shutdown.load(Acquire) {
                return;
            }
            // Advertise the park before the last check; see `wake`.
            shared.parked[index].store(true, SeqCst);
            if open().is_none() && !shared.shutdown.load(SeqCst) {
                let t = Instant::now();
                thread::park();
                parked_for = Some(t.elapsed());
            }
            shared.parked[index].store(false, SeqCst);
        };
        // Join, then confirm the region is still the one we saw open: if
        // it closed in between, the caller may already be gone.
        shared.barrier.active.fetch_add(1, SeqCst);
        let joined = shared.region.epoch.load(SeqCst) == epoch;
        hot = match parked_for.take() {
            None => true,
            Some(d) => d < HOT_GAP || !joined,
        };
        if !joined {
            shared.leave();
            continue;
        }
        last = epoch;
        let ptr = shared.region.job.load(Acquire);
        CURRENT.with(|c| c.set((shared as *const Shared, index)));
        let caught = panic::catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: the join above succeeded, so the caller is blocked in
            // `wait_done` until `leave` below; the Job is live.
            let job = unsafe { &*(ptr as *const Job) };
            (job.run)(index);
        }));
        CURRENT.with(|c| c.set((std::ptr::null(), 0)));
        if let Err(p) = caught {
            let mut slot = shared.panic.lock().unwrap_or_else(|e| e.into_inner());
            if slot.is_none() {
                *slot = Some(p);
            }
            shared.panicked.store(true, SeqCst);
        }
        shared.leave();
    }
}

/// Poll `ready` until it returns `Some`: on `pause` for [`SPIN_TIME`], then
/// on `yield_now` until [`YIELD_TIME`], then give up with `None`.
///
/// The clock is only consulted every [`SPIN_QUICK`] iterations: a dispatch
/// that is already in flight resolves in tens of nanoseconds and should not
/// pay for a `Instant::now`.
#[inline]
fn spin_until<T>(ready: impl Fn() -> Option<T>) -> Option<T> {
    let mut start = None;
    loop {
        for _ in 0..SPIN_QUICK {
            if let Some(v) = ready() {
                return Some(v);
            }
            hint::spin_loop();
        }
        let start = start.get_or_insert_with(Instant::now);
        if start.elapsed() >= SPIN_TIME {
            break;
        }
    }
    let start = start.unwrap();
    loop {
        if let Some(v) = ready() {
            return Some(v);
        }
        if start.elapsed() >= YIELD_TIME {
            return None;
        }
        thread::yield_now();
    }
}

// ---------------------------------------------------------------------------
// Work distribution

/// The item cursor for one region: guided self-scheduling above
/// `32 * threads` items, one item per claim below it.
struct Cursor {
    at: AtomicUsize,
    n: usize,
    /// Guided divisor, or 0 for single-item claims.
    div: usize,
}

impl Cursor {
    fn new(n: usize, threads: usize) -> Cursor {
        // Below 32 items per participant, hand out one item at a time: the
        // items are frames, row strips or palette buckets, each costing
        // orders of magnitude more than the `fetch_add` that claims it, and
        // one-at-a-time is the best balance available for work that uneven.
        // Above it the items are individual histogram entries or grid
        // cells, where the atomic would dominate, so claims start coarse
        // and decay to single items over the tail.
        let guided = n >= 32 * threads;
        Cursor {
            at: AtomicUsize::new(0),
            n,
            div: if guided { 4 * threads } else { 0 },
        }
    }

    #[inline]
    fn claim(&self) -> Option<(usize, usize)> {
        if self.div == 0 {
            let i = self.at.fetch_add(1, Relaxed);
            return (i < self.n).then_some((i, i + 1));
        }
        let mut at = self.at.load(Relaxed);
        loop {
            if at >= self.n {
                return None;
            }
            let take = ((self.n - at) / self.div).max(1);
            match self
                .at
                .compare_exchange_weak(at, at + take, Relaxed, Relaxed)
            {
                Ok(_) => return Some((at, at + take)),
                Err(cur) => at = cur,
            }
        }
    }
}

/// Output slots for a region, one per item index. Sound because every
/// index is claimed by exactly one worker.
struct Slots<T>(*mut T);
// SAFETY: workers write disjoint indices, and the values cross threads.
unsafe impl<T: Send> Send for Slots<T> {}
unsafe impl<T: Send> Sync for Slots<T> {}

impl<T> Slots<T> {
    /// SAFETY: `i` must be in bounds and written at most once.
    #[inline]
    unsafe fn write(&self, i: usize, v: T) {
        unsafe { self.0.add(i).write(v) }
    }

    /// SAFETY: `i` must be in bounds and read at most once, and the source
    /// must not be dropped again.
    #[inline]
    unsafe fn read(&self, i: usize) -> T {
        unsafe { self.0.add(i).read() }
    }

    /// SAFETY: `i` must be in bounds, and no other live reference to that
    /// element may exist.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    unsafe fn at(&self, i: usize) -> &mut T {
        unsafe { &mut *self.0.add(i) }
    }
}

impl Pool {
    /// `f(worker, i)` for every `i` in `0..n`.
    pub fn for_each<F>(&self, n: usize, f: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        if n == 0 {
            return;
        }
        let cursor = Cursor::new(n, self.threads());
        self.dispatch(n, &|w| {
            while let Some((a, b)) = cursor.claim() {
                for i in a..b {
                    f(w, i);
                }
            }
        });
    }

    /// `f(worker, i)` for every `i` in `0..n`, collected in index order.
    pub fn map<T, F>(&self, n: usize, f: F) -> Vec<T>
    where
        T: Send,
        F: Fn(usize, usize) -> T + Sync,
    {
        let mut out = Vec::with_capacity(n);
        let slots = Slots(out.as_mut_ptr());
        self.for_each(n, |w, i| {
            // SAFETY: one worker per index, each index once.
            unsafe { slots.write(i, f(w, i)) }
        });
        // SAFETY: `for_each` ran every index, so all `n` slots are written.
        unsafe { out.set_len(n) };
        out
    }

    /// Like [`Pool::map`], with per-worker scratch built by `init` on first
    /// use. A worker that claims no work never calls `init`; one that
    /// claims many items builds it once, unlike a splitting runtime that
    /// rebuilds it per split.
    pub fn map_init<T, S, I, F>(&self, n: usize, init: I, f: F) -> Vec<T>
    where
        T: Send,
        I: Fn() -> S + Sync,
        F: Fn(&mut S, usize, usize) -> T + Sync,
    {
        if n == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(n);
        let slots = Slots(out.as_mut_ptr());
        let cursor = Cursor::new(n, self.threads());
        self.dispatch(n, &|w| {
            let mut state: Option<S> = None;
            while let Some((a, b)) = cursor.claim() {
                let state = state.get_or_insert_with(&init);
                for i in a..b {
                    // SAFETY: one worker per index, each index once.
                    unsafe { slots.write(i, f(state, w, i)) }
                }
            }
        });
        // SAFETY: every index was claimed and written.
        unsafe { out.set_len(n) };
        out
    }

    /// `f(worker, i, &mut s[i])` over the whole slice.
    pub fn for_each_mut<T, F>(&self, s: &mut [T], f: F)
    where
        T: Send,
        F: Fn(usize, usize, &mut T) + Sync,
    {
        let n = s.len();
        let slots = Slots(s.as_mut_ptr());
        self.for_each(n, |w, i| {
            // SAFETY: disjoint indices, and the borrow ends with the region.
            f(w, i, unsafe { slots.at(i) })
        });
    }

    /// `f(worker, i, item)` over `v` by value, freeing each item as it is
    /// consumed rather than at the end of the region.
    ///
    /// A panic in `f` leaks the items not yet claimed; weft treats a panic
    /// in a region as fatal, so this only shortens an aborting process's
    /// teardown.
    pub fn for_each_into<T, F>(&self, mut v: Vec<T>, f: F)
    where
        T: Send,
        F: Fn(usize, usize, T) + Sync,
    {
        let n = v.len();
        // SAFETY: the elements are moved out below; the Vec must not drop
        // them a second time. Its allocation is still freed on drop.
        unsafe { v.set_len(0) };
        let slots = Slots(v.as_mut_ptr());
        self.for_each(n, |w, i| {
            // SAFETY: each index is claimed once, so each element moves once.
            let item = unsafe { slots.read(i) };
            f(w, i, item);
        });
    }

    /// [`Pool::for_each_into`] with a result per item, in index order.
    pub fn map_into<T, U, F>(&self, mut v: Vec<T>, f: F) -> Vec<U>
    where
        T: Send,
        U: Send,
        F: Fn(usize, usize, T) -> U + Sync,
    {
        let n = v.len();
        // SAFETY: as in `for_each_into`.
        unsafe { v.set_len(0) };
        let src = Slots(v.as_mut_ptr());
        let mut out = Vec::with_capacity(n);
        let dst = Slots(out.as_mut_ptr());
        self.for_each(n, |w, i| {
            // SAFETY: each index is claimed once: one move out, one write in.
            unsafe { dst.write(i, f(w, i, src.read(i))) }
        });
        // SAFETY: every index was claimed and written.
        unsafe { out.set_len(n) };
        out
    }
}

// ---------------------------------------------------------------------------
// Parallel sort

/// Bucket bits for [`sort_entries`]: 4096 buckets keeps the count arrays
/// small while staying fine enough that a clip whose colors crowd into one
/// red slab still splits across workers.
const SORT_BITS: u32 = 12;
const SORT_BUCKETS: usize = 1 << SORT_BITS;

/// Sort `(color, count)` histogram entries exactly as `sort_unstable`
/// would, in parallel.
///
/// One MSD radix pass on the top [`SORT_BITS`] of the 24-bit color key
/// splits the entries into ordered buckets, then each bucket is sorted with
/// the ordinary comparison sort. Entries in different buckets differ in the
/// key, so bucket order is key order and concatenation is the sorted array;
/// within a bucket the comparison sort gives the full tuple ordering. The
/// result is therefore identical to a sequential sort — which matters:
/// this order is what makes the palette independent of thread count.
pub fn sort_entries(pool: &Pool, v: &mut Vec<(u32, u32)>) {
    let n = v.len();
    let t = pool.threads();
    // Below this the radix pass (two full passes over the data plus a
    // 4096-bucket histogram per chunk) costs more than it saves.
    if t == 1 || n < 1 << 14 {
        v.sort_unstable();
        return;
    }
    let nchunks = (4 * t).min(n.div_ceil(1 << 12)).max(1);
    let chunk = n.div_ceil(nchunks);
    let nchunks = n.div_ceil(chunk);
    // The exact-histogram path arrives sorted (bucket order is key order),
    // and proving it costs one streaming pass against the radix's two.
    // Chunks overlap by one element so the seams are covered.
    let sorted = pool.map(nchunks, |_, c| {
        v[c * chunk..(((c + 1) * chunk) + 1).min(n)].is_sorted()
    });
    if sorted.iter().all(|&s| s) {
        return;
    }
    // Top bits of the 24-bit color key. Clamping keeps a hypothetical
    // wider key in bounds without breaking the ordering: anything past the
    // 24-bit range is larger than every key the last bucket can hold, and
    // the per-bucket comparison sort orders them there correctly.
    let bucket = |e: &(u32, u32)| ((e.0 >> (24 - SORT_BITS)) as usize).min(SORT_BUCKETS - 1);
    let src: &[(u32, u32)] = v;

    // Per-chunk bucket counts.
    let counts: Vec<Vec<u32>> = pool.map(nchunks, |_, c| {
        let mut counts = vec![0u32; SORT_BUCKETS];
        for e in &src[c * chunk..((c + 1) * chunk).min(n)] {
            counts[bucket(e)] += 1;
        }
        counts
    });
    // Exclusive prefix over (bucket, chunk) in that order, so each chunk
    // writes its elements in order and buckets come out in key order.
    let mut offs = vec![0u32; nchunks * SORT_BUCKETS];
    let mut bucket_at = vec![0u32; SORT_BUCKETS + 1];
    let mut at = 0u32;
    for b in 0..SORT_BUCKETS {
        bucket_at[b] = at;
        for (c, counts) in counts.iter().enumerate() {
            offs[c * SORT_BUCKETS + b] = at;
            at += counts[b];
        }
    }
    bucket_at[SORT_BUCKETS] = at;
    debug_assert_eq!(at as usize, n);

    let mut out: Vec<(u32, u32)> = Vec::with_capacity(n);
    let slots = Slots(out.as_mut_ptr());
    pool.for_each(nchunks, |_, c| {
        let mut at = offs[c * SORT_BUCKETS..(c + 1) * SORT_BUCKETS].to_vec();
        for e in &src[c * chunk..((c + 1) * chunk).min(n)] {
            let b = bucket(e);
            // SAFETY: the prefix sums give every element a distinct slot in
            // `0..n`, and every slot is filled exactly once.
            unsafe { slots.write(at[b] as usize, *e) };
            at[b] += 1;
        }
    });
    // SAFETY: the scatter wrote all `n` slots.
    unsafe { out.set_len(n) };

    // Sort each bucket. `bucket_at` is ascending, so the ranges are
    // disjoint and cover the array.
    let ends = &bucket_at;
    let slots = Slots(out.as_mut_ptr());
    pool.for_each(SORT_BUCKETS, |_, b| {
        let (a, z) = (ends[b] as usize, ends[b + 1] as usize);
        if z - a > 1 {
            // SAFETY: bucket ranges are disjoint, one worker per bucket.
            unsafe { std::slice::from_raw_parts_mut(slots.at(a), z - a) }.sort_unstable();
        }
    });
    *v = out;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    fn pools() -> Vec<Pool> {
        [1, 2, 3, 8, 33].into_iter().map(Pool::new).collect()
    }

    /// Sizes around every scheduling boundary: none, one, fewer than the
    /// participants, exactly them, one more, and both sides of the switch
    /// to guided claims at `32 * threads`.
    fn sizes(t: usize) -> Vec<usize> {
        vec![
            0,
            1,
            2,
            t - 1,
            t,
            t + 1,
            7,
            32 * t - 1,
            32 * t,
            32 * t + 1,
            1000,
            100_000,
        ]
    }

    #[test]
    fn map_preserves_order() {
        for p in pools() {
            for n in sizes(p.threads()) {
                let got = p.map(n, |w, i| {
                    assert!(w < p.threads());
                    i * 3
                });
                assert_eq!(got, (0..n).map(|i| i * 3).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn every_item_runs_once() {
        for p in pools() {
            for n in sizes(p.threads()) {
                let seen: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
                p.for_each(n, |_, i| {
                    seen[i].fetch_add(1, Relaxed);
                });
                assert!(seen.iter().all(|c| c.load(Relaxed) == 1));
            }
        }
    }

    #[test]
    fn map_init_builds_state_per_worker() {
        for p in pools() {
            let inits = AtomicU32::new(0);
            let got = p.map_init(
                5000,
                || {
                    inits.fetch_add(1, Relaxed);
                    Vec::<usize>::with_capacity(4)
                },
                |st: &mut Vec<usize>, _, i| {
                    st.push(i);
                    st.len()
                },
            );
            assert_eq!(got.len(), 5000);
            assert!(got.iter().all(|&v| v >= 1));
            assert!(inits.load(Relaxed) as usize <= p.threads());
        }
    }

    #[test]
    fn into_variants_move_items() {
        for p in pools() {
            let v: Vec<String> = (0..500).map(|i| i.to_string()).collect();
            let got = p.map_into(v, |_, i, s| {
                assert_eq!(s, i.to_string());
                s.len()
            });
            assert_eq!(got.len(), 500);
            let v: Vec<Box<u32>> = (0..500).map(|i| Box::new(i as u32)).collect();
            let sum = AtomicU32::new(0);
            p.for_each_into(v, |_, _, b| {
                sum.fetch_add(*b, Relaxed);
            });
            assert_eq!(sum.load(Relaxed), (0..500).sum::<u32>());
        }
    }

    #[test]
    fn for_each_mut_writes_each_element() {
        for p in pools() {
            let mut v: Vec<usize> = vec![0; 3000];
            p.for_each_mut(&mut v, |_, i, slot| *slot = i * 2);
            assert!(v.iter().enumerate().all(|(i, &x)| x == i * 2));
        }
    }

    #[test]
    fn sort_matches_sequential() {
        let mut state = 0x243f_6a88_85a3_08d3u64;
        let mut rng = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for p in pools() {
            for n in [0usize, 1, 100, 20_000, 200_000] {
                // A red-heavy distribution: most entries in one bucket.
                let v: Vec<(u32, u32)> = (0..n)
                    .map(|_| {
                        let r = rng();
                        // 24-bit keys, a red-heavy skew, and a few wider
                        // ones to pin the clamped bucket down.
                        let key = match r % 8 {
                            0 => (r >> 33) as u32,
                            1..=5 => (r & 0xffff) as u32,
                            _ => (r & 0xff_ffff) as u32,
                        };
                        (key, (r >> 40) as u32)
                    })
                    .collect();
                let mut want = v.clone();
                want.sort_unstable();
                let mut got = v;
                sort_entries(&p, &mut got);
                assert_eq!(got, want, "n={n} threads={}", p.threads());
                // Already-sorted input takes the scan fast path.
                sort_entries(&p, &mut got);
                assert_eq!(got, want, "resort n={n} threads={}", p.threads());
            }
        }
    }

    #[test]
    fn nested_region_runs_inline() {
        let p = Pool::new(4);
        let got = p.map(64, |_, i| {
            let inner: Vec<usize> = p.map(4, |_, j| i * 4 + j);
            inner.iter().sum::<usize>()
        });
        assert_eq!(got.len(), 64);
        assert_eq!(got[0], 1 + 2 + 3);
    }

    #[test]
    fn panic_propagates_to_the_caller() {
        let p = Pool::new(4);
        // Whether the panicking item lands on the submitting thread or on a
        // worker depends on timing, and the two leave `dispatch` by
        // different paths, so repeat until both have certainly happened.
        for item in [0usize, 63, 99] {
            for _ in 0..8 {
                let r = panic::catch_unwind(AssertUnwindSafe(|| {
                    p.for_each(100, |_, i| {
                        if i == item {
                            panic!("boom");
                        }
                    });
                }));
                assert!(r.is_err(), "item={item}");
                // The pool still works, and nothing leaks into the region
                // after it: a lock held across the unwind, or a worker's
                // payload left in the slot, would surface right here.
                assert_eq!(p.map(10, |_, i| i), (0..10).collect::<Vec<_>>());
            }
        }
        // Every participant panicking at once is fine too.
        let r = panic::catch_unwind(AssertUnwindSafe(|| p.for_each(100, |_, _| panic!("all"))));
        assert!(r.is_err());
        assert_eq!(p.map(10, |_, i| i), (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn panic_in_init_propagates() {
        let p = Pool::new(4);
        let r = panic::catch_unwind(AssertUnwindSafe(|| {
            p.map_init(100, || -> u32 { panic!("init") }, |_, _, i| i)
        }));
        assert!(r.is_err());
        assert_eq!(p.map(10, |_, i| i), (0..10).collect::<Vec<_>>());
    }

    /// Every item handed to a by-value region is dropped exactly once, and
    /// so is every result: no double drop, no leak.
    #[test]
    fn items_move_and_drop_exactly_once() {
        use std::sync::atomic::AtomicUsize;
        static LIVE: AtomicUsize = AtomicUsize::new(0);
        struct Counted(u32);
        impl Counted {
            fn new(v: u32) -> Counted {
                LIVE.fetch_add(1, SeqCst);
                Counted(v)
            }
        }
        impl Drop for Counted {
            fn drop(&mut self) {
                LIVE.fetch_sub(1, SeqCst);
            }
        }
        for p in pools() {
            for n in [0usize, 1, 50, 5000] {
                let v: Vec<Counted> = (0..n as u32).map(Counted::new).collect();
                assert_eq!(LIVE.load(SeqCst), n);
                let out: Vec<Counted> = p.map_into(v, |_, i, c| {
                    assert_eq!(c.0 as usize, i);
                    Counted::new(c.0 + 1)
                });
                // the inputs are gone, the outputs are live
                assert_eq!(LIVE.load(SeqCst), n);
                assert!(out.iter().enumerate().all(|(i, c)| c.0 as usize == i + 1));
                p.for_each_into(out, |_, i, c| assert_eq!(c.0 as usize, i + 1));
                assert_eq!(LIVE.load(SeqCst), 0);
                let out: Vec<Counted> = p.map(n, |_, i| Counted::new(i as u32));
                assert_eq!(LIVE.load(SeqCst), n);
                drop(out);
                assert_eq!(LIVE.load(SeqCst), 0);
            }
        }
    }

    /// Regions separated by gaps long enough for every worker to park:
    /// results stay correct and the workers do come back and take part.
    #[test]
    fn parked_workers_wake_and_join() {
        for t in [2, 4, 9] {
            let p = Pool::new(t);
            let joined: Vec<AtomicU32> = (0..t).map(|_| AtomicU32::new(0)).collect();
            for round in 0..20 {
                thread::sleep(Duration::from_millis(1));
                let got = p.map(4 * t, |w, i| {
                    joined[w].fetch_add(1, Relaxed);
                    // long enough that workers waking from a futex still
                    // find something to claim
                    thread::sleep(Duration::from_micros(300));
                    i * round
                });
                assert_eq!(got, (0..4 * t).map(|i| i * round).collect::<Vec<_>>());
            }
            let workers_seen = joined[..t - 1]
                .iter()
                .filter(|c| c.load(Relaxed) > 0)
                .count();
            assert!(
                workers_seen == t - 1,
                "threads={t}: only {workers_seen} workers ever joined"
            );
        }
    }

    /// Publishes arriving right around the moment workers commit to
    /// parking, many times over: a lost wake-up would hang this test.
    #[test]
    fn publish_races_with_parking() {
        let p = Pool::new(6);
        let mut state = 0x1234_5678_9abc_def1u64;
        for round in 0..1500u64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // jitter the gap across the spin, yield and park windows
            let gap = state % 400;
            thread::sleep(Duration::from_micros(gap));
            let got = p.map(30, |_, i| i as u64 + round);
            assert!(got.iter().enumerate().all(|(i, &v)| v == i as u64 + round));
        }
    }

    /// Regions so short the caller closes them almost immediately: workers
    /// that join late must back out cleanly, and the counts must balance.
    #[test]
    fn tiny_regions_exercise_late_joiners() {
        let p = Pool::new(8);
        let hits = AtomicU32::new(0);
        for _ in 0..50_000 {
            p.for_each(1, |_, _| {
                hits.fetch_add(1, Relaxed);
            });
        }
        assert_eq!(hits.load(Relaxed), 50_000);
        assert_eq!(p.map(100, |_, i| i), (0..100).collect::<Vec<_>>());
    }

    /// Several threads submitting to one pool at once queue behind each
    /// other; each sees exactly its own results.
    #[test]
    fn concurrent_submitters_serialize() {
        let p = Pool::new(5);
        thread::scope(|scope| {
            for k in 0..4usize {
                let p = &p;
                scope.spawn(move || {
                    for round in 0..300usize {
                        let n = 1 + (round * 7 + k) % 90;
                        let got = p.map(n, |_, i| i * 1000 + k);
                        assert_eq!(got, (0..n).map(|i| i * 1000 + k).collect::<Vec<_>>());
                    }
                });
            }
        });
    }

    /// Two pools driven from two threads at the same time, the way the
    /// hold pool runs alongside the global one.
    #[test]
    fn pools_run_independently() {
        let a = Pool::new(3);
        let b = Pool::new(4);
        thread::scope(|scope| {
            for (p, tag) in [(&a, 1usize), (&b, 2usize)] {
                scope.spawn(move || {
                    for round in 0..500usize {
                        let got = p.map(64, |_, i| i * tag + round);
                        assert_eq!(got, (0..64).map(|i| i * tag + round).collect::<Vec<_>>());
                    }
                });
            }
        });
    }

    /// After a long idle (workers cold and parking at once), a burst of
    /// back-to-back regions must warm every worker back up: the first
    /// region or two may be missed, the rest must not be.
    #[test]
    fn cold_workers_warm_up_for_a_burst() {
        for t in [3, 8] {
            let p = Pool::new(t);
            for _ in 0..3 {
                thread::sleep(Duration::from_millis(3));
                p.for_each(t, |_, _| thread::sleep(Duration::from_micros(50)));
            }
            thread::sleep(Duration::from_millis(3));
            let joined: Vec<AtomicU32> = (0..t).map(|_| AtomicU32::new(0)).collect();
            for _ in 0..2000 {
                p.for_each(4 * t, |w, _| {
                    joined[w].fetch_add(1, Relaxed);
                    thread::sleep(Duration::from_micros(20));
                });
            }
            let cold: Vec<usize> = (0..t - 1)
                .filter(|&w| joined[w].load(Relaxed) < 100)
                .collect();
            assert!(
                cold.is_empty(),
                "threads={t}: workers {cold:?} stayed cold through the burst"
            );
        }
    }

    /// Dropping a pool whose workers are parked, or mid-spin, joins them.
    #[test]
    fn drop_joins_workers_in_any_state() {
        let p = Pool::new(6);
        p.for_each(6, |_, _| {});
        drop(p); // spinning
        let p = Pool::new(6);
        p.for_each(6, |_, _| {});
        thread::sleep(Duration::from_millis(2));
        drop(p); // parked
        let p = Pool::new(6);
        drop(p); // never ran a region
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use std::time::Instant;

    /// Region open-to-close latency with nothing to do, back to back
    /// (workers spinning) and after a pause (workers parked). The hot
    /// figure is what every one of weft's ~100 regions per clip pays on
    /// top of its work; the parked figure is dominated by this machine's
    /// futex wake latency and the caller no longer waits for it.
    /// `cargo test --release pool::bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn region_latency() {
        for t in [2, 4, 8, 16, 22, 44] {
            let p = Pool::new(t);
            p.for_each(t, |_, _| {});
            let n = 20_000;
            let t0 = Instant::now();
            for _ in 0..n {
                p.for_each(t, |_, _| {});
            }
            let hot = t0.elapsed() / n;
            let n = 300;
            let mut cold = Duration::ZERO;
            for _ in 0..n {
                thread::sleep(Duration::from_micros(300));
                let t0 = Instant::now();
                p.for_each(t, |_, _| {});
                cold += t0.elapsed();
            }
            println!(
                "threads={t:2}  hot {hot:?}/region   parked {:?}/region",
                cold / n
            );
        }
    }
}
