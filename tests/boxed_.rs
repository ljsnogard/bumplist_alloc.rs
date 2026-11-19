//! `Box<'a, T>` 公开 API 的集成测试。
//!
//! Integration tests for the public API of `Box<'a, T>`.

use core::{
    alloc::{AllocError, Allocator, Layout},
    borrow::{Borrow, BorrowMut},
    cell::Cell,
    error::Error,
    fmt,
    future::Future,
    hash::{Hash, Hasher},
    iter::FusedIterator,
    marker::PhantomPinned,
    mem,
    pin::Pin,
    ptr::NonNull,
    task::{Context, Poll, Waker},
};
use std::{
    alloc::Global,
    collections::hash_map::DefaultHasher,
    string::String,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use bumplist_alloc::{Box, BumpListAlloc, PageSize};

/// 在任意 `BumpListAlloc<A>` 中分配一个 `T`，返回拥有它的 `Box`。
fn in_arena_with_<'a, A: Allocator + Send, T>(arena: &'a BumpListAlloc<A>, value: T) -> Box<'a, T> {
    let block = <&BumpListAlloc<A> as Allocator>::allocate(&arena, Layout::new::<T>())
        .expect("arena 分配应当成功");
    // SAFETY: `block` 来自 `arena` 且按 `Layout::new::<T>()` 分配；这里写入一次并
    // 立即交给 `Box` 独占。
    unsafe {
        let raw = block.as_ptr().cast::<T>();
        raw.write(value);
        Box::from_raw(raw)
    }
}

/// 在默认 arena（`A = Global`）中分配一个 `T`。
fn in_arena_<'a, T>(arena: &'a BumpListAlloc, value: T) -> Box<'a, T> {
    in_arena_with_(arena, value)
}

fn hash_of_<H: Hash>(value: &H) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

fn assert_unpin_<T: Unpin>() {}

fn assert_send_<T: Send>() {}

fn assert_send_sync_<T: Send + Sync>() {}

fn assert_fused_<I: FusedIterator>(_: &I) {}

/// 测试目标：`Box` 转发格式化、比较、哈希、借用与 `Hasher`。
/// Test goal: `Box` forwards formatting, comparison, hashing, borrow traits and `Hasher`.
///
/// - 手段：在 4 KiB 页 arena 中装箱两个 `u32`，分别使用各格式化 trait、比较运算符、
///   `Hash` / `Hasher` 以及 `AsRef` / `AsMut` / `Borrow` / `BorrowMut`。
/// - 判定：格式化、比较、哈希结果都与裸 `u32` 一致；借用/可变借用得到同一个值。
///
/// - Means: box two `u32` values in a 4 KiB-page arena and exercise every formatting
///   trait, comparison operators, `Hash` / `Hasher`, and the borrow traits.
/// - Verdict: formatting, comparison and hashing match the bare `u32`; shared and
///   mutable borrows observe the same value.
#[test]
fn forwarding_traits_() {
    let arena = BumpListAlloc::new(PageSize::P4K).expect("4 KiB 页应当受支持");
    let a = in_arena_(&arena, 5u32);
    let b = in_arena_(&arena, 7u32);

    assert_eq!(format!("{a}"), "5");
    assert_eq!(format!("{a:?}"), "5");
    assert_eq!(format!("{a:b}"), "101");
    assert_eq!(format!("{a:o}"), "5");
    assert_eq!(format!("{a:x}"), "5");
    assert_eq!(format!("{a:X}"), "5");
    assert_eq!(format!("{a:e}"), "5e0");
    assert_eq!(format!("{a:E}"), "5E0");
    assert_eq!(format!("{:p}", a), format!("{:p}", &*a as *const u32));

    assert!(a < b);
    assert_eq!(a, in_arena_(&arena, 5u32));
    assert_eq!(hash_of_(&a), hash_of_(&5u32));

    let mut plain = DefaultHasher::new();
    plain.write(b"abc");
    let mut boxed_hasher = in_arena_(&arena, DefaultHasher::new());
    boxed_hasher.write(b"abc");
    assert_eq!(plain.finish(), boxed_hasher.finish());

    let shared: &u32 = a.as_ref();
    let borrowed: &u32 = a.borrow();
    assert_eq!((*shared, *borrowed), (5, 5));

    let mut mutable = in_arena_(&arena, 5u32);
    *mutable.as_mut() = 6;
    *mutable.borrow_mut() = 7;
    assert_eq!(*mutable, 7);
}

/// 测试目标：裸指针相关方法能往返转换且不破坏所有权。
/// Test goal: The raw-pointer methods round-trip without breaking ownership.
///
/// - 手段：依次使用 `into_raw` / `from_raw`、`into_non_null` / `from_non_null`、
///   `as_ptr` / `as_mut_ptr` / `as_non_null` 读写同一个 arena 块。
/// - 判定：每一步读回的值符合预期，最后一次修改写入原块。
///
/// - Means: exercise `into_raw` / `from_raw`, `into_non_null` / `from_non_null`, and
///   `as_ptr` / `as_mut_ptr` / `as_non_null` on one arena block.
/// - Verdict: every step reads the expected value and the final write lands in the
///   original block.
#[test]
fn raw_pointer_roundtrip_() {
    let arena = BumpListAlloc::new(PageSize::P4K).expect("4 KiB 页应当受支持");

    let boxed = in_arena_(&arena, 11u32);
    let raw = Box::into_raw(boxed);
    assert_eq!(unsafe { *raw }, 11);

    let mut boxed = unsafe { Box::from_raw(raw) };
    *boxed += 1;

    let ptr = Box::into_non_null(boxed);
    assert_eq!(unsafe { *ptr.as_ptr() }, 12);

    let mut boxed = unsafe { Box::from_non_null(ptr) };
    assert_eq!(unsafe { *Box::as_ptr(&boxed) }, 12);

    unsafe { *Box::as_mut_ptr(&mut boxed) = 13 };
    let ptr = Box::as_non_null(&mut boxed);
    assert_eq!(unsafe { *ptr.as_ptr() }, 13);
    assert_eq!(*boxed, 13);
}

/// 测试目标：`leak`、`into_pin`、`From<Box> for Pin<Box>` 与无条件 `Unpin`。
/// Test goal: `leak`, `into_pin`, `From<Box> for Pin<Box>` and unconditional `Unpin`.
///
/// - 手段：泄漏一个 `u32` 并修改；把两个 `u32` 分别经 `into_pin` 和 `Pin::from`
///   转换；对 `PhantomPinned` 的 `Box` 做编译期 `Unpin` 断言。
/// - 判定：泄漏引用可写；两个 `Pin<Box<u32>>` 都能解引用到原值；`Box<PhantomPinned>`
///   仍满足 `Unpin`。
///
/// - Means: leak and mutate a `u32`; convert two boxes with `into_pin` and `Pin::from`;
///   assert `Unpin` for a `Box` of `PhantomPinned`.
/// - Verdict: the leaked reference is writable, both pinned boxes deref to the original
///   value, and `Box<PhantomPinned>` is still `Unpin`.
#[test]
fn leak_and_pin_() {
    let arena = BumpListAlloc::new(PageSize::P4K).expect("4 KiB 页应当受支持");

    let mut boxed = in_arena_(&arena, 21u32);
    let ptr = Box::as_non_null(&mut boxed);
    let leaked: &mut u32 = Box::leak(boxed);
    *leaked += 1;
    assert_eq!(*leaked, 22);
    // SAFETY: `ptr` 是同一块在泄漏前保存的指针；重新接管并析构，避免测试遗留活块。
    drop(unsafe { Box::from_non_null(ptr) });

    let pinned = Box::into_pin(in_arena_(&arena, 5u32));
    assert_eq!(*pinned, 5);

    let pinned_from = Pin::from(in_arena_(&arena, 6u32));
    assert_eq!(*pinned_from, 6);

    assert_unpin_::<Box<'static, PhantomPinned>>();
}

struct DropGuard(Arc<AtomicUsize>);

impl Drop for DropGuard {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

trait Named {
    fn name(&self) -> &'static str;
}

impl Named for DropGuard {
    fn name(&self) -> &'static str {
        "guard"
    }
}

/// 测试目标：`Box<'a, Concrete>` 可以 unsize 成 `Box<'a, dyn Trait>`，且析构正确。
/// Test goal: `Box<'a, Concrete>` unsizes to `Box<'a, dyn Trait>` and still drops the
/// concrete value exactly once.
///
/// - 手段：把带 `Drop` 计数的具体类型装箱后强制转换成 `Box<dyn Named>`，调用 trait
///   方法并离开作用域。
/// - 判定：trait 方法返回具体实现的结果；强制转换后的 `Box` 析构时计数恰好为 1。
///
/// - Means: box a concrete type with a drop counter, coerce it to `Box<dyn Named>`, call
///   the trait method and leave scope.
/// - Verdict: the trait method returns the concrete implementation's value and the drop
///   counter is exactly 1 after the coerced box goes away.
#[test]
fn coerce_to_trait_object_() {
    let drops = Arc::new(AtomicUsize::new(0));
    let arena = BumpListAlloc::new(PageSize::P4K).expect("4 KiB 页应当受支持");
    {
        let boxed = in_arena_(&arena, DropGuard(Arc::clone(&drops)));
        let dynamic: Box<dyn Named> = boxed;
        assert_eq!(dynamic.name(), "guard");
    }
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

/// 测试目标：`Box` 转发 `Iterator` / `DoubleEndedIterator` / `ExactSizeIterator` /
/// `FusedIterator`。
/// Test goal: `Box` forwards the iterator traits.
///
/// - 手段：分别装箱一个 `Once<u8>` 和一个 `Empty<u8>`，调用 `next`、`size_hint`、
///   `next_back`、`len`，并对 `FusedIterator` 做编译期断言。
/// - 判定：迭代结果与直接使用内层迭代器一致，空迭代器长度为零。
///
/// - Means: box a `Once<u8>` and an `Empty<u8>`, then call `next`, `size_hint`,
///   `next_back` and `len`, plus a compile-time `FusedIterator` assertion.
/// - Verdict: iteration matches the inner iterator and the empty one reports length 0.
#[test]
fn iterator_forwarding_() {
    let arena = BumpListAlloc::new(PageSize::P4K).expect("4 KiB 页应当受支持");

    let mut once = in_arena_(&arena, core::iter::once(3u8));
    assert_eq!(once.size_hint(), (1, Some(1)));
    assert_eq!(once.next(), Some(3));
    assert_eq!(once.next(), None);

    let mut empty = in_arena_(&arena, core::iter::empty::<u8>());
    assert_eq!(empty.len(), 0);
    assert_eq!(empty.next_back(), None);
    assert_fused_(&empty);
}

/// 测试目标：`Box` 转发 `Future`。
/// Test goal: `Box` forwards `Future`.
///
/// - 手段：装箱一个立即完成的 `Ready<u8>`，用 noop waker 手动 poll 一次。
/// - 判定：poll 返回 `Ready(9)`。
///
/// - Means: box an immediately-ready `Ready<u8>` and poll it once with a noop waker.
/// - Verdict: the poll returns `Ready(9)`.
#[test]
fn future_forwarding_() {
    let arena = BumpListAlloc::new(PageSize::P4K).expect("4 KiB 页应当受支持");
    let mut future = in_arena_(&arena, core::future::ready(9u8));
    let mut context = Context::from_waker(Waker::noop());
    assert_eq!(
        Future::poll(Pin::new(&mut future), &mut context),
        Poll::Ready(9u8)
    );
}

#[derive(Debug)]
struct Inner;

impl fmt::Display for Inner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("inner")
    }
}

impl Error for Inner {}

#[derive(Debug)]
struct Outer(Inner);

impl fmt::Display for Outer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("outer")
    }
}

impl Error for Outer {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.0)
    }
}

/// 测试目标：`Box` 转发 `Error`。
/// Test goal: `Box` forwards `Error`.
///
/// - 手段：装箱一个带 `source` 的错误，分别调用 `Display` 与 `Error::source`。
/// - 判定：`Display` 输出外层错误文本，`source` 返回内层错误。
///
/// - Means: box an error with a source and call `Display` and `Error::source`.
/// - Verdict: `Display` renders the outer text and `source` returns the inner error.
#[test]
fn error_forwarding_() {
    let arena = BumpListAlloc::new(PageSize::P4K).expect("4 KiB 页应当受支持");
    let boxed = in_arena_(&arena, Outer(Inner));
    assert_eq!(format!("{boxed}"), "outer");
    assert!(format!("{boxed:?}").contains("Outer"));
    assert!(Error::source(&boxed).is_some());
}

/// 测试目标：非零尺寸值的 `Clone` 在新块中复制，原值互不影响。
/// Test goal: `Clone` copies a non-zero-sized value into a new block, independent of the
/// original.
///
/// - 手段：分别克隆一个 `u32` 与一个 `String`，修改克隆的 `u32`。
/// - 判定：克隆值内容相同；修改克隆不影响原值。
///
/// - Means: clone a `u32` and a `String`, then mutate the cloned `u32`.
/// - Verdict: both copies start equal and mutating the clone leaves the original intact.
#[test]
fn clone_non_zst_() {
    let arena = BumpListAlloc::new(PageSize::P4K).expect("4 KiB 页应当受支持");

    let original = in_arena_(&arena, 5u32);
    let mut copy = original.clone();
    *copy += 1;
    assert_eq!(*original, 5);
    assert_eq!(*copy, 6);

    let text = in_arena_(&arena, String::from("hello"));
    let text_copy = text.clone();
    assert_eq!(*text, "hello");
    assert_eq!(*text_copy, "hello");
}

/// 测试目标：`Clone` 在当前页放不下时会向同一 arena 申请新页。
/// Test goal: `Clone` asks the same arena for a new page when the current page is full.
///
/// - 手段：1 KiB 页 arena 中装箱一个 600 字节数组，再克隆一份。
/// - 判定：两份内容都等于原数组，且没有越界或破坏块链。
///
/// - Means: box a 600-byte array in a 1 KiB-page arena and clone it.
/// - Verdict: both copies match the original array without corrupting the block chain.
#[test]
fn clone_spills_to_new_page_() {
    let arena = BumpListAlloc::new(PageSize::P1K).expect("1 KiB 页应当受支持");
    let original = in_arena_(&arena, [7u8; 600]);
    let copy = original.clone();
    assert_eq!(*original, [7u8; 600]);
    assert_eq!(*copy, [7u8; 600]);
}

static ZST_CLONES: AtomicUsize = AtomicUsize::new(0);
static ZST_DROPS: AtomicUsize = AtomicUsize::new(0);

struct ZstDrop;

impl Clone for ZstDrop {
    fn clone(&self) -> Self {
        ZST_CLONES.fetch_add(1, Ordering::SeqCst);
        Self
    }
}

impl Drop for ZstDrop {
    fn drop(&mut self) {
        ZST_DROPS.fetch_add(1, Ordering::SeqCst);
    }
}

/// 测试目标：零尺寸值的 `Clone` 不分配块，且克隆值与原值各析构一次。
/// Test goal: cloning a zero-sized value allocates no block and both values drop exactly
/// once.
///
/// - 手段：装箱一个 ZST `ZstDrop`，克隆一次，记录 clone/Drop 计数后依次析构两份。
/// - 判定：`clone` 计数为 1；析构计数最终为 2，说明没有多析构或漏析构。
///
/// - Means: box a ZST `ZstDrop`, clone it once, then drop both handles while counting
///   clones and drops.
/// - Verdict: the clone count is 1 and the drop count is exactly 2, so neither dropping
///   is duplicated nor skipped.
#[test]
fn clone_zst_() {
    ZST_CLONES.store(0, Ordering::SeqCst);
    ZST_DROPS.store(0, Ordering::SeqCst);
    assert_eq!(mem::size_of::<ZstDrop>(), 0);

    let arena = BumpListAlloc::new(PageSize::P4K).expect("4 KiB 页应当受支持");
    let original = in_arena_(&arena, ZstDrop);
    let copy = original.clone();
    assert_eq!(ZST_CLONES.load(Ordering::SeqCst), 1);

    drop(original);
    drop(copy);
    assert_eq!(ZST_DROPS.load(Ordering::SeqCst), 2);
}

/// 测试目标：`Box<'a, T>` 在 `T: Send + Sync` 时实现 `Send + Sync`。
/// Test goal: `Box<'a, T>` is `Send + Sync` when `T: Send + Sync`.
///
/// - 手段：用编译期 trait 断言检查 `Box<'static, u32>` 与 `Box<'static, dyn Trait>`
///   的自动/手写实现。
/// - 判定：`u32` 的 `Box` 满足 `Send + Sync`；只带 `Send` 的 trait object 只满足
///   `Send`，带 `Sync` 的 trait object 同时满足两者。
///
/// - Means: use compile-time trait assertions on `Box<'static, u32>` and boxed trait
///   objects.
/// - Verdict: the `u32` box is both `Send` and `Sync`; a `Send`-only trait object is only
///   `Send`, while a `Send + Sync` trait object satisfies both.
#[test]
fn box_is_send_sync_() {
    trait SendOnly: Send {}
    trait SendSync: Send + Sync {}

    assert_send_sync_::<Box<'static, u32>>();
    assert_send_::<Box<'static, dyn SendOnly>>();
    assert_send_sync_::<Box<'static, dyn SendSync>>();
}

/// 测试目标：`Box` 可以把块的所有权交给另一个线程并在线程内析构。
/// Test goal: a `Box` can move its block ownership to another thread and drop it there.
///
/// - 手段：在 4 KiB 页 arena 中装箱一个 `u32`，用 `std::thread::scope` 把它移动到
///   子线程，在子线程读取并析构。
/// - 判定：子线程读到原值；离开作用域后 arena 正常析构，没有块泄漏。
///
/// - Means: box a `u32` in a 4 KiB-page arena, move it into a scoped child thread, read
///   and drop it there.
/// - Verdict: the child thread reads the original value and the arena drops cleanly.
#[test]
fn box_moves_across_threads_() {
    let arena = BumpListAlloc::new(PageSize::P4K).expect("4 KiB 页应当受支持");
    let boxed = in_arena_(&arena, 5u32);

    std::thread::scope(|scope| {
        scope.spawn(move || {
            assert_eq!(*boxed, 5);
            drop(boxed);
        });
    });
}

/// 测试目标：`&Box` 可以跨线程共享，且多个线程可以同时克隆同一个块。
/// Test goal: `&Box` is shareable across threads and several threads may clone the same
/// block concurrently.
///
/// - 手段：装箱一个 `u32`，在 scoped 子线程中读取 `&Box` 并克隆一份，克隆后再改值。
/// - 判定：子线程读到原值；克隆得到独立块，修改克隆不影响原值。
///
/// - Means: box a `u32`, read the `&Box` from a scoped child thread, clone it there and
///   mutate the clone.
/// - Verdict: the child observes the original value and mutating the clone leaves the
///   original untouched.
#[test]
fn box_shared_across_threads_() {
    let arena = BumpListAlloc::new(PageSize::P4K).expect("4 KiB 页应当受支持");
    let boxed = in_arena_(&arena, 5u32);

    std::thread::scope(|scope| {
        scope.spawn(|| {
            assert_eq!(*boxed, 5);
            let mut copy = boxed.clone();
            *copy += 1;
            assert_eq!(*copy, 6);
        });
    });

    assert_eq!(*boxed, 5);
}

/// 只用 `Cell` 计数的 `Send + !Sync` 底层分配器，验证锁可以把 A 的访问降到 `Send`。
struct CellAlloc(Cell<u8>);

unsafe impl Allocator for CellAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.0.set(self.0.get().wrapping_add(1));
        Global.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { Global.deallocate(ptr, layout) }
    }
}

/// 测试目标：底层分配器是 `Send + !Sync` 时，`BumpListAlloc` 与 `Box` 仍可跨线程使用。
/// Test goal: `BumpListAlloc` and `Box` remain usable across threads even when the
/// backing allocator is `Send + !Sync`.
///
/// - 手段：用内部含 `Cell` 的 `Send + !Sync` 分配器构造 1 KiB 页 arena，编译期断言
///   `&BumpListAlloc<CellAlloc>: Send + Sync`；装箱 600 字节数组后移动到子线程克隆，
///   让 A::allocate / deallocate 在另一线程发生。
/// - 判定：`&arena` 满足 `Send + Sync`；子线程克隆出的内容与原值一致；没有数据竞争。
///
/// - Means: build a 1 KiB-page arena with a `Send + !Sync` allocator containing a
///   `Cell`, assert `&BumpListAlloc<CellAlloc>: Send + Sync`, move a 600-byte array box
///   to a child thread and clone it there so A is called from another thread.
/// - Verdict: `&arena` is `Send + Sync`, the clone matches the original, and no data race
///   is reported.
#[test]
fn send_not_sync_arena_() {
    assert_send_sync_::<&BumpListAlloc<CellAlloc>>();

    let arena =
        BumpListAlloc::new_in(CellAlloc(Cell::new(0)), PageSize::P1K).expect("1 KiB 页应当受支持");
    let boxed = in_arena_with_(&arena, [9u8; 600]);

    std::thread::scope(|scope| {
        scope.spawn(move || {
            let copy = boxed.clone();
            assert_eq!(*copy, [9u8; 600]);
        });
    });
}
