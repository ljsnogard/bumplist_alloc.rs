//! `denco_arena` 特性的集成测试。
//!
//! Integration tests for the `denco_arena` feature.
#![cfg(feature = "denco_arena")]

use core::{
    future::Future,
    pin::pin,
    sync::atomic::{AtomicUsize, Ordering},
    task::{Context, Poll, Waker},
};

use std::sync::Arc;

use abs_denco::de::{TrArena, TrArenaBytes};
use bumplist_alloc::{BumpListAlloc, PageSize};

/// 立即完成的测试执行器：两个 arena 入口都返回 `Ready`。
fn block_on_<TyFut>(future: TyFut) -> TyFut::Output
where
    TyFut: Future,
{
    let mut cx = Context::from_waker(Waker::noop());
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(output) => return output,
            Poll::Pending => core::hint::spin_loop(),
        }
    }
}

/// 测试 `TrArena::alloc_value_async` 返回一个活到 allocator 析构的可变引用。
/// Test that `TrArena::alloc_value_async` yields a mutable reference that lives as long as
/// the allocator.
///
/// - 手段：在 4 KiB 页的 `BumpListAlloc` 上分配一个 `u32`，通过返回的 `&mut u32` 修改它。
/// - 判断：分配成功，修改后的值读回为预期值；引用没有在分配调用结束时失效。
///
/// - Means: allocate a `u32` on a 4 KiB-page `BumpListAlloc` and mutate it through the
///   returned `&mut u32`.
/// - Verdict: allocation succeeds and the mutated value reads back as expected; the
///   reference has not expired at the end of the allocation call.
#[test]
fn arena_value_outlives_call_() {
    let bump = BumpListAlloc::new(PageSize::P4K).expect("4 KiB 页应当受支持");
    let mut value = block_on_(bump.alloc_value_async(41u32)).expect("arena 分配应当成功");
    *value += 1;
    assert_eq!(*value, 42u32);
}

/// 测试 `TrArenaBytes::alloc_bytes_async` 把源字节拷进 arena。
/// Test that `TrArenaBytes::alloc_bytes_async` copies the source bytes into the arena.
///
/// - 手段：把 `b"hello"` 分配给 arena，再与源切片比较。
/// - 判断：返回切片内容与源完全相同，长度一致。
///
/// - Means: allocate `b"hello"` into the arena and compare it with the source slice.
/// - Verdict: the returned slice has the same contents and length as the source.
#[test]
fn arena_bytes_copy_() {
    let bump = BumpListAlloc::new(PageSize::P4K).expect("4 KiB 页应当受支持");
    let bytes = block_on_(bump.alloc_bytes_async(b"hello")).expect("arena 分配应当成功");
    assert_eq!(bytes, b"hello");
}

/// 测试 `BumpListBox` 在析构时运行 `T` 的析构函数。
/// Test that `BumpListBox` runs `T`'s destructor when it is dropped.
///
/// - 手段：在 arena 里分配一个带 `Drop` 计数的值，先检查计数为 0，再显式 `drop` 句柄。
/// - 判断：drop 前计数为 0，drop 后计数恰好为 1，说明析构函数被执行且只执行一次。
///
/// - Means: allocate a value with a `Drop` counter, check the counter is 0, then drop the
///   handle explicitly.
/// - Verdict: the counter is 0 before the drop and exactly 1 afterwards, proving the
///   destructor ran exactly once.
#[test]
fn arena_value_runs_drop_() {
    struct DropCounter(Arc<AtomicUsize>);

    impl Drop for DropCounter {
        fn drop(&mut self) {
            self.0.fetch_add(1usize, Ordering::SeqCst);
        }
    }

    let count = Arc::new(AtomicUsize::new(0usize));
    let bump = BumpListAlloc::new(PageSize::P4K).expect("4 KiB 页应当受支持");
    let value =
        block_on_(bump.alloc_value_async(DropCounter(count.clone()))).expect("arena 分配应当成功");
    assert_eq!(count.load(Ordering::SeqCst), 0usize);
    drop(value);
    assert_eq!(count.load(Ordering::SeqCst), 1usize);
}
