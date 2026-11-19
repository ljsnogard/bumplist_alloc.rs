//! `big_page` 特性的集成测试：独立大页链表、混合分配与回收。
//!
//! Integration tests for the `big_page` feature: the separate big-page list, mixed
//! allocations, and reclamation.
#![cfg(feature = "big_page")]

mod common;

use std::{
    alloc::{Allocator, Layout},
    ptr::{self, NonNull},
    vec::Vec,
};

use bumplist_alloc::{BumpListAlloc, PageSize};
use common::CountingAlloc;

type TestBump = BumpListAlloc<CountingAlloc>;

/// 一块仍然活跃的测试分配。
struct LiveBlock {
    ptr_: NonNull<u8>,
    layout_: Layout,
    tag_: u8,
}

impl LiveBlock {
    fn new_(ptr: NonNull<u8>, layout: Layout, tag: u8) -> Self {
        // SAFETY: 调用方刚刚分配并独占这块内存。
        unsafe { ptr::write_bytes(ptr.as_ptr(), tag, layout.size()) };
        Self {
            ptr_: ptr,
            layout_: layout,
            tag_: tag,
        }
    }

    fn verify_(&self) {
        for offset in (0..self.layout_.size()).step_by(97) {
            // SAFETY: 块仍然活跃，偏移落在 layout.size() 内。
            assert_eq!(unsafe { self.ptr_.as_ptr().add(offset).read() }, self.tag_);
        }
    }

    /// # Safety
    ///
    /// 必须只为同一个 allocator 与 layout 调用一次。
    unsafe fn release_(&self, bump: &TestBump) {
        // SAFETY: 由调用方保证 allocator / layout / 一次性。
        unsafe { Allocator::deallocate(bump, self.ptr_, self.layout_) };
    }
}

/// 测试目标：超过页大小的请求由独立大页承接，内容可读写，释放后立即归还 backing。
/// Test goal: An oversized request is served by a big page, remains readable/writable,
/// and is returned to the backing allocator immediately on release.
/// - 手段：1 KiB 页上分配 8 KiB，观察计数分配器的申请大小与次数；写入并抽读。
/// - 判定：申请次数只增加 1，且存在大于 1 KiB 的申请；释放后释放次数增加 1。
///
/// - Means: allocate 8 KiB on a 1 KiB-page arena, observe the counting allocator, then
///   write and spot-check the block.
/// - Verdict: exactly one extra allocation larger than a page; releasing it increases the
///   backing deallocation count by one.
#[test]
fn oversized_allocation_uses_big_page_() {
    let (alloc, counters) = CountingAlloc::new();
    let bump = BumpListAlloc::new_in(alloc, PageSize::P1K).expect("1 KiB 页应当受支持");
    let allocs_after_init = counters.allocs();
    let layout = Layout::from_size_align(8 * 1024, 8).expect("布局合法");

    let block = Allocator::allocate(&bump, layout)
        .expect("开启 big_page 后 8 KiB 请求应当成功")
        .cast::<u8>();
    assert_eq!(counters.allocs(), allocs_after_init + 1);
    assert!(
        counters
            .sizes()
            .iter()
            .any(|size| *size > PageSize::P1K.bytes()),
        "大页必须使用大于普通页的 backing 布局"
    );

    let live = LiveBlock::new_(block, layout, 0xA5);
    live.verify_();
    // SAFETY: 同一 allocator / layout，且只释放一次。
    unsafe { live.release_(&bump) };
    assert_eq!(counters.deallocs(), 1, "大页应当在释放时立即归还");

    drop(bump);
    assert_eq!(counters.allocs(), counters.deallocs());
}

/// 测试目标：小页与大页交错分配、交错释放时互不干扰，且大页不会进入小页空页缓存。
/// Test goal: Interleaved small/big allocations and releases do not interfere, and big
/// pages never enter the small-page empty-page cache.
/// - 手段：1 KiB 页上循环分配 4 个小块 + 1 个 3 KiB 大块，写入不同字节；交错释放后
///   再分配一个小块。
/// - 判定：所有块内容正确；小页申请的布局仍是 1 KiB，大页申请大于 1 KiB；析构后
///   申请 / 释放次数配平。
///
/// - Means: interleave four small blocks and one 3 KiB big block on a 1 KiB-page arena,
///   write distinct bytes, release in interleaved order, then allocate a small block
///   again.
/// - Verdict: every block keeps its pattern; normal pages still request 1 KiB while big
///   pages request more; allocation/deallocation counts balance after dropping the arena.
#[test]
fn mixed_small_and_big_allocations_() {
    let (alloc, counters) = CountingAlloc::new();
    let bump = BumpListAlloc::new_in(alloc, PageSize::P1K).expect("1 KiB 页应当受支持");
    let small = Layout::from_size_align(24, 8).expect("布局合法");
    let big = Layout::from_size_align(3 * 1024, 8).expect("布局合法");
    let mut live = Vec::new();

    for round in 0..8u8 {
        for index in 0..4u8 {
            let block = Allocator::allocate(&bump, small)
                .expect("小块分配应当成功")
                .cast::<u8>();
            let tag = round.wrapping_mul(16).wrapping_add(index);
            live.push(LiveBlock::new_(block, small, tag));
        }
        let block = Allocator::allocate(&bump, big)
            .expect("大块分配应当成功")
            .cast::<u8>();
        let tag = 0xF0u8.wrapping_sub(round);
        live.push(LiveBlock::new_(block, big, tag));
    }

    for block in &live {
        block.verify_();
    }

    for index in (0..live.len()).step_by(2) {
        // SAFETY: 每个块只释放一次，且来自同一个 allocator / layout。
        unsafe { live[index].release_(&bump) };
    }
    for index in (1..live.len()).step_by(2) {
        // SAFETY: 同上。
        unsafe { live[index].release_(&bump) };
    }

    let sizes = counters.sizes();
    assert!(
        sizes.iter().any(|size| *size == PageSize::P1K.bytes()),
        "小页仍应使用配置的页大小"
    );
    assert!(
        sizes.iter().any(|size| *size > PageSize::P1K.bytes()),
        "大页应使用独立的更大布局"
    );

    let again = Allocator::allocate(&bump, small)
        .expect("大页全部归还后小页仍可分配")
        .cast::<u8>();
    // SAFETY: 指针来自同一 allocator 与 layout。
    unsafe { Allocator::deallocate(&bump, again, small) };

    drop(bump);
    assert_eq!(counters.allocs(), counters.deallocs());
}

/// 测试目标：大页支持远超普通页大小的用户对齐，且块头仍能反查 Page。
/// Test goal: A big page supports user alignment far beyond a normal page while its
/// block header still recovers the page.
/// - 手段：1 KiB 页上分配 8 KiB、64 KiB 对齐的块，检查地址与首尾写入。
/// - 判定：返回地址满足 64 KiB 对齐；首尾字节读回一致；释放后计数配平。
///
/// - Means: allocate 8 KiB with 64 KiB alignment on a 1 KiB-page arena, then check the
///   address and write the first/last byte.
/// - Verdict: the address is 64 KiB aligned, both bytes read back, and the counts balance
///   after release.
#[test]
fn big_allocation_respects_large_alignment_() {
    let (alloc, counters) = CountingAlloc::new();
    let bump = BumpListAlloc::new_in(alloc, PageSize::P1K).expect("1 KiB 页应当受支持");
    let align = 64 * 1024;
    let layout = Layout::from_size_align(8 * 1024, align).expect("布局合法");

    let block = Allocator::allocate(&bump, layout)
        .expect("高对齐大块应当成功")
        .cast::<u8>();
    assert_eq!(block.as_ptr().addr() % align, 0);
    // SAFETY: 块至少 8 KiB；首尾写入都在 layout 内。
    unsafe {
        block.as_ptr().write(0x5A);
        block.as_ptr().add(layout.size() - 1).write(0xC3);
        assert_eq!(block.as_ptr().read(), 0x5A);
        assert_eq!(block.as_ptr().add(layout.size() - 1).read(), 0xC3);
    }
    // SAFETY: 指针来自同一 allocator 与 layout。
    unsafe { Allocator::deallocate(&bump, block, layout) };

    drop(bump);
    assert_eq!(counters.allocs(), counters.deallocs());
}

/// 测试目标：`Box` 可以持有大页上的值，析构时归还整页。
/// Test goal: `Box` can own a value in a big page and returns the whole page on drop.
/// - 手段：分配 16 KiB 的 `[u8; 16384]`，写入后交给 `bumplist_alloc::Box`。
/// - 判定：索引读回正确；`Box` 析构后 backing 释放次数增加 1；arena 析构后配平。
///
/// - Means: allocate a 16 KiB `[u8; 16384]`, write it, and hand it to
///   `bumplist_alloc::Box`.
/// - Verdict: indexing reads back the value; dropping the box increases backing
///   deallocations by one, and the arena counts balance after drop.
#[test]
fn big_box_release_returns_backing_() {
    let (alloc, counters) = CountingAlloc::new();
    let bump = BumpListAlloc::new_in(alloc, PageSize::P1K).expect("1 KiB 页应当受支持");
    let layout = Layout::new::<[u8; 16 * 1024]>();
    let block = Allocator::allocate(&bump, layout)
        .expect("大块分配应当成功")
        .cast::<[u8; 16 * 1024]>();
    // SAFETY: 块是未初始化的 [u8; 16384]；只写入一次。
    unsafe { block.as_ptr().write([7u8; 16 * 1024]) };
    // SAFETY: 指针来自本 arena，布局匹配，且只交给一个 Box。
    let boxed = unsafe { bumplist_alloc::Box::from_raw(block.as_ptr()) };
    assert_eq!(boxed[0], 7);
    assert_eq!(boxed[16 * 1024 - 1], 7);
    drop(boxed);
    assert_eq!(counters.deallocs(), 1, "Box 析构应立即归还大页");

    drop(bump);
    assert_eq!(counters.allocs(), counters.deallocs());
}

/// 测试目标：大页释放后立即归还 backing，而不是像小页一样保留空页。
/// Test goal: A big page returns to the backing allocator immediately instead of being
/// cached like a small empty page.
/// - 手段：记录构造后的计数，分配 / 释放 8 KiB 大块，再做一次小块分配。
/// - 判定：大页申请与释放各增加 1；小块仍可复用 root，析构后计数配平。
///
/// - Means: snapshot counters after construction, allocate/release an 8 KiB block, then
///   allocate a small block.
/// - Verdict: the big page adds exactly one allocation and one deallocation; small
///   allocations still work and the final counts balance.
#[test]
fn big_page_is_returned_immediately_() {
    let (alloc, counters) = CountingAlloc::new();
    let bump = BumpListAlloc::new_in(alloc, PageSize::P1K).expect("1 KiB 页应当受支持");
    let after_init = (counters.allocs(), counters.deallocs());
    let big = Layout::from_size_align(8 * 1024, 8).expect("布局合法");

    let block = Allocator::allocate(&bump, big).expect("大块分配应当成功");
    assert_eq!(counters.allocs(), after_init.0 + 1);
    // SAFETY: 指针来自同一 allocator 与 layout。
    unsafe { Allocator::deallocate(&bump, block.cast(), big) };
    assert_eq!(counters.deallocs(), after_init.1 + 1);

    let small = Layout::from_size_align(32, 8).expect("布局合法");
    let block = Allocator::allocate(&bump, small).expect("小块仍应可分配");
    // SAFETY: 指针来自同一 allocator 与 layout。
    unsafe { Allocator::deallocate(&bump, block.cast(), small) };

    drop(bump);
    assert_eq!(counters.allocs(), counters.deallocs());
}

/// 测试目标：开启 `big_page` 但只做小分配时，backing 布局与关闭 feature 时完全一致。
/// Test goal: With `big_page` enabled but only small allocations, backing layouts match
/// the feature-off behavior.
/// - 手段：1 KiB 页上分配 200 个 24 字节小块，抽读后全部释放并析构 arena。
/// - 判定：每次 backing 申请的大小都是 1 KiB；申请 / 释放次数配平。
///
/// - Means: allocate 200 small 24-byte blocks on a 1 KiB-page arena, spot-check them,
///   release everything, and drop the arena.
/// - Verdict: every backing allocation is exactly one page; allocation and deallocation
///   counts balance.
#[test]
fn small_only_workload_keeps_page_layout_() {
    let (alloc, counters) = CountingAlloc::new();
    let bump = BumpListAlloc::new_in(alloc, PageSize::P1K).expect("1 KiB 页应当受支持");
    let layout = Layout::from_size_align(24, 8).expect("布局合法");
    let mut blocks = Vec::new();
    for index in 0..200u8 {
        let block = Allocator::allocate(&bump, layout)
            .expect("小块分配应当成功")
            .cast::<u8>();
        blocks.push((block, index));
    }
    for (block, tag) in &blocks {
        // SAFETY: 块仍然活跃，24 字节可读。
        unsafe { ptr::write_bytes(block.as_ptr(), *tag, layout.size()) };
    }
    for (block, tag) in blocks {
        // SAFETY: 块仍然活跃，24 字节可读。
        unsafe { assert_eq!(block.as_ptr().read(), tag) };
        // SAFETY: 指针来自同一 allocator 与 layout，且只释放一次。
        unsafe { Allocator::deallocate(&bump, block, layout) };
    }

    assert!(
        counters
            .sizes()
            .iter()
            .all(|size| *size == PageSize::P1K.bytes()),
        "只做小分配时不应出现大页布局"
    );
    drop(bump);
    assert_eq!(counters.allocs(), counters.deallocs());
}

/// 测试目标：`Vec` 扩容超过页大小后可以继续工作，并在析构时归还大页。
/// Test goal: `Vec` can grow beyond a page and returns its big buffer on drop.
/// - 手段：1 KiB 页上 `try_reserve(8 KiB)`，填入数据后读取并析构。
/// - 判定：扩容成功、内容正确；arena 析构后申请 / 释放次数配平。
///
/// - Means: call `try_reserve(8 KiB)` on a 1 KiB-page arena, fill and read the data, then
///   drop it.
/// - Verdict: the reservation succeeds with correct contents, and the arena counts
///   balance after drop.
#[test]
fn vec_can_grow_beyond_page_() {
    let (alloc, counters) = CountingAlloc::new();
    let bump = BumpListAlloc::new_in(alloc, PageSize::P1K).expect("1 KiB 页应当受支持");
    let mut values: Vec<u8, &TestBump> = Vec::new_in(&bump);
    values
        .try_reserve(8 * 1024)
        .expect("开启 big_page 后 Vec 扩容应当成功");
    values.extend(std::iter::repeat_n(0x7E, 8 * 1024));
    assert_eq!(values.len(), 8 * 1024);
    assert!(values.iter().all(|byte| *byte == 0x7E));

    drop(values);
    drop(bump);
    assert_eq!(counters.allocs(), counters.deallocs());
}

/// 测试目标：arena 析构时仍然存活的大页会被归还，而不是泄漏。
/// Test goal: A live big block is freed when the arena is dropped instead of leaking.
/// - 手段：分配 16 KiB 大块并写入，保持存活直到 `drop(bump)`。
/// - 判定：析构后 backing 申请次数等于释放次数。
///
/// - Means: allocate and write a 16 KiB block, keep it live, then `drop(bump)`.
/// - Verdict: backing allocation and deallocation counts match after drop.
#[test]
fn drop_with_live_big_allocation_() {
    let (alloc, counters) = CountingAlloc::new();
    let bump = BumpListAlloc::new_in(alloc, PageSize::P1K).expect("1 KiB 页应当受支持");
    let layout = Layout::from_size_align(16 * 1024, 8).expect("布局合法");
    let block = Allocator::allocate(&bump, layout)
        .expect("大块分配应当成功")
        .cast::<u8>();
    // SAFETY: 块是有效的 16 KiB 可写内存。
    unsafe { ptr::write_bytes(block.as_ptr(), 0x3C, layout.size()) };

    drop(bump);
    assert_eq!(counters.allocs(), counters.deallocs());
}

/// 测试目标：多线程同时进行小块与大块分配 / 释放时，两条链表与两把锁不产生冲突。
/// Test goal: Concurrent small/big allocation and release across threads does not
/// conflict between the two page lists and their locks.
/// - 手段：4 个线程各循环 32 次，每次分配 64 B 小块与 16 KiB 大块，写入后读回并释放。
/// - 判定：每个线程读到自己的字节；作用域结束后 arena 可正常析构。
///
/// - Means: four threads each loop 32 times, allocating a 64 B small block and a 16 KiB
///   big block, writing and reading them back, then releasing both.
/// - Verdict: each thread reads its own pattern and the arena drops cleanly.
#[test]
fn concurrent_mixed_allocations_() {
    let bump = BumpListAlloc::new(PageSize::P4K).expect("4 KiB 页应当受支持");
    let small = Layout::from_size_align(64, 8).expect("布局合法");
    let big = Layout::from_size_align(16 * 1024, 8).expect("布局合法");

    std::thread::scope(|scope| {
        for worker in 0..4u8 {
            let bump = &bump;
            scope.spawn(move || {
                for round in 0..32u8 {
                    let small_block = Allocator::allocate(bump, small)
                        .expect("小块分配应当成功")
                        .cast::<u8>();
                    let big_block = Allocator::allocate(bump, big)
                        .expect("大块分配应当成功")
                        .cast::<u8>();
                    let small_tag = worker.wrapping_mul(32).wrapping_add(round);
                    let big_tag = small_tag.wrapping_add(128);
                    // SAFETY: 两块都是本次刚分配、独占、足够大的内存。
                    unsafe {
                        ptr::write_bytes(small_block.as_ptr(), small_tag, small.size());
                        ptr::write_bytes(big_block.as_ptr(), big_tag, big.size());
                        assert_eq!(small_block.as_ptr().read(), small_tag);
                        assert_eq!(big_block.as_ptr().read(), big_tag);
                        // SAFETY: 由本线程分配，且只释放一次。
                        Allocator::deallocate(bump, big_block, big);
                        Allocator::deallocate(bump, small_block, small);
                    }
                }
            });
        }
    });

    drop(bump);
}
