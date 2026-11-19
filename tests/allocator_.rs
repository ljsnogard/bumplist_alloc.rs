//! `BumpListAlloc` 的集成测试：标准容器、固定页限制、页大小与 drop 行为。

mod common;

use bumplist_alloc::{BumpListAlloc, BumpListAllocError, PageSize};
use common::CountingAlloc;
use std::{
    alloc::{Allocator, Layout},
    boxed::Box,
    vec::Vec,
};

type TestBump = BumpListAlloc<CountingAlloc>;

/// 测试目标：`Vec` 与 `Box` 可以通过 `&BumpListAlloc` 使用 borrowed allocator，
/// 并在容器析构后让空页归还给底层分配器。
/// Test goal: `Vec` and `Box` can use the borrowed allocator and release the empty page
/// after the containers are dropped.
/// - 手段 / Method: 使用 4 KiB 页，向 `Vec` 写入 4 个 u32，再分配一个 `Box<u64>`；
///   离开作用域后析构 allocator，观察计数分配器。
/// - 判定 / Assertion: 读写结果正确；Page 申请次数等于释放次数；所有 Page 的申请大小
///   都是 4096 字节。
#[test]
fn vec_and_box_share_borrowed_allocator() {
    let (alloc, counters) = CountingAlloc::new();
    let bump = BumpListAlloc::new_in(alloc, PageSize::P4K).expect("4 KiB 页应当受支持");

    {
        let mut values: Vec<u32, &TestBump> = Vec::new_in(&bump);
        values.extend_from_slice(&[10, 20, 30, 40]);

        let boxed: Box<u64, &TestBump> = Box::new_in(7, &bump);
        assert_eq!(&values[..], &[10, 20, 30, 40]);
        assert_eq!(*boxed, 7);
    }

    drop(bump);

    assert!(counters.allocs() >= 1);
    assert_eq!(counters.allocs(), counters.deallocs());
    assert!(
        counters
            .sizes()
            .iter()
            .all(|size| *size == PageSize::P4K.bytes())
    );
}

/// 测试目标：零尺寸分配不追加 Page，返回按对齐的悬垂地址，释放无副作用。
/// Test goal: A zero-sized allocation appends no page, returns an aligned dangling
/// pointer, and its deallocation is a no-op.
/// - 手段 / Method: 用 1 KiB 页构造 allocator（构造时申请 root 页），再分配
///   `Layout::new::<()>()` 并释放。
/// - 判定 / Assertion: 返回地址满足对齐；计数分配器的申请次数与构造后相同，
///   没有额外的 Page 申请，也没有 Page 释放。
#[test]
fn zero_sized_allocation_does_not_create_page() {
    let (alloc, counters) = CountingAlloc::new();
    let bump = BumpListAlloc::new_in(alloc, PageSize::P1K).expect("1 KiB 页应当受支持");
    let allocs_after_construction = counters.allocs();
    let alloc: &TestBump = &bump;

    let layout = Layout::new::<()>();
    let block = Allocator::allocate(&alloc, layout).expect("零尺寸分配应当成功");
    let ptr = block.cast::<u8>();
    assert_eq!(ptr.as_ptr() as usize % layout.align(), 0);

    // SAFETY: 该指针来自同一 allocator 与 layout 的零尺寸分配。
    unsafe { Allocator::deallocate(&alloc, ptr, layout) };

    assert_eq!(counters.allocs(), allocs_after_construction);
    assert_eq!(counters.deallocs(), 0);
    drop(bump);
    assert_eq!(counters.allocs(), counters.deallocs());
}

/// 测试目标：固定页大小下，超过页可容纳空间的请求返回 `AllocError`，且不追加 Page。
/// Test goal: With fixed page sizes, an oversized request returns `AllocError` and appends
/// no page.
/// - 手段 / Method: 1 KiB 页上直接分配 8 KiB，并用 `Vec::try_reserve` 触发扩容失败。
/// - 判定 / Assertion: 两次尝试都返回错误；计数分配器的申请次数与构造后相同。
#[test]
fn fixed_page_size_limits_allocation() {
    let (alloc, counters) = CountingAlloc::new();
    let bump = BumpListAlloc::new_in(alloc, PageSize::P1K).expect("1 KiB 页应当受支持");
    let allocs_after_construction = counters.allocs();
    let alloc: &TestBump = &bump;

    let layout = Layout::from_size_align(8 * 1024, 8).expect("布局合法");
    assert!(Allocator::allocate(&alloc, layout).is_err());
    assert_eq!(counters.allocs(), allocs_after_construction);

    let mut values: Vec<u8, &TestBump> = Vec::new_in(&bump);
    assert!(values.try_reserve(8 * 1024).is_err());
    assert_eq!(counters.allocs(), allocs_after_construction);

    drop(values);
    drop(bump);
    assert_eq!(counters.allocs(), counters.deallocs());
}

/// 测试目标：同一 allocator 的所有 Page 都使用配置的页大小，页满后会继续开新页。
/// Test goal: Every page uses the configured size; more pages are created when needed.
/// - 手段 / Method: 1 KiB 页上连续分配 12 个 200 字节块，再全部释放。
/// - 判定 / Assertion: 至少申请两个 Page；每次申请大小都是 1024；全部块释放后只保留
///   一个空 tail 页，drop allocator 后申请/释放次数配平。
#[test]
fn all_pages_use_configured_size() {
    let (alloc, counters) = CountingAlloc::new();
    let bump = BumpListAlloc::new_in(alloc, PageSize::P1K).expect("1 KiB 页应当受支持");
    let alloc: &TestBump = &bump;

    let layout = Layout::from_size_align(200, 8).expect("布局合法");
    let mut blocks = Vec::new();
    for _ in 0..12 {
        let block = Allocator::allocate(&alloc, layout).expect("块应当放得下");
        let ptr = block.cast::<u8>();
        assert_eq!(ptr.as_ptr() as usize % 8, 0);
        blocks.push(ptr);
    }

    assert!(counters.allocs() >= 2);
    assert!(
        counters
            .sizes()
            .iter()
            .all(|size| *size == PageSize::P1K.bytes())
    );

    for ptr in blocks {
        // SAFETY: 指针来自同一 allocator 与 layout 的分配。
        unsafe { Allocator::deallocate(&alloc, ptr, layout) };
    }
    // 空页策略：只保留一个空 tail 页；它会在 allocator drop 时归还。
    assert_eq!(counters.allocs() - counters.deallocs(), 1);

    drop(bump);
    assert_eq!(counters.allocs(), counters.deallocs());
}

/// 测试目标：allocator 析构时会释放仍存活的 Page，并扫描页内块链。
/// Test goal: Dropping the allocator releases pages that still have live blocks and
/// scans the in-page block chain.
/// - 手段 / Method: 1 KiB 页上分配一个块但不释放，直接析构 allocator。
/// - 判定 / Assertion: 计数分配器看到一次 Page 申请和一次 Page 释放。
#[test]
fn drop_releases_live_pages() {
    let (alloc, counters) = CountingAlloc::new();
    let bump = BumpListAlloc::new_in(alloc, PageSize::P1K).expect("1 KiB 页应当受支持");

    {
        let alloc: &TestBump = &bump;
        let layout = Layout::from_size_align(32, 8).expect("布局合法");
        let block = Allocator::allocate(&alloc, layout).expect("块应当放得下");
        let _keep_alive = block.cast::<u8>();
    }

    drop(bump);

    assert_eq!(counters.allocs(), 1);
    assert_eq!(counters.deallocs(), 1);
}

/// 测试目标：`BumpListAlloc::clone` 只增加引用计数，不申请新 Page，且 clone 可以释放
/// 原句柄分配的块。
/// Test goal: `BumpListAlloc::clone` only bumps the reference count and allocates no page;
/// a clone can release a block allocated through the original handle.
/// - 手段 / Method: 构造 4 KiB 页 arena，clone 一次后用 clone 分配 64 字节块，再用原句柄
///   释放；先 drop clone，再 drop 原句柄。
/// - 判定 / Assertion: clone 后 Page 申请次数仍为 1；drop clone 时第一页不归还；最后一个
///   句柄 drop 后申请与释放次数配平。
#[test]
fn cloned_handles_share_arena() {
    let (alloc, counters) = CountingAlloc::new();
    let bump = BumpListAlloc::new_in(alloc, PageSize::P4K).expect("4 KiB 页应当受支持");
    assert_eq!(counters.allocs(), 1);

    let clone = bump.clone();
    assert_eq!(counters.allocs(), 1, "Clone 不应申请新的 Page");

    let layout = Layout::from_size_align(64, 8).expect("布局合法");
    let block = {
        let alloc: &TestBump = &clone;
        Allocator::allocate(&alloc, layout).expect("分配成功")
    };
    {
        let alloc: &TestBump = &bump;
        // SAFETY: 块来自同一个共享 arena 与 layout。
        unsafe { Allocator::deallocate(&alloc, block.cast(), layout) };
    }

    drop(clone);
    assert_eq!(counters.deallocs(), 0, "还有句柄存活时不能归还第一页");
    drop(bump);
    assert_eq!(counters.allocs(), counters.deallocs());
}

/// 测试目标：`BumpListAlloc` 作为拥有型 `Vec` allocator 可用，且 `Vec::clone` 通过
/// `AllocatorClone` 共享同一 arena。
/// Test goal: `BumpListAlloc` works as an owned `Vec` allocator, and `Vec::clone` shares
/// the same arena through `AllocatorClone`.
/// - 手段 / Method: 用 `Vec<u32, TestBump>` 写入数据并 clone；先 drop 原 Vec，再 drop
///   clone。
/// - 判定 / Assertion: 两份数据一致；drop 原 Vec 时第一页仍存活；全部 drop 后 Page
///   申请与释放次数配平。
#[test]
fn owned_allocator_supports_vec_clone() {
    let (alloc, counters) = CountingAlloc::new();
    let bump = BumpListAlloc::new_in(alloc, PageSize::P4K).expect("4 KiB 页应当受支持");

    let mut values: Vec<u32, TestBump> = Vec::new_in(bump);
    values.extend_from_slice(&[1u32, 2u32, 3u32]);
    let copy = values.clone();
    assert_eq!(&values[..], &[1u32, 2u32, 3u32]);
    assert_eq!(&copy[..], &[1u32, 2u32, 3u32]);

    drop(values);
    assert_eq!(counters.deallocs(), 0, "clone 仍持有 arena");
    drop(copy);
    assert_eq!(counters.allocs(), counters.deallocs());
}

/// 测试目标：默认 allocator 是 `Send + Sync`，`&BumpListAlloc` 也可跨线程共享；
/// 同时验证 256 KiB 页的平台支持性。
/// Test goal: The default allocator is `Send + Sync`, `&BumpListAlloc` is shareable,
/// and platform support for the 256 KiB page size is reported correctly.
/// - 手段 / Method: 编译期 `Send + Sync` 断言；按目标指针宽度构造 `P256K`。
/// - 判定 / Assertion: 64 位构造成功，32 位返回 `UnsupportedPageSize`。
#[test]
fn send_sync_and_p256k_platform_support() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<BumpListAlloc>();
    assert_send_sync::<&BumpListAlloc>();

    #[cfg(target_pointer_width = "64")]
    {
        assert!(PageSize::P256K.is_supported());
        assert!(BumpListAlloc::new(PageSize::P256K).is_ok());
    }

    #[cfg(target_pointer_width = "32")]
    {
        assert!(!PageSize::P256K.is_supported());
        let error = match BumpListAlloc::new(PageSize::P256K) {
            Ok(_) => panic!("32 位不支持 256 KiB 页"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            BumpListAllocError::UnsupportedPageSize {
                page_size: PageSize::P256K
            }
        ));
    }
}

/// 测试目标：`UnsupportedPageSize` 的 `Display` 输出包含页大小与错误上下文。
/// Test goal: The `Display` output of `UnsupportedPageSize` carries the page size and
/// error context.
/// - 手段 / Method: 构造错误并格式化。
/// - 判定 / Assertion: 输出非空，且包含页大小调试名。
#[test]
fn unsupported_page_size_display() {
    let error = BumpListAllocError::UnsupportedPageSize {
        page_size: PageSize::P256K,
    };
    let text = error.to_string();
    assert!(!text.is_empty());
    assert!(text.contains("P256K"));
}
