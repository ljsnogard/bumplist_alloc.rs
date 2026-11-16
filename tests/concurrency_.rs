//! `BumpListAlloc` 的并发集成测试。

mod common;

use bumplist_alloc::{BumpListAlloc, PageSize};
use common::CountingAlloc;
use std::{thread, vec::Vec};

type TestBump = BumpListAlloc<CountingAlloc>;

/// 测试目标：多个线程可以同时通过共享的 `&BumpListAlloc` 分配与归还内存；页锁与
/// 链表锁协作，不产生死锁或丢失 Page。
/// Test goal: Multiple threads can allocate and deallocate through a shared
/// `&BumpListAlloc`; page locks and the list lock cooperate without deadlock or page
/// leaks.
/// - 手段 / Method: 4 个线程在 `thread::scope` 中各自持有 64 个 u64 的 `Vec`，
///   让它们竞争同一个 4 KiB 页的页锁与分配路径。
/// - 判定 / Assertion: 每个线程的数据正确；所有容器析构后 Page 申请/释放次数配平。
#[test]
fn concurrent_vecs_on_shared_bump() {
    let (alloc, counters) = CountingAlloc::new();
    let bump = BumpListAlloc::new_in(alloc, PageSize::P4K).expect("4 KiB 页应当受支持");

    thread::scope(|scope| {
        for thread_id in 0..4u64 {
            let bump_ref = &bump;
            scope.spawn(move || {
                let mut values: Vec<u64, &TestBump> = Vec::new_in(bump_ref);
                for index in 0..64u64 {
                    values.push(thread_id * 1000 + index);
                }
                assert_eq!(values.len(), 64);
                assert_eq!(values[0], thread_id * 1000);
                assert_eq!(values[63], thread_id * 1000 + 63);
            });
        }
    });

    // 所有线程结束后只保留一个空 tail 页；drop allocator 时归还。
    assert_eq!(counters.allocs() - counters.deallocs(), 1);
    drop(bump);
    assert_eq!(counters.allocs(), counters.deallocs());
}

/// 测试目标：并发分配后，部分线程只归还自己的一半块，页不会提前释放；最后全部归还
/// 后页链表清空。
/// Test goal: After concurrent allocation, releasing only half of each thread's blocks
/// must not free pages early; releasing everything empties the page list.
/// - 手段 / Method: 每个线程直接通过 `Allocator` 分配 16 个 64 字节块，先释放偶数
///   块，在两个 `Barrier` 之间让主线程观察计数分配器，再释放奇数块。
/// - 判定 / Assertion: 中途 Page 释放次数仍为 0（每页仍有奇数块存活）；最终申请/释放
///   次数配平。
#[test]
fn concurrent_partial_release_keeps_pages_alive() {
    use std::{
        alloc::{Allocator, Layout},
        ptr::NonNull,
        sync::Barrier,
    };

    let (alloc, counters) = CountingAlloc::new();
    let bump = BumpListAlloc::new_in(alloc, PageSize::P4K).expect("4 KiB 页应当受支持");
    let layout = Layout::from_size_align(64, 8).expect("布局合法");
    let barrier_half = Barrier::new(5);
    let barrier_finish = Barrier::new(5);

    // 每个 keeper 约 3 KiB，一个 4 KiB 页只放得下一个；它们保证至少四页在并发
    // 半量释放期间仍有存活块。
    let keeper_layout = Layout::from_size_align(3000, 8).expect("布局合法");
    let alloc: &TestBump = &bump;
    let keepers: Vec<NonNull<u8>> = (0..4)
        .map(|_| {
            Allocator::allocate(&alloc, keeper_layout)
                .expect("keeper 应当放得下")
                .cast::<u8>()
        })
        .collect();

    thread::scope(|scope| {
        for thread_id in 0..4u64 {
            let bump_ref = &bump;
            let barrier_half = &barrier_half;
            let barrier_finish = &barrier_finish;
            scope.spawn(move || {
                let alloc: &TestBump = bump_ref;
                let mut mine = Vec::new();
                for index in 0..16u64 {
                    let block = Allocator::allocate(&alloc, layout).expect("块应当放得下");
                    let ptr = block.cast::<u8>();
                    // SAFETY: 块至少 8 字节，写入 u64 不越界。
                    unsafe { ptr.as_ptr().cast::<u64>().write(thread_id * 100 + index) };
                    mine.push((ptr, thread_id * 100 + index));
                }
                for (ptr, value) in &mine {
                    // SAFETY: 块仍然存活，读取之前写入的 u64。
                    assert_eq!(unsafe { ptr.as_ptr().cast::<u64>().read() }, *value);
                }
                for (index, (ptr, _)) in mine.iter().enumerate() {
                    if index % 2 == 0 {
                        // SAFETY: 指针来自同一 allocator 与 layout。
                        unsafe { Allocator::deallocate(&alloc, *ptr, layout) };
                    }
                }
                barrier_half.wait();
                barrier_finish.wait();
                for (index, (ptr, _)) in mine.iter().enumerate() {
                    if index % 2 == 1 {
                        // SAFETY: 指针来自同一 allocator 与 layout。
                        unsafe { Allocator::deallocate(&alloc, *ptr, layout) };
                    }
                }
            });
        }

        barrier_half.wait();
        let allocs_at_half = counters.allocs();
        let deallocs_at_half = counters.deallocs();
        barrier_finish.wait();
        assert!(
            deallocs_at_half < allocs_at_half,
            "keeper 页不应在仍有存活块时释放"
        );
    });

    for keeper in keepers {
        // SAFETY: keeper 来自同一 allocator 与 keeper_layout。
        unsafe { Allocator::deallocate(&alloc, keeper, keeper_layout) };
    }

    // 所有块释放后只保留一个空 tail 页。
    assert_eq!(counters.allocs() - counters.deallocs(), 1);
    drop(bump);
    assert_eq!(counters.allocs(), counters.deallocs());
}
