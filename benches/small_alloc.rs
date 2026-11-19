//! 小页分配 / 回收的微基准。
//!
//! 用同一份工作负载分别在 `big_page` 关闭与开启时运行，对比每批“分配 64 个小块 +
//! 全部归还”的耗时。批量大小刻意让一批填不满 4 KiB 页，因此每批结束都会走到
//! `cleanup_empty_page_` 的空页路径，能同时观察热路径与页变空的冷路径。
//!
//! 运行：
//!
//! ```bash
//! cargo bench --bench small_alloc
//! cargo bench --features big_page --bench small_alloc
//! ```

use std::{
    alloc::{Allocator, Layout},
    hint::black_box,
    time::{Duration, Instant},
};

use bumplist_alloc::{BumpListAlloc, PageSize};

const BATCH: usize = 64;

fn measure_small_(bump: &BumpListAlloc, rounds: usize, batches: usize) -> Duration {
    let layout = Layout::from_size_align(32, 8).expect("布局合法");
    let mut best = Duration::MAX;
    for _ in 0..rounds {
        let start = Instant::now();
        for _ in 0..batches {
            let mut blocks = [None; BATCH];
            for slot in &mut blocks {
                *slot = Some(Allocator::allocate(bump, layout).expect("小页分配应当成功"));
            }
            for block in blocks.into_iter().flatten() {
                // SAFETY: 指针来自同一 allocator 与 layout，且只释放一次。
                unsafe { Allocator::deallocate(bump, block.cast(), layout) };
            }
        }
        best = best.min(start.elapsed());
    }
    black_box(best);
    best
}

fn main() {
    let bump = BumpListAlloc::new(PageSize::P4K).expect("4 KiB 页应当受支持");

    // 预热：让 root 页、分支预测与缓存进入稳定状态。
    let _ = measure_small_(&bump, 1, 200);

    let rounds = 5;
    let batches = 4_000;
    let elapsed = measure_small_(&bump, rounds, batches);
    let ops = (batches * BATCH) as f64;
    println!(
        "small_alloc ns/op: {:.2} (best of {rounds} rounds, {batches} batches)",
        elapsed.as_nanos() as f64 / ops
    );

    #[cfg(feature = "big_page")]
    {
        let big_layout = Layout::from_size_align(64 * 1024, 8).expect("布局合法");
        let big = Allocator::allocate(&bump, big_layout).expect("大页分配应当成功");
        let elapsed = measure_small_(&bump, rounds, batches);
        println!(
            "small_alloc_with_live_big ns/op: {:.2} (best of {rounds} rounds, {batches} batches)",
            elapsed.as_nanos() as f64 / ops
        );
        // SAFETY: 大块来自同一 allocator 与 layout，且只释放一次。
        unsafe { Allocator::deallocate(&bump, big.cast(), big_layout) };
    }
}
