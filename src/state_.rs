//! 引用计数与两把 Core 锁共享的原子状态。
//!
//! `BumpListCore` 只用一个 [`AtomicUsize`] 同时保存三类信息：
//!
//! ```text
//! 64 位示例：
//!   bit 63                  list lock 标记
//!   bit 62                  backing lock 标记
//!   bit 61..0               BumpListAlloc 句柄引用计数
//! ```
//!
//! 引用计数从 1 开始；[`BumpListAlloc::clone`](crate::BumpListAlloc) 只增加低位计数，
//! Drop 只减少低位计数，两把锁通过 CAS 置位 / 清位。计数上限是低位的全 1 值，克隆时
//! 用 CAS 循环检查溢出，因此正常情况下计数增加不会进位到锁标记位。
//!
//! 锁的语义仍是不可重入自旋锁：
//! - list lock 保护 Page 链表的 `head_` / `tail_`、`Page.links_` 与挂链 / 摘链；
//! - backing lock 串行化对底层分配器 `A` 的所有 allocate / deallocate 调用。

use core::{
    hint::spin_loop,
    sync::atomic::{AtomicUsize, Ordering},
};

/// 次高位：backing lock 标记。
const BACKING_LOCK_BIT: usize = 1usize << (usize::BITS - 2);

/// 最高位：list lock 标记。
const LIST_LOCK_BIT: usize = 1usize << (usize::BITS - 1);

/// 低位引用计数掩码。
const REF_COUNT_MASK: usize = BACKING_LOCK_BIT - 1;

/// Core 的共享原子状态：低位引用计数 + 最高两位锁标记。
#[repr(transparent)]
pub(crate) struct CoreState {
    state_: AtomicUsize,
}

impl CoreState {
    /// 新建状态：引用计数为 1，锁标记都为 0。
    pub(crate) const fn new() -> Self {
        Self {
            state_: AtomicUsize::new(1),
        }
    }

    /// 增加一个句柄引用。
    ///
    /// # Panics
    ///
    /// 引用计数达到低位可表示上限时 panic。该上限在 64 位平台为 `2^62 - 1`，
    /// 在 32 位平台为 `2^30 - 1`，实际使用中不可达。
    pub(crate) fn clone_ref_(&self) {
        let mut current = self.state_.load(Ordering::Relaxed);
        loop {
            if current & REF_COUNT_MASK == REF_COUNT_MASK {
                panic!("BumpListAlloc reference count overflow");
            }
            // count 未到上限，因此 +1 不会进位到锁标记位；CAS 会保留锁位。
            match self.state_.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Result::Ok(_) => return,
                Result::Err(actual) => current = actual,
            }
        }
    }

    /// 减少一个句柄引用；返回是否减到了 0（即当前 Drop 是最后一个句柄）。
    ///
    /// 锁标记位于高位，减法只作用于低位计数，因此不会破坏正在进行的锁。
    pub(crate) fn release_ref_(&self) -> bool {
        let previous = self.state_.fetch_sub(1, Ordering::AcqRel);
        let count = previous & REF_COUNT_MASK;
        debug_assert_ne!(count, 0, "BumpListAlloc reference count underflow");
        if count == 1 {
            // 最后一个句柄不可能在别的线程还持有 Core 锁时被 Drop；若出现说明
            // 所有权或锁生命周期被破坏。
            debug_assert_eq!(
                previous & !REF_COUNT_MASK,
                0,
                "last BumpListAlloc handle dropped while a Core lock is held"
            );
        }
        count == 1
    }

    /// 获取 list lock。
    pub(crate) fn lock_list_(&self) -> CoreLockGuard<'_> {
        self.lock_(LIST_LOCK_BIT)
    }

    /// 获取 backing lock。
    pub(crate) fn lock_backing_(&self) -> CoreLockGuard<'_> {
        self.lock_(BACKING_LOCK_BIT)
    }

    /// 阻塞式获取 `bit` 对应的锁。
    fn lock_(&self, bit: usize) -> CoreLockGuard<'_> {
        loop {
            if let Option::Some(guard) = self.try_lock_(bit) {
                return guard;
            }
            spin_loop();
        }
    }

    /// 尝试获取 `bit` 对应的锁；已被占用时返回 [`None`]。
    fn try_lock_(&self, bit: usize) -> Option<CoreLockGuard<'_>> {
        let mut current = self.state_.load(Ordering::Relaxed);
        loop {
            if current & bit != 0 {
                return Option::None;
            }
            match self.state_.compare_exchange_weak(
                current,
                current | bit,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Result::Ok(_) => {
                    return Option::Some(CoreLockGuard {
                        state_: &self.state_,
                        bit_: bit,
                    });
                }
                Result::Err(actual) => current = actual,
            }
        }
    }
}

/// Core 锁的 RAII guard；析构时清掉对应锁标记，保留引用计数与另一把锁。
#[derive(Debug)]
pub(crate) struct CoreLockGuard<'a> {
    state_: &'a AtomicUsize,
    bit_: usize,
}

impl Drop for CoreLockGuard<'_> {
    fn drop(&mut self) {
        self.state_.fetch_and(!self.bit_, Ordering::Release);
    }
}

#[cfg(test)]
mod tests_ {
    use super::*;

    /// 测试目标：锁标记正好占用最高两位，其余低位全部留给引用计数。
    /// Test goal: The lock flags occupy exactly the top two bits and all remaining low
    /// bits are reserved for the reference count.
    /// - 手段 / Method: 直接比较 `LIST_LOCK_BIT` / `BACKING_LOCK_BIT` /
    ///   `REF_COUNT_MASK` 与 `usize` 位宽。
    /// - 判定 / Assertion: 两个锁位分别是最高位与次高位，掩码恰好是次高位减一。
    #[test]
    fn lock_bits_are_top_two_() {
        assert_eq!(LIST_LOCK_BIT, 1usize << (usize::BITS - 1));
        assert_eq!(BACKING_LOCK_BIT, 1usize << (usize::BITS - 2));
        assert_eq!(REF_COUNT_MASK, BACKING_LOCK_BIT - 1);
        assert_eq!(LIST_LOCK_BIT.count_ones(), 1);
        assert_eq!(BACKING_LOCK_BIT.count_ones(), 1);
    }

    /// 测试目标：引用计数与两把锁共享同一个原子字，互不破坏对方的位。
    /// Test goal: The reference count and both lock flags share one atomic word without
    /// corrupting each other's bits.
    /// - 手段 / Method: 在持有 list lock 时 clone 引用，再检查低位计数与锁标记；随后释放
    ///   锁并依次 release 引用。
    /// - 判定 / Assertion: clone 后计数为 2 且 list lock 位仍为 1；释放 list lock 后
    ///   计数不变；两次 release 依次返回 false / true。
    #[test]
    fn ref_count_and_lock_bits_share_word_() {
        let state = CoreState::new();
        assert_eq!(state.state_.load(Ordering::Relaxed), 1);

        let list_guard = state.lock_list_();
        state.clone_ref_();
        let word = state.state_.load(Ordering::Relaxed);
        assert_eq!(word & REF_COUNT_MASK, 2);
        assert_ne!(word & LIST_LOCK_BIT, 0);
        assert_eq!(word & BACKING_LOCK_BIT, 0);
        drop(list_guard);

        let word = state.state_.load(Ordering::Relaxed);
        assert_eq!(word & REF_COUNT_MASK, 2);
        assert_eq!(word & LIST_LOCK_BIT, 0);
        assert!(!state.release_ref_());
        assert!(state.release_ref_());
    }

    /// 测试目标：list lock 与 backing lock 是两个独立标记，可以分别获取。
    /// Test goal: The list lock and backing lock are independent flags and can be
    /// acquired separately.
    /// - 手段 / Method: 先持有 list lock，再尝试获取 backing lock 与第二次 list lock。
    /// - 判定 / Assertion: backing lock 获取成功，第二次 list lock 失败；全部释放后
    ///   原子字回到初始计数 1。
    #[test]
    fn list_and_backing_locks_are_independent_() {
        let state = CoreState::new();
        let list_guard = state.lock_list_();
        assert!(state.try_lock_(LIST_LOCK_BIT).is_none());
        let backing_guard = state.try_lock_(BACKING_LOCK_BIT).expect("backing lock");
        drop(backing_guard);
        drop(list_guard);
        assert_eq!(state.state_.load(Ordering::Relaxed), 1);
    }
}
