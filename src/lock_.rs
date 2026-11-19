//! 无依赖的自旋锁。
//!
//! `BumpListAlloc` 的链表锁用它。Page 的页锁不使用本类型：Page 把 owner 指针的最低
//! 位当作页锁标记（见 `page_`）。锁不可重入；持有锁时不得调用可能再次获取同一把锁的
//! 代码。临界区应只包含指针操作、算术和 BlockHeader 字段写入。

use core::{
    hint::spin_loop,
    sync::atomic::{AtomicBool, Ordering},
};

/// 一把不可重入的自旋锁。
///
/// 嵌入 `BumpListCore`，因此使用 `#[repr(C)]` 固定为 1 字节的
/// `AtomicBool` 布局。
#[repr(C)]
#[derive(Debug)]
pub(crate) struct SpinLock {
    locked_: AtomicBool,
}

const _: () = assert!(core::mem::size_of::<SpinLock>() == 1);

impl SpinLock {
    /// 创建一把未锁定状态的自旋锁。
    pub(crate) const fn new() -> Self {
        Self {
            locked_: AtomicBool::new(false),
        }
    }

    /// 阻塞式获取锁。
    ///
    /// 成功后返回的 guard 在析构时释放锁。
    pub(crate) fn lock_(&self) -> SpinGuard<'_> {
        loop {
            if let Some(guard) = self.try_lock_() {
                return guard;
            }
            spin_loop();
        }
    }

    /// 尝试获取锁；锁已被占用时返回 [`None`]。
    pub(crate) fn try_lock_(&self) -> Option<SpinGuard<'_>> {
        if self
            .locked_
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            Some(SpinGuard {
                lock_: &self.locked_,
            })
        } else {
            None
        }
    }
}

/// 自旋锁的 RAII guard。
#[derive(Debug)]
pub(crate) struct SpinGuard<'a> {
    lock_: &'a AtomicBool,
}

impl Drop for SpinGuard<'_> {
    fn drop(&mut self) {
        self.lock_.store(false, Ordering::Release);
    }
}
