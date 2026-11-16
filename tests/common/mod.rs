//! 集成测试共用的计数分配器。
//!
//! 它把 Page 申请/释放转发给 `Global`，同时记录申请次数、释放次数与每次申请的
//! `layout.size()`，用来验证“所有 Page 同大小”和“空页归还”。

#![allow(dead_code)]

use std::{
    alloc::{AllocError, Allocator, Global, Layout},
    ptr::NonNull,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

/// 计数分配器共享的观测数据。
#[derive(Default)]
pub struct Counters {
    allocs: AtomicUsize,
    deallocs: AtomicUsize,
    sizes: Mutex<Vec<usize>>,
}

impl Counters {
    /// Page 申请次数。
    pub fn allocs(&self) -> usize {
        self.allocs.load(Ordering::Acquire)
    }

    /// Page 释放次数。
    pub fn deallocs(&self) -> usize {
        self.deallocs.load(Ordering::Acquire)
    }

    /// 每次 Page 申请时请求的字节数。
    pub fn sizes(&self) -> Vec<usize> {
        self.sizes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn record_alloc_(&self, size: usize) {
        self.allocs.fetch_add(1, Ordering::AcqRel);
        self.sizes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(size);
    }

    fn record_dealloc_(&self) {
        self.deallocs.fetch_add(1, Ordering::AcqRel);
    }
}

/// 把 Page 申请委托给 `Global` 的计数分配器。
#[derive(Clone)]
pub struct CountingAlloc {
    counters: Arc<Counters>,
}

impl CountingAlloc {
    /// 创建一个计数分配器和它的观测句柄。
    pub fn new() -> (Self, Arc<Counters>) {
        let counters = Arc::new(Counters::default());
        (
            Self {
                counters: Arc::clone(&counters),
            },
            counters,
        )
    }
}

unsafe impl Allocator for CountingAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let ptr = Global.allocate(layout)?;
        self.counters.record_alloc_(layout.size());
        Ok(ptr)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.counters.record_dealloc_();
        // SAFETY: 由 Allocator::deallocate 的调用方保证。
        unsafe { Global.deallocate(ptr, layout) };
    }
}
