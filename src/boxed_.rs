use core::{
    alloc::{Allocator, Layout}, marker::PhantomData, mem, ops::{Deref, DerefMut}, ptr::{self, NonNull},
};

use abs_mm::res_man::{TrBoxed, TrUnique};

use crate::{BumpListAlloc, page_::{ORDER_MASK, PTR, Page}};

/// [`BumpListAlloc`] 的独占句柄，作为 [`TrArena::Ref`]。
///
/// 它拥有 arena 里的一块 `T`：析构时运行 `T` 的析构函数，并通过块头反推所属 `Page`、
/// 调用 [`Page::release_`] 把块归还到页内。它不需要知道 `BumpListAlloc` 的类型，也不
/// 持有分配器引用。
///
/// A unique handle into a `BumpListAlloc`, used as `TrArena::Ref`. It owns one `T`, runs
/// `T`'s destructor on drop, and returns the block to its `Page` via `Page::release_`.
/// It needs neither the allocator type nor an allocator reference.
pub struct Box<'a, T> {
    /// arena 里的值。
    ptr_: NonNull<T>,

    /// 只把 `'alloc` 带进类型，不持有分配器。
    _marker_: PhantomData<&'a T>,
}

impl<'a, T> Box<'a, T> {
    /// 从 arena 分配出的裸指针接管所有权。
    #[cfg(feature = "denco_arena")]
    pub(crate) fn from_raw_(raw: NonNull<T>) -> Self {
        Box {
            ptr_: raw,
            _marker_: PhantomData,
        }
    }

    unsafe fn locate_page_(&self) -> (NonNull<Page>, u16) {
        unsafe {
            let header_addr = self.ptr_.as_ptr().cast::<u8>().sub(PTR);
            let flags = (header_addr as *const u16).read();
            let header_cell = flags & ORDER_MASK;
            let page_addr = header_addr.sub(usize::from(header_cell) * PTR);
            (NonNull::new_unchecked(page_addr as *mut Page), header_cell)
        }
    }
}

impl<T> Deref for Box<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: `ptr_` 指向一块已写入、由本句柄独占的 `T`。
        unsafe { self.ptr_.as_ref() }
    }
}

impl<T> DerefMut for Box<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: `ptr_` 指向一块已写入、由本句柄独占的 `T`。
        unsafe { self.ptr_.as_mut() }
    }
}

impl<'a, T> TrBoxed for Box<'a, T> {
    type Item = T;
}

impl<T> TrUnique for Box<'_, T> {}

impl<T> Drop for Box<'_, T> {
    fn drop(&mut self) {
        // SAFETY: `ptr_` 来自 arena、按 `Layout::new::<T>()` 分配且只写入一次；
        // `drop_in_place` 之后这块内存不再被读取。
        unsafe { ptr::drop_in_place(self.ptr_.as_ptr()) };
        if mem::size_of::<T>() == 0 {
            // 零尺寸分配没有 BlockHeader，也没有 Page 需要归还。
            return;
        }
        // SAFETY: 非零尺寸块由 `alloc_value_async` 通过 `BumpListAlloc` 分配，用户块前
        // 紧邻一个 `BlockHeader`；头里的 cell 下标可 O(1) 反推 Page 首地址。块此刻仍然
        // 活跃，Page 在归还前有效。
        unsafe {
            let (page, header_cell) = self.locate_page_();
            Page::release_page_(page, header_cell);
        }
    }
}
