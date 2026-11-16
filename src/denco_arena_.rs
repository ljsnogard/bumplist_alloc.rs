//! `abs_denco` 的 arena 适配（`denco_arena` 特性）。
//!
//! 本模块把 [`BumpListAlloc`](crate::BumpListAlloc) 实现成
//! [`TrArena`](abs_denco::de::TrArena) 与 [`TrArenaBytes`](abs_denco::de::TrArenaBytes)：
//!
//! - `Ref<T>` 是 [`BumpListBox`]：一个拥有 `T` 的特制 Box，析构时运行 `T` 的析构函数，
//!   并通过块头反推 `Page`、调用 [`Page::release_`] 把块归还到页内（空页会重置为可复用
//!   状态）；页链表的摘链 / 底层归还仍由 [`BumpListAlloc`] 的回收路径负责；
//! - 分配走本 crate 已有的 [`Allocator`](core::alloc::Allocator) 实现（只对
//!   `&BumpListAlloc` 实现）；
//! - `TrArenaBytes` 交出的是借用切片，字节没有 Drop 语义；它们同样随分配器析构整体释放。
//!
//! 这不是 `BumpListAlloc` 的默认能力：只有开启 `denco_arena` 特性才会引入
//! `abs_denco` 依赖与这两个实现。
//!
//! # `denco_arena_`: the `abs_denco` arena adapter
//!
//! Behind the `denco_arena` feature, `BumpListAlloc` implements `TrArena` / `TrArenaBytes`.
//! `TrArena::Ref<T>` is a `BumpListBox` that runs `T`'s destructor and returns the block to
//! its `Page` via `Page::release_` on drop.

use core::{
    alloc::{Allocator, Layout},
    error::Error,
    fmt, future,
    marker::PhantomData,
    mem,
    ops::{Deref, DerefMut},
    ptr::{self, NonNull},
    slice,
};

use abs_denco::{
    TrBoxed, TrUnique,
    de::{TrArena, TrArenaBytes},
};

use crate::{
    BumpListAlloc,
    page_::{ORDER_MASK, PTR, Page},
};

/// [`BumpListAlloc`] 的独占句柄，作为 [`TrArena::Ref`]。
///
/// 它拥有 arena 里的一块 `T`：析构时运行 `T` 的析构函数，并通过块头反推所属 `Page`、
/// 调用 [`Page::release_`] 把块归还到页内。它不需要知道 `BumpListAlloc` 的类型，也不
/// 持有分配器引用。
///
/// A unique handle into a `BumpListAlloc`, used as `TrArena::Ref`. It owns one `T`, runs
/// `T`'s destructor on drop, and returns the block to its `Page` via `Page::release_`.
/// It needs neither the allocator type nor an allocator reference.
pub struct BumpListBox<'alloc, T> {
    /// arena 里的值。
    ptr_: NonNull<T>,

    /// 只把 `'alloc` 带进类型，不持有分配器。
    _marker_: PhantomData<&'alloc T>,
}

impl<'alloc, T> BumpListBox<'alloc, T> {
    /// 从 arena 分配出的裸指针接管所有权。
    pub(crate) fn from_raw_(raw: NonNull<T>) -> Self {
        BumpListBox {
            ptr_: raw,
            _marker_: PhantomData,
        }
    }
}

impl<T> Deref for BumpListBox<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: `ptr_` 指向一块已写入、由本句柄独占的 `T`。
        unsafe { self.ptr_.as_ref() }
    }
}

impl<T> DerefMut for BumpListBox<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: `ptr_` 指向一块已写入、由本句柄独占的 `T`。
        unsafe { self.ptr_.as_mut() }
    }
}

impl<T> TrBoxed for BumpListBox<'_, T> {
    type Item = T;
}

impl<T> TrUnique for BumpListBox<'_, T> {}

impl<T> Drop for BumpListBox<'_, T> {
    fn drop(&mut self) {
        // SAFETY: `ptr_` 来自 arena、按 `Layout::new::<T>()` 分配且只写入一次；
        // `drop_in_place` 之后这块内存不再被读取。
        unsafe {
            ptr::drop_in_place(self.ptr_.as_ptr());
        }
        if mem::size_of::<T>() == 0 {
            // 零尺寸分配没有 BlockHeader，也没有 Page 需要归还。
            return;
        }
        // SAFETY: 非零尺寸块由 `alloc_value_async` 通过 `BumpListAlloc` 分配，用户块前
        // 紧邻一个 `BlockHeader`；头里的 cell 下标可 O(1) 反推 Page 首地址。块此刻仍然
        // 活跃，Page 在归还前有效。
        unsafe {
            let header_addr = self.ptr_.as_ptr().cast::<u8>().sub(PTR);
            let flags = (header_addr as *const u16).read();
            let header_cell = flags & ORDER_MASK;
            let page_addr = header_addr.sub(usize::from(header_cell) * PTR);
            let page = NonNull::new_unchecked(page_addr as *mut Page);
            Page::release_page_(page, header_cell);
        }
    }
}

/// `TrArena` 分配失败。
///
/// [`core::alloc::AllocError`] 不实现 [`Error`]，因此这里提供一个薄适配。它只在
/// `denco_arena` 特性下存在。
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BumpListArenaError;

impl fmt::Display for BumpListArenaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("BumpListAlloc arena allocation failed")
    }
}

impl Error for BumpListArenaError {}

impl<'alloc, A> TrArena<'alloc> for BumpListAlloc<A>
where
    A: Allocator,
{
    type Error = BumpListArenaError;

    type Ref<T: 'alloc>
        = BumpListBox<'alloc, T>
    where
        Self: 'alloc;

    type AllocValueAsync<'a, T>
        = future::Ready<Result<BumpListBox<'alloc, T>, BumpListArenaError>>
    where
        'alloc: 'a,
        T: 'alloc + 'a,
        Self: 'alloc + 'a;

    fn alloc_value_async<'a, T>(&'alloc self, value: T) -> Self::AllocValueAsync<'a, T>
    where
        'alloc: 'a,
        T: 'alloc + 'a,
        Self: 'alloc + 'a,
    {
        let layout = Layout::new::<T>();
        let result = match <&Self as Allocator>::allocate(&self, layout) {
            Result::Ok(block) => {
                let ptr = block.as_ptr().cast::<T>();
                // SAFETY: `allocate` 返回一个按 `layout` 对齐、大小至少 `size_of::<T>()`
                // 的未初始化块；这里写入一次，之后由 `BumpListBox` 独占，析构时运行
                // `T` 的析构并归还块。
                unsafe {
                    ptr.write(value);
                    Result::Ok(BumpListBox::from_raw_(NonNull::new_unchecked(ptr)))
                }
            }
            Result::Err(_) => Result::Err(BumpListArenaError),
        };
        future::ready(result)
    }
}

impl<'alloc, A> TrArenaBytes<'alloc> for BumpListAlloc<A>
where
    A: Allocator,
{
    type AllocBytesAsync<'a>
        = future::Ready<Result<&'alloc [u8], BumpListArenaError>>
    where
        'alloc: 'a,
        Self: 'alloc + 'a;

    fn alloc_bytes_async<'a>(&'alloc self, src: &'a [u8]) -> Self::AllocBytesAsync<'a>
    where
        'alloc: 'a,
        Self: 'alloc + 'a,
    {
        let result = match <&Self as Allocator>::allocate(&self, Layout::for_value(src)) {
            Result::Ok(block) => {
                let ptr = block.as_ptr().cast::<u8>();
                // SAFETY: `allocate` 返回一个按 `u8` 对齐、长度不小于 `src.len()` 的
                // 未初始化块；这里只写入一次，之后由 arena 持有到分配器析构。切片只以
                // 共享引用交出，不与其它分配重叠。
                unsafe {
                    core::ptr::copy_nonoverlapping(src.as_ptr(), ptr, src.len());
                    Result::Ok(slice::from_raw_parts(ptr, src.len()))
                }
            }
            Result::Err(_) => Result::Err(BumpListArenaError),
        };
        future::ready(result)
    }
}
