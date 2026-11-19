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

    type Ref<T: 'alloc> = BumpListBox<'alloc, T> where Self: 'alloc;

    type AllocValueAsync<'a, T> =
        future::Ready<Result<BumpListBox<'alloc, T>, BumpListArenaError>>
    where
        'alloc: 'a,
        T: 'alloc + 'a,
        Self: 'alloc + 'a;

    fn alloc_value_async<'a, T>(
        &'alloc self,
        value: T,
    ) -> Self::AllocValueAsync<'a, T>
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
    type AllocBytesAsync<'f> =
        future::Ready<Result<&'alloc [u8], BumpListArenaError>>
    where
        'alloc: 'f,
        Self: 'alloc + 'f;

    fn alloc_bytes_async<'f>(
        &'alloc self,
        src: &'f [u8],
    ) -> Self::AllocBytesAsync<'f>
    where
        'alloc: 'f,
        Self: 'alloc + 'f,
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
