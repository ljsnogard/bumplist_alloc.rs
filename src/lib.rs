//! `bumplist-alloc`：以固定大小 Page 链表组织的 bump 分配器。
//!
//! 设计目标：
//!
//! - 标准 [`core::alloc::Allocator`]：对 `&BumpListAlloc` 直接实现；`Arc` / `Rc` /
//!   `Box` 等稳定 `Deref` 句柄通过 [`BumpListAllocRef`] 适配。每块分配仍带有借用
//!   生命周期；
//! - `A: Send` 时 `Send + Sync`（不要求 `A: Sync`），但不实现 `Clone`；
//! - 每个实例的所有 Page 大小相同，页大小从有限的 [`PageSize`] 选项中选取；
//! - Page 之间用指针串成双向链表，Page 头与块头都内嵌在 Page 内存中；
//! - 每个 Page 有独立的页锁，普通归还只需要页锁；只有空页摘链才需要 allocator 的
//!   链表锁；
//! - 普通页整页块全部释放后按策略保留一个空页或归还底层分配器；第一页承载分配器
//!   元数据，永久保留。
//!
//! # 基本用法
//!
//! ```
//! use bumplist_alloc::{BumpListAlloc, PageSize};
//! use std::vec::Vec;
//!
//! let bump = BumpListAlloc::new(PageSize::P4K)?;
//! let mut values: Vec<u32, &BumpListAlloc> = Vec::new_in(&bump);
//! values.extend_from_slice(&[1, 2, 3, 4]);
//! assert_eq!(&values[..], &[1, 2, 3, 4]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # 固定页大小
//!
//! 同一个分配器实例的所有 Page 大小相同。一个分配块必须完整落在一个 Page 内；超过
//! 页可容纳空间的请求会返回 [`core::alloc::AllocError`]，不会跨页，也不会临时申请大页。
//!
//! # 单指针句柄与第一页元数据
//!
//! 与 `A` 无关的页链表、页大小与链表锁放在非泛型 `BumpListCore` 中；Core、Page 头
//! 与底层分配器 `A` 一起内嵌在第一页的 `InitPage<A>` 里。`BumpListAlloc<A>` 只是
//! `NonNull<InitPage<A>>`，与一个指针同宽，因此可以安全移动；Page 的 owner 指向第一页
//! 里地址稳定的 Core，不依赖句柄自身的地址。
//!
//! `BumpListCore` 保存一个指向 `A` 的 `NonNull<dyn Allocator>`，所有对 `A` 的
//! allocate / deallocate 都通过它转发，并由 `backing_lock_` 串行化。第一页在构造时
//! 申请；析构时先把 `A` 移出第一页，再用移出的 `A` 归还第一页本身。
//!
//! # `no_std` 与 `std` 特性
//!
//! 默认 `no_std`，只依赖 `core` 与 `alloc`。开启 `std` 特性后，[`BumpListAlloc`]
//! 析构时会扫描页内块链，并在发现未释放块时把警告写到 stderr；默认构建不输出警告，
//! 但仍然会通过块链检查页状态。

#![feature(allocator_ext, coerce_unsized, unsize)]
#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

extern crate alloc;

#[cfg(any(test, feature = "std"))]
extern crate std;

mod alloc_;
mod alloc_ref_;
mod boxed_;
#[cfg(feature = "denco_arena")]
mod denco_arena_;
mod error_;
mod init_page_;
mod lock_;
mod page_;
mod page_size_;

pub use alloc_::BumpListAlloc;
pub use alloc_ref_::BumpListAllocRef;
pub use boxed_::Box;
#[cfg(feature = "denco_arena")]
pub use denco_arena_::BumpListArenaError;
pub use error_::BumpListAllocError;
pub use page_size_::{MAX_PAGE_BYTES, MAX_PAGE_CELLS, PageSize};

/// 导出依赖的 crate
pub mod x_deps {
    pub use abs_mm;

    #[cfg(feature = "denco_arena")]
    pub use abs_denco;
}
