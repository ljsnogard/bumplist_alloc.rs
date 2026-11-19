//! `BumpListAlloc` 的构造错误。

use core::fmt;

use crate::page_size_::{MAX_PAGE_CELLS, PageSize};

/// 创建 [`crate::BumpListAlloc`] 时可能出现的错误。
///
/// # Examples
///
/// ```
/// use bumplist_alloc::{BumpListAlloc, BumpListAllocError, PageSize};
///
/// let result = BumpListAlloc::new(PageSize::P4K);
/// assert!(result.is_ok());
/// # let _ = result;
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BumpListAllocError {
    /// 当前平台不支持请求的页大小。
    ///
    /// 典型情况是 32 位平台上的 [`PageSize::P256K`]：它的页内 cell 数超过
    /// [`MAX_PAGE_CELLS`]，无法用 15 位位置下标表示。
    UnsupportedPageSize {
        /// 被拒绝的页大小。
        page_size: PageSize,
    },

    /// 页大小本身受平台支持，但第一页扣除 `Page`、`BumpListCore` 与底层分配器 `A`
    /// 的存储后，剩余空间不足以容纳一个最小分配块（块头 + 一个数据 cell）。
    ///
    /// 第一页同时保存分配器元数据，因此它的可用容量比普通页小；典型情况是底层
    /// 分配器 `A` 自身较大，或选择了该分配器无法承载的小页。
    PageSizeTooSmallForAllocator {
        /// 被拒绝的页大小。
        page_size: PageSize,
    },

    /// 申请第一页（包含 `Page`、`BumpListCore` 与 `A`）时，底层分配器返回失败。
    InitPageAllocationFailed {
        /// 请求的页大小。
        page_size: PageSize,
    },
}

impl fmt::Display for BumpListAllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPageSize { page_size } => write!(
                f,
                "页大小 {page_size:?} 在当前平台不受支持（页内 cell 数上限为 {MAX_PAGE_CELLS}）"
            ),
            Self::PageSizeTooSmallForAllocator { page_size } => write!(
                f,
                "页大小 {page_size:?} 不足以容纳分配器元数据（Page、BumpListCore 与底层分配器）及最小分配块"
            ),
            Self::InitPageAllocationFailed { page_size } => write!(
                f,
                "底层分配器无法提供 {page_size:?} 的第一页（分配器元数据页）"
            ),
        }
    }
}

impl core::error::Error for BumpListAllocError {}
