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
}

impl fmt::Display for BumpListAllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPageSize { page_size } => write!(
                f,
                "页大小 {page_size:?} 在当前平台不受支持（页内 cell 数上限为 {MAX_PAGE_CELLS}）"
            ),
        }
    }
}

impl core::error::Error for BumpListAllocError {}
