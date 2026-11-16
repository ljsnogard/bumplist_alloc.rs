//! 有限页大小选项与页内 cell 上限。

use core::mem;

/// 一个 Page 内允许的最大 cell 数。
///
/// `BlockHeader` 使用 15 位记录自身在 Page 内的 cell 下标，最高位另作已释放标记，
/// 因此页内 cell 数不能超过 `1 << 15`。
pub const MAX_PAGE_CELLS: usize = 1 << 15;

/// 当前平台上单个 Page 允许的最大字节数。
///
/// 等于 [`MAX_PAGE_CELLS`] 乘以指针宽度：64 位平台为 256 KiB，32 位平台为 128 KiB。
pub const MAX_PAGE_BYTES: usize = MAX_PAGE_CELLS * mem::size_of::<usize>();

/// 可选的 Page 大小。
///
/// 页大小不是任意 `usize`，而是这组有限的 2 的幂选项。所有 Page 的大小必须一致；
/// 页大小超过 [`MAX_PAGE_CELLS`] 个 cell 时，构造 [`crate::BumpListAlloc`] 会失败。
///
/// # Examples
///
/// ```
/// use bumplist_alloc::{PageSize, MAX_PAGE_CELLS};
///
/// assert_eq!(PageSize::P4K.bytes(), 4 * 1024);
/// assert_eq!(PageSize::P4K.cells(), 4 * 1024 / core::mem::size_of::<usize>());
/// assert!(PageSize::P4K.is_supported());
/// assert!(PageSize::P1K.cells() <= MAX_PAGE_CELLS);
/// ```
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum PageSize {
    /// 1 KiB。
    P1K,
    /// 2 KiB。
    P2K,
    /// 4 KiB。
    P4K,
    /// 8 KiB。
    P8K,
    /// 16 KiB。
    P16K,
    /// 32 KiB。
    P32K,
    /// 64 KiB。
    P64K,
    /// 128 KiB。
    P128K,
    /// 256 KiB。32 位平台上超过 [`MAX_PAGE_CELLS`]，因此不受支持。
    P256K,
}

impl PageSize {
    /// 返回该选项对应的字节数。
    ///
    /// # Examples
    ///
    /// ```
    /// use bumplist_alloc::PageSize;
    ///
    /// assert_eq!(PageSize::P1K.bytes(), 1024);
    /// assert_eq!(PageSize::P8K.bytes(), 8192);
    /// ```
    pub const fn bytes(self) -> usize {
        match self {
            Self::P1K => 1024,
            Self::P2K => 2 * 1024,
            Self::P4K => 4 * 1024,
            Self::P8K => 8 * 1024,
            Self::P16K => 16 * 1024,
            Self::P32K => 32 * 1024,
            Self::P64K => 64 * 1024,
            Self::P128K => 128 * 1024,
            Self::P256K => 256 * 1024,
        }
    }

    /// 返回该页大小对应的页内 cell 数。
    ///
    /// 这个值在 32 位平台上可能超过 [`MAX_PAGE_CELLS`]；此时
    /// [`PageSize::is_supported`] 返回 `false`。
    ///
    /// # Examples
    ///
    /// ```
    /// use bumplist_alloc::PageSize;
    ///
    /// assert_eq!(
    ///     PageSize::P2K.cells(),
    ///     2 * 1024 / core::mem::size_of::<usize>(),
    /// );
    /// ```
    pub const fn cells(self) -> usize {
        self.bytes() / mem::size_of::<usize>()
    }

    /// 当前平台是否支持该页大小。
    ///
    /// # Examples
    ///
    /// ```
    /// use bumplist_alloc::PageSize;
    ///
    /// assert!(PageSize::P1K.is_supported());
    /// ```
    pub const fn is_supported(self) -> bool {
        self.cells() <= MAX_PAGE_CELLS
    }
}

#[cfg(test)]
mod tests_ {
    use super::*;

    /// 测试目标：每个 `PageSize` 的字节数、cell 数与支持性都符合公式。
    /// Test goal: Every `PageSize` reports the expected bytes, cells, and support flag.
    /// - 手段 / Method: 遍历 1 KiB 到 256 KiB 的选项，与 `1 << (10 + i)` 比较。
    /// - 判定 / Assertion: 字节数、cell 数、`is_supported` 与 `MAX_PAGE_CELLS` 一致。
    #[test]
    fn page_sizes_match_bytes_and_cells() {
        let sizes = [
            PageSize::P1K,
            PageSize::P2K,
            PageSize::P4K,
            PageSize::P8K,
            PageSize::P16K,
            PageSize::P32K,
            PageSize::P64K,
            PageSize::P128K,
            PageSize::P256K,
        ];
        for (index, size) in sizes.into_iter().enumerate() {
            let expected = 1usize << (10 + index);
            assert_eq!(size.bytes(), expected);
            assert_eq!(size.cells(), expected / mem::size_of::<usize>());
            assert_eq!(size.is_supported(), size.cells() <= MAX_PAGE_CELLS);
        }
        assert_eq!(MAX_PAGE_BYTES, MAX_PAGE_CELLS * mem::size_of::<usize>());
    }

    /// 测试目标：256 KiB 选项的支持性随目标指针宽度变化。
    /// Test goal: Support for the 256 KiB option depends on the target pointer width.
    /// - 手段 / Method: 按 `target_pointer_width` 检查 `P256K.is_supported()`。
    /// - 判定 / Assertion: 64 位为真；32 位为假。
    #[test]
    fn p256k_support_depends_on_pointer_width() {
        #[cfg(target_pointer_width = "64")]
        assert!(PageSize::P256K.is_supported());

        #[cfg(target_pointer_width = "32")]
        assert!(!PageSize::P256K.is_supported());
    }
}
