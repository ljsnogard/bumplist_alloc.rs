//! 大页：只放一个块的专用 Page。
//!
//! 由 `big_page` 特性启用。一次超过普通页容量的分配会申请一块独立的 backing 内存，
//! 在其中构造 `Page` + [`BigPage`] 扩展头 + 一个 `BlockHeader`，用户块紧跟在块头之后。
//! 块头仍位于页内很小的 cell 下标 `BIG_PAGE_CELLS`，因此 15 位 `order_flags_` /
//! `next_free_` 足以让现有 `release_locked_` 把整页判空；真实的数据大小只记录在
//! `alloc_size_` 里，由 `dealloc_parts_` 用于归还 backing。
//!
//! 大页与普通页分属两条链表（见 `alloc_::BigPageList` 的使用），因此普通页的分配、
//! 释放与空页缓存完全看不到大页。

use core::{alloc::Layout, mem, ptr::NonNull};

use crate::{
    alloc_::BumpListCore,
    page_::{CellIndex, LARGE_BIT, PTR, Page},
    page_size_::MAX_PAGE_CELLS,
};

/// 大页的页头 + 扩展元数据。
///
/// `Page` 必须在偏移 0：`NonNull<BigPage>` 与 `NonNull<Page>` 同址，链表与页锁都按
/// `Page` 首地址工作。三个 `usize` 记录 backing 的原始布局；用户块从
/// `base + (BIG_PAGE_CELLS + 1) * PTR` 开始，因此不会覆盖这些字段。
#[repr(C)]
pub(crate) struct BigPage {
    /// 必须是第一个字段。
    pub(crate) page_: Page,
    /// backing 分配返回的原始指针（可能早于 Page 首地址）。
    raw_: NonNull<u8>,
    /// 申请 / 归还 backing 时使用的 size。
    alloc_size_: usize,
    /// 申请 / 归还 backing 时使用的 align。
    alloc_align_: usize,
}

/// 大页 Page 头占用的 cell 数。
pub(crate) const BIG_PAGE_CELLS: usize = mem::size_of::<BigPage>().div_ceil(PTR);

const _: () = {
    assert!(mem::offset_of!(BigPage, page_) == 0);
    assert!(BIG_PAGE_CELLS < MAX_PAGE_CELLS);
};

/// 大页的几何参数：块头 cell、backing 布局。
#[derive(Clone, Copy)]
pub(crate) struct BigGeometry {
    header_cell_: CellIndex,
    alloc_size_: usize,
    alloc_align_: usize,
}

impl BigGeometry {
    /// 由用户 `Layout` 计算 backing 布局。
    ///
    /// 返回 [`None`] 表示尺寸溢出或布局非法；调用方应把它映射为 `AllocError`。
    pub(crate) fn new_(layout: Layout) -> Option<Self> {
        let align = layout.align().max(PTR);
        let header_cell = BIG_PAGE_CELLS;
        let head_bytes = (header_cell + 1).checked_mul(PTR)?;
        let alloc_size = head_bytes
            .checked_add(layout.size())?
            .checked_add(align - 1)?;
        // 提前校验：既检查 size + align - 1 不溢出 isize::MAX，也保证后续
        // from_size_align_unchecked 的前提成立。
        Layout::from_size_align(alloc_size, align).ok()?;
        Some(Self {
            header_cell_: header_cell as CellIndex,
            alloc_size_: alloc_size,
            alloc_align_: align,
        })
    }

    /// 申请 backing 时使用的布局。
    pub(crate) fn alloc_layout_(&self) -> Layout {
        // SAFETY: new_ 已用 Layout::from_size_align 校验过同样的 size/align。
        unsafe { Layout::from_size_align_unchecked(self.alloc_size_, self.alloc_align_) }
    }

    /// 由 backing 返回的 `raw` 计算 `(Page 首地址, 用户块首地址)`。
    ///
    /// 用户块优先：先在 `raw + head_bytes` 之后按 `alloc_align_` 对齐得到 `user`，
    /// 再由 `user` 反推 `base`。这样用户块与块头紧邻，块头的 cell 下标始终是
    /// [`BIG_PAGE_CELLS`]，不会被巨大的对齐 padding 撑大。
    ///
    /// 返回 [`None`] 表示地址运算溢出；在 [`BigGeometry::new_`] 成功时不应发生，调用方
    /// 应归还刚申请的 `raw` 并把请求映射为 `AllocError`。
    ///
    /// # Safety
    ///
    /// `raw` 必须来自 [`BigGeometry::alloc_layout_`] 的 `allocate`，且仍然独占。
    pub(crate) unsafe fn place_(&self, raw: NonNull<u8>) -> Option<(NonNull<Page>, NonNull<u8>)> {
        let head_bytes = (self.header_cell_ as usize + 1) * PTR;
        let raw_addr = raw.as_ptr().addr();
        let start = raw_addr.checked_add(head_bytes)?;
        let user_addr = align_up_(start, self.alloc_align_)?;
        let offset = user_addr.checked_sub(raw_addr)?;
        if offset > self.alloc_size_ {
            return None;
        }
        // SAFETY: offset 落在 raw 的分配内，且 raw 仍然独占。
        let user = unsafe { raw.add(offset) };
        // SAFETY: head_bytes <= offset，base 仍在分配内。
        let base = unsafe { user.sub(head_bytes) };
        Some((base.cast::<Page>(), user))
    }
}

impl BigPage {
    /// 初始化大页的 `Page` 头、扩展字段与单块块头。
    ///
    /// # Safety
    ///
    /// `base` / `raw` / `geometry` 必须来自同一个 [`BigGeometry::place_`] 结果；
    /// 内存刚申请、独占、尚未共享；`core` 在 Page 生命周期内地址稳定。
    pub(crate) unsafe fn init_(
        base: NonNull<Page>,
        raw: NonNull<u8>,
        geometry: &BigGeometry,
        core: NonNull<BumpListCore>,
    ) -> NonNull<Page> {
        let head_bytes = (geometry.header_cell_ as usize + 1) * PTR;
        let mem = NonNull::slice_from_raw_parts(base.cast::<u8>(), head_bytes);
        // SAFETY: slice 覆盖 Page + BigPage 扩展字段 + 块头；内存独占且对齐。
        let page =
            unsafe { Page::init_in_place_with_tag_(mem, core, geometry.header_cell_, LARGE_BIT) };
        let big = page.cast::<BigPage>().as_ptr();
        // SAFETY: 扩展字段位于 [base, base + head_bytes) 内，且尚未共享。
        unsafe {
            (*big).raw_ = raw;
            (*big).alloc_size_ = geometry.alloc_size_;
            (*big).alloc_align_ = geometry.alloc_align_;
            debug_assert_eq!(
                core::ptr::addr_of!((*big).page_).cast::<u8>(),
                base.cast::<u8>().as_ptr()
            );
            (*page.as_ref()).init_single_block_(page, geometry.header_cell_);
        }
        page
    }

    /// 读出归还 backing 所需的原始指针与布局。
    ///
    /// # Safety
    ///
    /// `page` 必须是仍然存活、已经初始化的大页；调用后该页立即失效。
    pub(crate) unsafe fn dealloc_parts_(page: NonNull<Page>) -> (NonNull<u8>, Layout) {
        let big = page.cast::<BigPage>().as_ptr();
        // SAFETY: 扩展字段位于页头内，且调用方保证 page 是有效大页。
        let (raw, size, align) = unsafe { ((*big).raw_, (*big).alloc_size_, (*big).alloc_align_) };
        // SAFETY: size/align 正是 BigGeometry::new_ 校验并用于 allocate 的布局。
        let layout = unsafe { Layout::from_size_align_unchecked(size, align) };
        (raw, layout)
    }
}

fn align_up_(addr: usize, align: usize) -> Option<usize> {
    debug_assert!(align.is_power_of_two());
    let mask = align - 1;
    addr.checked_add(mask).map(|value| value & !mask)
}
