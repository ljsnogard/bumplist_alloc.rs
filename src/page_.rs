//! Page 头、BlockHeader、页锁与页内块链。
//!
//! # 布局
//!
//! ```text
//! Page base
//! ├── Page { links, lock, state }
//! ├── ... padding 到首个可分配 cell ...
//! ├── BlockHeader cell (order_flags, next_free, reserved)
//! ├── ... 对齐 padding ...
//! ├── user block
//! ├── ...
//! ```
//!
//! Page 内以指针宽度 `P = size_of::<usize>()` 为一个 cell。`PageState.base_` 是首个
//! 可分配 cell 的页内绝对下标（可以大于 `Page::CELLS`，页与页可以不同），`cells_`
//! 是从 `base_` 起可用的 cell 数。普通页的 `base_` 是 `Page::CELLS`；第一页在 Page
//! 头之后还内嵌了 `BumpListCore` 与底层分配器 `A`（见 `init_page_`），因此 `base_`
//! 更大、容量更小。`BlockHeader.order_flags` 的低 15 位是自身 cell
//! 的页内绝对下标，最高位表示该块已经释放；`next_free` 是本块末尾（也就是下一块头）
//! 的 cell 下标。最后一块的 `next_free` 等于 `base_ + used_`，因此扫描时不需要
//! `first_` / `last_`，直接从 `base_` 沿链走到 `base_ + used_` 即可。
//!
//! 分配时块头放在当前 bump 位置，用户数据在块头之后按 `Layout::align()` 对齐；
//! 块头与用户数据之间的 padding 属于本块。若 padding 存在，紧邻用户块之前的
//! padding cell 会写入 locator（最高位置 [`LOCATOR_BIT`]，低 15 位是到块头的 cell
//! 距离），这样 `Allocator::deallocate` / `Box::drop` 仍能从用户指针 O(1) 找到块头。
//! 没有 padding 时，紧邻用户块的 cell 就是块头的 `order_flags`（最高位为 0）。
//!
//! 释放块只设置 `FREE_BIT`，并把本块的 cell 数累加到 `PageState.returned_`；当
//! `returned_ == used_` 时整页已经全部归还。这同时支持不同页使用不同的
//! `base_` / `cells_`。

use core::{
    alloc::Layout,
    cell::UnsafeCell,
    hint::spin_loop,
    mem,
    ptr::NonNull,
    sync::atomic::{AtomicPtr, Ordering},
};

use crate::{alloc_::BumpListCore, page_size_::MAX_PAGE_CELLS};

pub(crate) type CellIndex = u16;

/// 指针宽度，也是 Page 内一个 cell 的字节数。
pub(crate) const PTR: usize = mem::size_of::<usize>();

/// `order_flags` 的最高位：该块已释放。
pub(crate) const FREE_BIT: CellIndex = 1 << 15;

/// `order_flags` 中保存 cell 下标的低位掩码。
pub(crate) const ORDER_MASK: CellIndex = FREE_BIT - 1;

/// 用户块前有对齐 padding 时，紧邻用户块的 locator cell 的最高位。
///
/// locator 的低 15 位保存“紧邻用户块的 cell 到块头的 cell 距离”。没有 padding 时
/// 该 cell 就是活跃块头的 `order_flags`，最高位为 0，因此可以安全地区分。
pub(crate) const LOCATOR_BIT: CellIndex = 1 << 15;

/// 页内块头。
///
/// 字段布局在 64 位平台占 8 字节、32 位平台占 4 字节；`_reserved` 在 64 位平台保留
/// 4 字节，在 32 位平台长度为 0。块头位于 cell 边界，但只使用开头的两个 `u16`。
#[repr(C)]
pub(crate) struct BlockHeader {
    /// 低 15 位：自身 cell 下标；最高位：已释放标记。
    pub(crate) order_flags_: CellIndex,
    /// 本块末尾（下一块头 / bump 末尾）的页内绝对 cell 下标。
    pub(crate) next_free_: CellIndex,
    /// 保留字节，当前不读写。
    _reserved_: [u8; PTR - 4],
}

const _: () = assert!(mem::size_of::<BlockHeader>() == PTR);

/// Page 链表的链接字段；由 `BumpListAlloc` 的链表锁保护。
#[repr(C)]
pub(crate) struct PageLinks {
    pub(crate) prev_: Option<NonNull<Page>>,
    pub(crate) next_: Option<NonNull<Page>>,
}

/// Page 的可变状态；由本页的页锁保护。
///
/// 四个字段都是 `u16`，`repr(C)` 下正好 8 字节，没有内部 padding。
///
/// - `base_` / `cells_` 是页的初始参数：首个可分配 cell 的页内绝对下标，以及从它
///   开始可用的 cell 数；
/// - `used_` 是相对 `base_` 的 bump 偏移（已消耗的 cell 数）；
/// - `returned_` 是相对 `base_` 的已归还 cell 数。
///
/// 整页是否全部归还只需比较 `returned_ == used_`；空页复用后两者都归零。
#[repr(C)]
pub(crate) struct PageState {
    /// 首个可分配 cell 的页内绝对下标。
    pub(crate) base_: CellIndex,
    /// 从 `base_` 起可分配的 cell 数。
    pub(crate) cells_: CellIndex,
    /// 相对 `base_` 的当前 bump 偏移。
    pub(crate) used_: CellIndex,
    /// 相对 `base_` 的已归还 cell 数。
    pub(crate) returned_: CellIndex,
}

/// Page 头。
///
/// 字段按大小降序手动布置：`links_`（2 个指针）、`state_`（8 字节）、`owner_`
/// （指向非泛型 [`BumpListCore`] 的 owner 指针，低 1 位兼作页锁；开启 `big_page` 时
/// 次低位标记大页）。64 位下 32 字节，32 位下 20 字节，分别占 4/5 个 cell。
///
/// `links_` 由 `BumpListCore` 的链表锁保护，`state_` 由页锁保护。owner 指针指向
/// 创建本页的 `BumpListCore`；它的最低位一定是 0（Core 至少按指针宽度对齐），因此
/// 用最低位作为页锁标记，不额外占一个锁字段。空页需要摘链 / 归还底层分配器时，
/// `release_page_` 直接调用 `BumpListCore::cleanup_empty_page_`，不需要每个 Page 保存
/// 函数指针。
///
/// Page 本身不对外暴露，只通过稳定的 Page 首地址访问；页内写入从 `NonNull<Page>` 的
/// 原始指针偏移，不依赖 `&Page` 的 provenance。
#[repr(C)]
pub(crate) struct Page {
    links_: UnsafeCell<PageLinks>,
    state_: UnsafeCell<PageState>,
    owner_: AtomicPtr<BumpListCore>,
}

/// owner 指针的最低有效位：页锁占用标记。
const LOCK_BIT: usize = 1;

/// owner 指针的次低位：大页标记；仅在 `big_page` 特性开启时使用。
#[cfg(feature = "big_page")]
pub(crate) const LARGE_BIT: usize = 1 << 1;

/// 从 owner 指针中剥离所有标记位。
#[cfg(feature = "big_page")]
const PAGE_TAG_MASK: usize = LOCK_BIT | LARGE_BIT;

/// 未开启大页特性时，owner 指针只有页锁一位标记。
#[cfg(not(feature = "big_page"))]
const PAGE_TAG_MASK: usize = LOCK_BIT;

const _: () = assert!(mem::size_of::<PageState>() == 8);
const _: () = assert!(mem::size_of::<PageLinks>() == 2 * PTR);
#[cfg(target_pointer_width = "64")]
const _: () = assert!(mem::size_of::<Page>() == 4 * PTR);
#[cfg(target_pointer_width = "32")]
const _: () = assert!(mem::size_of::<Page>() == 5 * PTR);

/// 页锁的 RAII guard：最高位之外只用最低位，析构时清掉锁标记。
#[derive(Debug)]
pub(crate) struct PageGuard<'a> {
    owner_: &'a AtomicPtr<BumpListCore>,
}

impl Drop for PageGuard<'_> {
    fn drop(&mut self) {
        let mut current = self.owner_.load(Ordering::Relaxed);
        loop {
            let cleared = current.map_addr(|addr| addr & !LOCK_BIT);
            match self.owner_.compare_exchange_weak(
                current,
                cleared,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Result::Ok(_) => return,
                Result::Err(actual) => current = actual,
            }
        }
    }
}

impl Page {
    /// Page 头占用的 cell 数。
    pub(crate) const CELLS: usize = mem::size_of::<Self>().div_ceil(PTR);

    /// 在刚申请的独占内存上初始化 Page 头。
    ///
    /// # Safety
    ///
    /// `mem` 必须是至少 `page_size` 字节、按 `align_of::<Page>()` 对齐、尚未被写入的
    /// 独占内存块；调用方保证它来自同一个 `BumpListAlloc` 的 backing allocator。
    pub(crate) unsafe fn init_in_place_(
        mem: NonNull<[u8]>,
        owner: NonNull<BumpListCore>,
        first_cell: CellIndex,
    ) -> NonNull<Self> {
        // SAFETY: 由调用方保证 init_in_place_with_tag_ 的契约；普通页不带 kind 标记。
        unsafe { Self::init_in_place_with_tag_(mem, owner, first_cell, 0) }
    }

    /// 与 [`Page::init_in_place_`] 相同，但允许在 owner 指针上附加 kind 标记。
    ///
    /// `tag` 只能是 0 或 `LARGE_BIT`；大页初始化由 `big_page_` 传入
    /// `LARGE_BIT`。
    ///
    /// # Safety
    ///
    /// 同 [`Page::init_in_place_`]；调用方必须在挂链前独占本页。
    pub(crate) unsafe fn init_in_place_with_tag_(
        mem: NonNull<[u8]>,
        owner: NonNull<BumpListCore>,
        first_cell: CellIndex,
        tag: usize,
    ) -> NonNull<Self> {
        let page_cells = mem.len() / PTR;
        let first = first_cell as usize;
        assert!(
            first >= Self::CELLS,
            "first_cell must not overlap the Page header"
        );
        assert!(
            first < page_cells,
            "first_cell must leave at least one cell"
        );
        let cells = page_cells - first;
        assert!(
            cells <= MAX_PAGE_CELLS,
            "allocatable cells exceed MAX_PAGE_CELLS"
        );
        let base = mem.cast::<Self>();
        let owner_ptr = owner.as_ptr();
        debug_assert_eq!(
            owner_ptr.addr() & PAGE_TAG_MASK,
            0,
            "owner 必须按指针宽度对齐"
        );
        debug_assert_eq!(tag & !PAGE_TAG_MASK, 0, "tag 必须是合法的 owner 标记");
        let owner_ptr = owner_ptr.map_addr(|addr| addr | tag);
        // SAFETY: 调用方保证 mem 是足够的、对齐的独占内存；Page 头写在起始处。
        unsafe {
            base.as_ptr().write(Self {
                links_: UnsafeCell::new(PageLinks {
                    prev_: None,
                    next_: None,
                }),
                state_: UnsafeCell::new(PageState {
                    base_: first_cell,
                    cells_: cells as CellIndex,
                    used_: 0,
                    returned_: 0,
                }),
                owner_: AtomicPtr::new(owner_ptr),
            });
        }
        base
    }

    /// 阻塞式获取页锁。
    pub(crate) fn lock_(&self) -> PageGuard<'_> {
        loop {
            if let Option::Some(guard) = self.try_lock_() {
                return guard;
            }
            spin_loop();
        }
    }

    /// 尝试获取页锁；锁已被占用时返回 [`None`]。
    pub(crate) fn try_lock_(&self) -> Option<PageGuard<'_>> {
        let mut current = self.owner_.load(Ordering::Relaxed);
        loop {
            if current.addr() & LOCK_BIT != 0 {
                return Option::None;
            }
            let tagged = current.map_addr(|addr| addr | LOCK_BIT);
            match self.owner_.compare_exchange_weak(
                current,
                tagged,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Result::Ok(_) => {
                    return Option::Some(PageGuard {
                        owner_: &self.owner_,
                    });
                }
                Result::Err(actual) => current = actual,
            }
        }
    }

    /// owner 指针（去掉页锁与大页标记）。
    pub(crate) fn owner_ptr_(&self) -> NonNull<BumpListCore> {
        let raw = self
            .owner_
            .load(Ordering::Acquire)
            .map_addr(|addr| addr & !PAGE_TAG_MASK);
        // SAFETY: owner 指针在 Page 初始化时写入，永不为空。
        unsafe { NonNull::new_unchecked(raw) }
    }

    /// 读取 Page 链表的 `prev_`。
    ///
    /// # Safety
    ///
    /// 调用方必须持有 `BumpListAlloc` 的链表锁，或能保证没有并发写。
    pub(crate) unsafe fn prev_(&self) -> Option<NonNull<Page>> {
        // SAFETY: 由调用方保证链表锁或独占访问。
        unsafe { (*self.links_.get()).prev_ }
    }

    /// 读取 Page 链表的 `next_`。
    ///
    /// # Safety
    ///
    /// 同 [`Page::prev_`]。
    pub(crate) unsafe fn next_(&self) -> Option<NonNull<Page>> {
        // SAFETY: 由调用方保证链表锁或独占访问。
        unsafe { (*self.links_.get()).next_ }
    }

    /// 改写 Page 链表的 `prev_`。
    ///
    /// # Safety
    ///
    /// 同 [`Page::prev_`]。
    pub(crate) unsafe fn set_prev_(&self, prev: Option<NonNull<Page>>) {
        // SAFETY: 由调用方保证链表锁或独占访问。
        unsafe { (*self.links_.get()).prev_ = prev };
    }

    /// 改写 Page 链表的 `next_`。
    ///
    /// # Safety
    ///
    /// 同 [`Page::prev_`]。
    pub(crate) unsafe fn set_next_(&self, next: Option<NonNull<Page>>) {
        // SAFETY: 由调用方保证链表锁或独占访问。
        unsafe { (*self.links_.get()).next_ = next };
    }

    /// 本页是否为大页（`owner_` 的 [`LARGE_BIT`]）。
    ///
    /// kind 在 `Page` 初始化后不再变化，因此不需要页锁。
    #[cfg(feature = "big_page")]
    #[inline]
    pub(crate) fn is_large_(&self) -> bool {
        self.owner_.load(Ordering::Acquire).addr() & LARGE_BIT != 0
    }

    /// 在独占初始化阶段把本页登记为“只放一个块”的页。
    ///
    /// 写入块头 `order_flags_ = header_cell`、`next_free_ = header_cell + 1`，并把
    /// `PageState.used_` 设为 1。大页的 `cells_` 由 `init_in_place_with_tag_` 计算为
    /// 1，因此现有 `release_locked_` 会把这次释放视为整页变空。
    ///
    /// # Safety
    ///
    /// 调用方必须独占本页且尚未挂链；`header_cell + 1` 必须落在页内。
    #[cfg(feature = "big_page")]
    pub(crate) unsafe fn init_single_block_(&self, page: NonNull<Page>, header_cell: CellIndex) {
        let header = Self::header_at_(page, header_cell);
        // SAFETY: 调用方保证 header_cell 落在页内且独占访问。
        unsafe {
            (*header).order_flags_ = header_cell;
            (*header).next_free_ = header_cell + 1;
            let state = &mut *self.state_.get();
            debug_assert_eq!(state.base_, header_cell);
            debug_assert_eq!(state.cells_, 1);
            state.used_ = 1;
        }
    }

    /// 取得某个页内绝对 cell 下标处的 BlockHeader 指针。
    ///
    /// # Safety
    ///
    /// `page` 必须是有效的 Page 首指针；`cell` 必须落在页内。
    pub(crate) fn header_at_(page: NonNull<Page>, cell: CellIndex) -> *mut BlockHeader {
        let base = page.cast::<u8>().as_ptr();
        // SAFETY: 由调用方保证 cell 落在页内；page 指针带整页 provenance。
        unsafe { base.add(cell as usize * PTR).cast::<BlockHeader>() }
    }

    /// 由用户块首址反推所属 Page 与块头 cell 下标。
    ///
    /// 若块头与用户块之间存在对齐 padding，紧邻用户块的 cell 是 locator（最高位
    /// [`LOCATOR_BIT`]，低 15 位是到块头的 cell 距离）；否则该 cell 就是活跃块头的
    /// `order_flags`（最高位为 0）。
    ///
    /// # Safety
    ///
    /// `ptr` 必须指向本页一个仍然活跃的块（块头尚未设置 `FREE_BIT`）。
    pub(crate) unsafe fn locate_block_(ptr: NonNull<u8>) -> (NonNull<Page>, CellIndex) {
        // SAFETY: 调用方保证 ptr 是有效用户块首址，其前一个 cell 可读。
        let cell_addr = unsafe { ptr.as_ptr().sub(PTR) };
        // SAFETY: 同上。
        let tag = unsafe { (cell_addr as *const CellIndex).read() };
        let header_addr = if tag & LOCATOR_BIT != 0 {
            let distance = (tag & ORDER_MASK) as usize;
            // SAFETY: locator 由本页分配时写入，distance 指向本块的块头。
            unsafe { cell_addr.sub(distance * PTR) }
        } else {
            cell_addr
        };
        // SAFETY: header_addr 指向本块的块头。
        let flags = unsafe { (header_addr as *const CellIndex).read() };
        debug_assert_eq!(
            flags & FREE_BIT,
            0,
            "locate_block_ must only be used on a live block"
        );
        let order = flags & ORDER_MASK;
        // SAFETY: order 是块头自身的页内绝对 cell 下标。
        let page_addr = unsafe { header_addr.sub(order as usize * PTR) };
        // SAFETY: Page 首地址非空且落在同一页分配内。
        (
            unsafe { NonNull::new_unchecked(page_addr as *mut Page) },
            order,
        )
    }

    /// 获取页锁后尝试在页内分配。
    ///
    /// 返回用户块首址；空间不足或尺寸为零时返回 [`None`]。
    pub(crate) fn try_alloc_(&self, page: NonNull<Page>, layout: Layout) -> Option<NonNull<u8>> {
        let _guard = self.lock_();
        // SAFETY: 当前线程持有页锁。
        unsafe { self.try_alloc_locked_(page, layout) }
    }

    /// 在已持有页锁的前提下尝试分配。
    ///
    /// 块头放在当前 bump 位置；用户数据在块头之后按 `Layout::align()` 对齐，块头与
    /// 用户数据之间的 padding 属于本块。`next_free` 记录本块末尾（下一块头位置）。
    ///
    /// # Safety
    ///
    /// 调用方必须持有本 Page 的页锁，且 `page` 是有效的 Page 首指针。
    pub(crate) unsafe fn try_alloc_locked_(
        &self,
        page: NonNull<Page>,
        layout: Layout,
    ) -> Option<NonNull<u8>> {
        if layout.size() == 0 {
            return None;
        }
        // SAFETY: 调用方持有页锁，state_ 的访问是独占的。
        let state = unsafe { &mut *self.state_.get() };

        let base_cell = state.base_ as usize;
        let cells = state.cells_ as usize;
        let used = state.used_ as usize;
        if used >= cells {
            return None;
        }
        let header_cell = base_cell.checked_add(used)?;
        if header_cell >= MAX_PAGE_CELLS {
            return None;
        }

        let base_ptr = page.cast::<u8>().as_ptr();
        let base_addr = base_ptr as usize;
        let header_start = base_addr.checked_add(header_cell.checked_add(1)?.checked_mul(PTR)?)?;
        let align = layout.align().max(PTR);
        let user_addr = align_up_(header_start, align)?;
        let user_off = user_addr.checked_sub(base_addr)?;
        if !user_off.is_multiple_of(PTR) {
            return None;
        }
        let user_cell = user_off / PTR;
        if user_cell < header_cell + 1 {
            return None;
        }

        let data_cells = layout.size().div_ceil(PTR);
        let end_cell = user_cell.checked_add(data_cells)?;
        let page_end = base_cell.checked_add(cells)?;
        if end_cell > page_end || end_cell > MAX_PAGE_CELLS {
            return None;
        }

        let header_off = header_cell.checked_mul(PTR)?;
        // SAFETY: header_off 已校验落在页内；从页首指针偏移保留页内 provenance，且
        // 本线程持有页锁。
        let header = unsafe { base_ptr.add(header_off).cast::<BlockHeader>() };
        // SAFETY: header 落在页内，且本线程持有页锁。
        unsafe {
            (*header).order_flags_ = header_cell as CellIndex;
            (*header).next_free_ = end_cell as CellIndex;
        }

        if user_cell > header_cell + 1 {
            // 块头与用户块之间有 padding：在紧邻用户块的 cell 写入 locator，
            // 供 deallocate 从用户指针反推块头。
            let distance = (user_cell - 1 - header_cell) as CellIndex;
            debug_assert!(distance < ORDER_MASK);
            let locator_cell = user_cell.checked_sub(1)?;
            let locator_off = locator_cell.checked_mul(PTR)?;
            // SAFETY: locator_off 落在本块 padding 内，且持有页锁。
            let locator = unsafe { base_ptr.add(locator_off).cast::<CellIndex>() };
            // SAFETY: 同上。
            unsafe { locator.write(LOCATOR_BIT | distance) };
        }

        state.used_ = (end_cell - base_cell) as CellIndex;

        // SAFETY: user_off 已校验落在页内，user_addr 非零；从页首偏移保留 provenance。
        Some(unsafe { NonNull::new_unchecked(base_ptr.add(user_off)) })
    }

    /// 归还一个块；返回该页是否刚刚变为空页。
    ///
    /// 它是关联函数而不是 `&self` 方法：页刚变空时会经 owner 回调把「空页回收」升级给
    /// [`BumpListAlloc`]，清理可能释放 Page 本身。所有 `&Page` 借用都只活在取页锁 /
    /// 读取 owner 的短作用域内，调用清理前已经结束，避免在已释放页上保留引用。
    pub(crate) fn release_page_(page: NonNull<Page>, header_cell: CellIndex) -> bool {
        let became_empty = {
            // SAFETY: `page` 是有效的 Page 首指针；页锁在释放块期间防止并发分配。
            let page_ref = unsafe { page.as_ref() };
            let guard = page_ref.lock_();
            // SAFETY: 持有页锁，page 有效。
            let empty = unsafe { page_ref.release_locked_(page, header_cell) };
            drop(guard);
            empty
        };
        if became_empty {
            let owner = {
                // SAFETY: 页刚被本线程置空，尚未升级回收；Page 仍然有效。
                let page_ref = unsafe { page.as_ref() };
                page_ref.owner_ptr_()
            };
            // SAFETY: owner 仍存活（Page 由它创建且尚未被回收）；这里已没有 `&Page`
            // 借用，清理路径可以重新取页锁、链表锁，必要时释放 Page。
            unsafe { owner.as_ref().cleanup_empty_page_(page) };
        }
        became_empty
    }

    /// 在已持有页锁的前提下归还一个块；返回该页是否刚刚变为空页。
    ///
    /// `header_cell` 是 BlockHeader 自身的页内绝对 cell 下标。函数会把块头的
    /// `FREE_BIT` 置真，按 `next_free - header_cell` 把本块占用的 cell 数累加到
    /// `returned_`；当 `returned_ == used_` 时把 `used_` / `returned_` 归零，使这个
    /// Page 可以直接复用（不设退休状态）。重复释放或块头不一致时返回 `false`。
    ///
    /// # Safety
    ///
    /// 调用方必须持有本 Page 的页锁，且 `page` 是有效的 Page 首指针。
    pub(crate) unsafe fn release_locked_(
        &self,
        page: NonNull<Page>,
        header_cell: CellIndex,
    ) -> bool {
        if header_cell >= MAX_PAGE_CELLS as CellIndex {
            return false;
        }
        // SAFETY: 调用方持有页锁。
        let state = unsafe { &mut *self.state_.get() };
        let base = state.base_ as usize;
        let end = base.saturating_add(state.used_ as usize);
        let header = Self::header_at_(page, header_cell);
        // SAFETY: header_cell 已校验小于 MAX_PAGE_CELLS，且调用方持有页锁。
        let flags = unsafe { (*header).order_flags_ };
        if flags & ORDER_MASK != header_cell {
            return false;
        }
        if flags & FREE_BIT != 0 {
            return false;
        }
        // SAFETY: 同上。
        let next = unsafe { (*header).next_free_ } as usize;
        if (header_cell as usize) < base || next <= header_cell as usize || next > end {
            return false;
        }
        let span = next - header_cell as usize;
        let Some(returned) = state.returned_.checked_add(span as CellIndex) else {
            return false;
        };
        if returned > state.used_ {
            return false;
        }

        // SAFETY: header_cell 落在页内，且持有页锁。
        unsafe { (*header).order_flags_ = flags | FREE_BIT };
        state.returned_ = returned;
        if returned == state.used_ {
            // 整页已全部归还：回到初始 bump 状态，base_ / cells_ 保持不变。
            state.used_ = 0;
            state.returned_ = 0;
            true
        } else {
            false
        }
    }

    /// 当前页是否已经全部归还（`returned_ == used_`）。
    ///
    /// # Safety
    ///
    /// 调用方必须持有本 Page 的页锁。
    pub(crate) unsafe fn empty_locked_(&self) -> bool {
        // SAFETY: 由调用方保证页锁。
        let state = unsafe { &*self.state_.get() };
        state.returned_ == state.used_
    }

    /// 本页从 `base_` 起可用的 cell 数（`PageState.cells_`）。
    ///
    /// 第一页的 `base_` 还要容纳 `Page` 之外的 Core 与 `A`，因此它的容量可能小于
    /// 普通页；分配路径据此判断“空的第一页放不下”是否可能由普通页解决。
    ///
    /// # Safety
    ///
    /// 调用方必须持有本 Page 的页锁。
    pub(crate) unsafe fn capacity_locked_(&self) -> CellIndex {
        // SAFETY: 由调用方保证页锁。
        let state = unsafe { &*self.state_.get() };
        state.cells_
    }

    /// 扫描整页块链，返回 `(live, freed)`。
    ///
    /// 从 `base_` 开始，沿每个块头的 `next_free`（本块末尾 / 下一块头）走到
    /// `base_ + used_`；同时校验链范围、order 一致性，以及已归还 cell 数是否等于
    /// `state.returned_`。用于 debug 与 `BumpListAlloc` 析构时的未释放内存警告。
    ///
    /// # Errors
    ///
    /// 链损坏、order 不一致、`next_free` 越界或已归还计数不匹配时返回错误描述。
    pub(crate) fn scan_chain_(&self, page: NonNull<Page>) -> Result<(u16, u16), &'static str> {
        let _guard = self.lock_();
        // SAFETY: 持有页锁。
        let state = unsafe { &*self.state_.get() };
        let base = state.base_ as usize;
        let cells = state.cells_ as usize;
        let used = state.used_ as usize;

        let page_end = base.checked_add(cells).ok_or("page capacity overflow")?;
        if page_end > MAX_PAGE_CELLS {
            return Err("page capacity exceeds MAX_PAGE_CELLS");
        }
        if state.returned_ > state.used_ {
            return Err("returned exceeds used");
        }
        if used == 0 {
            if state.returned_ != 0 {
                return Err("non-zero returned on empty page");
            }
            return Result::Ok((0, 0));
        }

        let end = base.checked_add(used).ok_or("used overflow")?;
        let mut cell = base;
        let mut live = 0u16;
        let mut freed = 0u16;
        let mut freed_cells = 0usize;
        let mut steps = 0usize;

        while cell < end {
            if cell >= MAX_PAGE_CELLS {
                return Err("block cell out of range");
            }
            if steps > cells {
                return Err("block chain cycle");
            }
            // SAFETY: cell 已校验落在页内。
            let header = unsafe { &*Self::header_at_(page, cell as CellIndex) };
            if header.order_flags_ & ORDER_MASK != cell as CellIndex {
                return Err("block order mismatch");
            }
            let next = header.next_free_ as usize;
            if next <= cell || next > end {
                return Err("block next out of range");
            }
            if header.order_flags_ & FREE_BIT != 0 {
                freed = freed.checked_add(1).ok_or("freed count overflow")?;
                freed_cells = freed_cells
                    .checked_add(next - cell)
                    .ok_or("freed cell overflow")?;
            } else {
                live = live.checked_add(1).ok_or("live count overflow")?;
            }
            cell = next;
            steps += 1;
        }

        if cell != end {
            return Err("block chain end mismatch");
        }
        if freed_cells != state.returned_ as usize {
            return Err("returned count mismatch");
        }
        Result::Ok((live, freed))
    }
}

const _: () = assert!(Page::CELLS <= MAX_PAGE_CELLS);

fn align_up_(addr: usize, align: usize) -> Option<usize> {
    if !align.is_power_of_two() {
        return None;
    }
    let mask = align - 1;
    addr.checked_add(mask).map(|value| value & !mask)
}

#[cfg(test)]
mod tests_ {
    use super::*;
    use std::{
        alloc::{alloc, dealloc},
        boxed::Box,
        vec::Vec,
    };

    struct TestPage {
        page: NonNull<Page>,
        layout: Layout,
        first_cell: CellIndex,
        capacity: u16,
        // owner 的堆地址在 TestPage 生命周期内保持稳定；Drop 时重建 Box 释放。
        bump_ptr_: NonNull<crate::BumpListAlloc>,
    }

    impl TestPage {
        fn new_(bytes: usize) -> Self {
            Self::new_with_base_(bytes, Page::CELLS as CellIndex)
        }

        fn new_with_base_(bytes: usize, first_cell: CellIndex) -> Self {
            let layout =
                Layout::from_size_align(bytes, mem::align_of::<Page>()).expect("测试页布局合法");
            // SAFETY: 测试中直接申请一块独立内存。
            let ptr = unsafe { alloc(layout) };
            let page_ptr = NonNull::new(ptr).expect("测试页分配成功");
            let mem = NonNull::slice_from_raw_parts(page_ptr, bytes);
            // 用 Box::into_raw 固定句柄地址；`BumpListAlloc` 本身已经可以移动，这里保留
            // 这种写法只是为了不把 allocator 放进 TestPage 后参与字段析构顺序。
            let bump_ptr_ = NonNull::new(Box::into_raw(Box::new(
                crate::BumpListAlloc::new(crate::PageSize::P4K).expect("4 KiB 页应当受支持"),
            )))
            .expect("Box::into_raw 不返回空指针");
            // SAFETY: Core 位于 BumpListAlloc 的第一页内存中，地址在 allocator 析构前
            // 保持稳定；TestPage 的 Drop 会最后才析构这个 Box。
            let core_ptr_ = unsafe { NonNull::from(bump_ptr_.as_ref().core_()) };
            // SAFETY: 内存足够、按 Page 对齐且独占；owner 在 TestPage 生命周期内稳定。
            let page = unsafe { Page::init_in_place_(mem, core_ptr_, first_cell) };
            Self {
                page,
                layout,
                first_cell,
                capacity: (bytes / PTR) as u16 - first_cell,
                bump_ptr_,
            }
        }

        fn page_ref_(&self) -> &Page {
            // SAFETY: 测试页在整个 TestPage 生命周期内有效。
            unsafe { self.page.as_ref() }
        }
    }

    impl Drop for TestPage {
        fn drop(&mut self) {
            // SAFETY: 布局与申请时一致；测试页不再被访问。
            unsafe { dealloc(self.page.as_ptr() as *mut u8, self.layout) };
            // SAFETY: `bump_ptr_` 来自 Box::into_raw，且只在这里重建一次。
            unsafe { drop(Box::from_raw(self.bump_ptr_.as_ptr())) };
        }
    }

    fn header_cell_(ptr: NonNull<u8>) -> CellIndex {
        // SAFETY: 指针来自本测试页的活跃块。
        unsafe { Page::locate_block_(ptr).1 }
    }

    fn page_of_block_(ptr: NonNull<u8>) -> NonNull<Page> {
        // SAFETY: 指针来自本测试页的活跃块。
        unsafe { Page::locate_block_(ptr).0 }
    }

    /// 测试目标：页内块链能正确记录三个块的存活/释放状态，并能由块头反查页首。
    /// Test goal: The in-page block chain tracks three blocks and each block header can
    /// recover the page base.
    /// - 手段 / Method: 在 4 KiB 测试页上分配 8/24/16 字节块，依次释放。
    /// - 判定 / Assertion: `scan_chain_` 的 `(live, freed)` 依次为 (3,0)、(2,1)、
    ///   (1,2)、(0,0)；最后一个块释放后 Page 重置，可立即复用且每个块反查出的页首
    ///   都等于测试页。
    #[test]
    fn block_chain_tracks_live_and_freed_blocks() {
        let page = TestPage::new_(4096);
        let page_ref = page.page_ref_();

        let layout_a = Layout::from_size_align(8, 8).expect("布局合法");
        let layout_b = Layout::from_size_align(24, 8).expect("布局合法");
        let layout_c = Layout::from_size_align(16, 8).expect("布局合法");

        let a = page_ref.try_alloc_(page.page, layout_a).expect("a");
        let b = page_ref.try_alloc_(page.page, layout_b).expect("b");
        let c = page_ref.try_alloc_(page.page, layout_c).expect("c");
        let (cell_a, cell_b, cell_c) = (header_cell_(a), header_cell_(b), header_cell_(c));

        assert_eq!(page_of_block_(a), page.page);
        assert_eq!(page_of_block_(b), page.page);
        assert_eq!(page_of_block_(c), page.page);
        assert_eq!(page_ref.scan_chain_(page.page).expect("链有效"), (3, 0));

        assert!(!Page::release_page_(page.page, cell_a));
        assert_eq!(page_ref.scan_chain_(page.page).expect("链有效"), (2, 1));

        assert!(!Page::release_page_(page.page, cell_b));
        assert_eq!(page_ref.scan_chain_(page.page).expect("链有效"), (1, 2));

        assert!(Page::release_page_(page.page, cell_c));
        // 最后一个块释放后 Page 回到初始 bump 状态，旧块链被清空。
        assert_eq!(page_ref.scan_chain_(page.page).expect("链有效"), (0, 0));

        // 空页可以立即复用，不需要先归还。
        let d = page_ref
            .try_alloc_(page.page, layout_a)
            .expect("空页应当可复用");
        assert_eq!(page_ref.scan_chain_(page.page).expect("链有效"), (1, 0));
        assert_eq!(page_of_block_(d), page.page);
    }

    /// 测试目标：页内分配同时满足用户块的最小指针对齐和 `Layout::align()`，包括高对齐。
    /// Test goal: In-page allocation satisfies pointer alignment and `Layout::align()`,
    /// including higher alignment.
    /// - 手段 / Method: 在 8 KiB 页上依次分配 align 为 1/2/4/8/16/64/4096 的 1 字节块。
    /// - 判定 / Assertion: 每个用户地址都满足对应 align 且至少是指针宽度倍数；分配后
    ///   块链统计为 7 个存活、0 个已释放；全部释放后空页重置为 (0,0)。
    #[test]
    fn allocation_alignment_is_respected() {
        let page = TestPage::new_(8192);
        let page_ref = page.page_ref_();
        let aligns = [1usize, 2, 4, 8, 16, 64, 4096];
        let mut cells = Vec::new();

        for align in aligns {
            let layout = Layout::from_size_align(1, align).expect("布局合法");
            let ptr = page_ref.try_alloc_(page.page, layout).expect("应当放得下");
            assert_eq!(ptr.as_ptr() as usize % align, 0);
            assert_eq!(ptr.as_ptr() as usize % PTR, 0);
            cells.push(header_cell_(ptr));
        }

        assert_eq!(
            page_ref.scan_chain_(page.page).expect("链有效"),
            (aligns.len() as u16, 0)
        );

        for (index, cell) in cells.iter().enumerate() {
            let empty = Page::release_page_(page.page, *cell);
            assert_eq!(empty, index + 1 == cells.len());
        }
        assert_eq!(page_ref.scan_chain_(page.page).expect("链有效"), (0, 0));
    }

    /// 测试目标：释放中间块后再分配，块链仍能覆盖新旧块并统计正确的存活/释放数量。
    /// Test goal: Allocating after freeing a middle block keeps the chain coherent and
    /// reports the right live/freed counts.
    /// - 手段 / Method: 分配 a/b/c，释放 b，再分配 d。
    /// - 判定 / Assertion: 扫描结果为 3 存活、1 已释放；全部释放后空页重置为 (0,0)。
    #[test]
    fn chain_survives_free_then_allocate() {
        let page = TestPage::new_(4096);
        let page_ref = page.page_ref_();
        let layout = Layout::from_size_align(8, 8).expect("布局合法");

        let a = page_ref.try_alloc_(page.page, layout).expect("a");
        let b = page_ref.try_alloc_(page.page, layout).expect("b");
        let c = page_ref.try_alloc_(page.page, layout).expect("c");
        let cell_a = header_cell_(a);
        let cell_b = header_cell_(b);
        let cell_c = header_cell_(c);

        assert!(!Page::release_page_(page.page, cell_b));
        assert_eq!(page_ref.scan_chain_(page.page).expect("链有效"), (2, 1));

        let d = page_ref.try_alloc_(page.page, layout).expect("d");
        assert_eq!(page_of_block_(d), page.page);
        assert_eq!(page_ref.scan_chain_(page.page).expect("链有效"), (3, 1));

        assert!(!Page::release_page_(page.page, cell_a));
        assert!(!Page::release_page_(page.page, cell_c));
        assert!(Page::release_page_(page.page, header_cell_(d)));
        assert_eq!(page_ref.scan_chain_(page.page).expect("链有效"), (0, 0));
    }

    /// 测试目标：1 KiB 页填满时不会越界，order 与 used 都保持在页内，链扫描仍有效。
    /// Test goal: Filling a 1 KiB page does not overflow; order/used stay in range and
    /// chain scanning remains valid.
    /// - 手段 / Method: 反复分配 8 字节块直到 `try_alloc_` 返回 `None`。
    /// - 判定 / Assertion: 分配数大于 0 且不超过页 cell 数；扫描为 (N,0)，全部释放后
    ///   空页重置为 (0,0)。
    #[test]
    fn full_page_boundary_is_safe() {
        let page = TestPage::new_(1024);
        let page_ref = page.page_ref_();
        let layout = Layout::from_size_align(8, 8).expect("布局合法");
        let mut cells = Vec::new();

        while let Some(ptr) = page_ref.try_alloc_(page.page, layout) {
            cells.push(header_cell_(ptr));
            assert!(cells.len() <= page.capacity as usize);
        }

        let live = cells.len() as u16;
        assert!(live > 0);
        assert_eq!(page_ref.scan_chain_(page.page).expect("链有效"), (live, 0));

        for (index, cell) in cells.iter().enumerate() {
            let empty = Page::release_page_(page.page, *cell);
            assert_eq!(empty, index + 1 == cells.len());
        }
        assert_eq!(page_ref.scan_chain_(page.page).expect("链有效"), (0, 0));
    }

    /// 测试目标：零尺寸请求不会在页内分配，也不会改变块链。
    /// Test goal: A zero-sized request allocates nothing in the page and leaves the
    /// block chain unchanged.
    /// - 手段 / Method: 对 1 KiB 页调用 `try_alloc_(Layout::new::<()>(), ...)`。
    /// - 判定 / Assertion: 返回 `None`；页状态没有新增块。
    #[test]
    fn zero_sized_request_does_not_allocate() {
        let page = TestPage::new_(1024);
        let page_ref = page.page_ref_();
        assert!(
            page_ref
                .try_alloc_(page.page, Layout::new::<()>())
                .is_none()
        );
    }

    /// 测试目标：`PageState.base_` / `cells_` 可以按页不同，容量随首个可分配 cell
    /// 后移而缩小，且整页仍然按 `returned_ == used_` 判定为空。
    /// Test goal: `PageState.base_` / `cells_` are per-page; moving the first allocatable
    /// cell later reduces capacity, and emptiness is still derived from `returned_ ==
    /// used_`.
    ///
    /// - 手段：用比 `Page::CELLS` 大 10 个 cell 的 `first_cell` 构造 1 KiB 测试页，
    ///   分配 8 字节块直到放不下，记录首块块头与扫描结果，再全部释放。
    /// - 判定：首块块头正好在 `first_cell`；容量等于总 cell 数减 `first_cell`；释放
    ///   完毕后 `used_` / `returned_` 归零、`empty_locked_` 为真、链扫描为 (0,0)。
    ///
    /// - Means: build a 1 KiB test page with `first_cell = Page::CELLS + 10`, allocate
    ///   8-byte blocks until full, check the first header and the scan, then release all.
    /// - Verdict: the first header is exactly at `first_cell`; capacity is total cells
    ///   minus `first_cell`; after releasing everything `used_` / `returned_` are zero,
    ///   `empty_locked_` is true and the chain scans as (0, 0).
    #[test]
    fn variable_first_cell_reduces_capacity_() {
        let first_cell = (Page::CELLS + 10) as CellIndex;
        let page = TestPage::new_with_base_(1024, first_cell);
        let page_ref = page.page_ref_();
        assert_eq!(page.first_cell, first_cell);
        assert_eq!(
            page.capacity,
            (page.layout.size() / PTR) as u16 - first_cell
        );

        let layout = Layout::from_size_align(8, 8).expect("布局合法");
        let first = page_ref
            .try_alloc_(page.page, layout)
            .expect("首个块应当放得下");
        assert_eq!(header_cell_(first), first_cell);
        assert_eq!(page_of_block_(first), page.page);

        let mut ptrs = Vec::new();
        ptrs.push(first);
        while let Some(ptr) = page_ref.try_alloc_(page.page, layout) {
            ptrs.push(ptr);
            assert!(ptrs.len() <= page.capacity as usize);
        }
        assert_eq!(
            page_ref.scan_chain_(page.page).expect("链有效"),
            (ptrs.len() as u16, 0)
        );

        let state = unsafe { &*page_ref.state_.get() };
        assert_eq!((state.base_, state.cells_), (first_cell, page.capacity));
        assert_eq!(state.returned_, 0);
        assert!(!unsafe { page_ref.empty_locked_() });

        for (index, ptr) in ptrs.iter().enumerate() {
            let cell = header_cell_(*ptr);
            let empty = Page::release_page_(page.page, cell);
            assert_eq!(empty, index + 1 == ptrs.len());
        }
        let state = unsafe { &*page_ref.state_.get() };
        assert_eq!((state.used_, state.returned_), (0, 0));
        assert!(unsafe { page_ref.empty_locked_() });
        assert_eq!(page_ref.scan_chain_(page.page).expect("链有效"), (0, 0));
    }

    /// 测试目标：对齐 padding 存在时，用户块前的 locator 能 O(1) 找回块头。
    /// Test goal: when alignment padding exists, the locator immediately before the
    /// user block recovers the header in O(1).
    ///
    /// - 手段：在 4 KiB 页上分配一个 64 字节对齐的 1 字节块，读取紧邻用户块之前的
    ///   cell 与 `locate_block_` 的结果。
    /// - 判定：locator 最高位为 1，距离与用户 cell/块头 cell 一致；`header_cell_` /
    ///   `page_of_block_` 都能正确反推；释放后页变空。
    ///
    /// - Means: allocate a 64-byte-aligned 1-byte block on a 4 KiB page and inspect the
    ///   cell immediately before the user block plus `locate_block_`.
    /// - Verdict: the locator tag bit is set and its distance matches the user/header
    ///   cells; `header_cell_` / `page_of_block_` recover correctly; releasing empties the
    ///   page.
    #[test]
    fn locator_recovers_header_with_alignment_padding_() {
        // 找一个让首个块头不落在 64 字节边界的 base，强制产生对齐 padding。
        let mut first_cell = Page::CELLS as CellIndex;
        let page = loop {
            let page = TestPage::new_with_base_(4096, first_cell);
            let header_start = page.page.as_ptr() as usize + (first_cell as usize + 1) * PTR;
            if header_start % 64 != 0 {
                break page;
            }
            first_cell += 1;
            assert!(first_cell < 1024, "测试页容量不足以找到非对齐块头");
        };
        let page_ref = page.page_ref_();
        let layout = Layout::from_size_align(1, 64).expect("布局合法");
        let ptr = page_ref.try_alloc_(page.page, layout).expect("应当放得下");
        assert_eq!(ptr.as_ptr() as usize % 64, 0);

        let header_cell = header_cell_(ptr);
        assert_eq!(header_cell, page.first_cell);
        assert_eq!(page_of_block_(ptr), page.page);

        // SAFETY: 用户块前面的 cell 可读。
        let locator = unsafe { (ptr.as_ptr().sub(PTR) as *const CellIndex).read() };
        assert_ne!(locator & LOCATOR_BIT, 0);
        let user_cell = (ptr.as_ptr() as usize - page.page.as_ptr() as usize) / PTR;
        let distance = locator & ORDER_MASK;
        assert_eq!(header_cell as usize + distance as usize, user_cell - 1);

        assert!(Page::release_page_(page.page, header_cell));
    }

    /// 测试目标：重复释放同一个块会被忽略，不破坏 `used_` / `returned_` 计数。
    /// Test goal: releasing the same block twice is ignored and does not corrupt the
    /// `used_` / `returned_` counters.
    ///
    /// - 手段：分配一个 8 字节块，连续调用两次 `Page::release_page_`。
    /// - 判定：第一次返回 true（页变空），第二次返回 false；页状态回到 (0,0)，
    ///   链扫描为 (0,0)。
    ///
    /// - Means: allocate an 8-byte block and call `Page::release_page_` twice.
    /// - Verdict: the first call returns true, the second false; page state is (0,0) and
    ///   the chain scans as (0,0).
    #[test]
    fn double_release_is_ignored_() {
        let page = TestPage::new_(1024);
        let page_ref = page.page_ref_();
        let layout = Layout::from_size_align(8, 8).expect("布局合法");
        let ptr = page_ref.try_alloc_(page.page, layout).expect("应当放得下");
        let cell = header_cell_(ptr);

        assert!(Page::release_page_(page.page, cell));
        assert!(!Page::release_page_(page.page, cell));
        let state = unsafe { &*page_ref.state_.get() };
        assert_eq!((state.used_, state.returned_), (0, 0));
        assert!(unsafe { page_ref.empty_locked_() });
        assert_eq!(page_ref.scan_chain_(page.page).expect("链有效"), (0, 0));
    }

    /// 测试目标：`returned_` 随释放 cell 数增长，`returned_ == used_` 时才判定为空页。
    /// Test goal: `returned_` grows with released cells and the page is empty only when
    /// `returned_ == used_`.
    ///
    /// - 手段：分配两个 8 字节块，先释放第一个，再释放第二个。
    /// - 判定：释放第一个后 `returned_ > 0` 且小于 `used_`，`empty_locked_` 为假；
    ///   释放第二个后两者归零，`empty_locked_` 为真。
    ///
    /// - Means: allocate two 8-byte blocks and release them one by one.
    /// - Verdict: after the first release `returned_ > 0` but `< used_` and the page is
    ///   not empty; after the second both are zero and the page is empty.
    #[test]
    fn returned_count_derives_empty_() {
        let page = TestPage::new_(1024);
        let page_ref = page.page_ref_();
        let layout = Layout::from_size_align(8, 8).expect("布局合法");
        let a = page_ref.try_alloc_(page.page, layout).expect("a");
        let b = page_ref.try_alloc_(page.page, layout).expect("b");
        let (cell_a, cell_b) = (header_cell_(a), header_cell_(b));

        let state = unsafe { &*page_ref.state_.get() };
        assert!(state.used_ > 0);
        assert_eq!(state.returned_, 0);
        assert!(!unsafe { page_ref.empty_locked_() });

        assert!(!Page::release_page_(page.page, cell_a));
        let state = unsafe { &*page_ref.state_.get() };
        assert!(state.returned_ > 0);
        assert!(state.returned_ < state.used_);
        assert!(!unsafe { page_ref.empty_locked_() });

        assert!(Page::release_page_(page.page, cell_b));
        let state = unsafe { &*page_ref.state_.get() };
        assert_eq!((state.used_, state.returned_), (0, 0));
        assert!(unsafe { page_ref.empty_locked_() });
    }

    /// 测试目标：BlockHeader 与 Page 头的静态尺寸约定成立。
    /// Test goal: The static size assumptions for BlockHeader and Page hold.
    /// - 手段 / Method: 检查 `size_of::<BlockHeader>()` 与 `Page::CELLS`。
    /// - 判定 / Assertion: BlockHeader 大小等于指针宽度；Page 头 cell 数在 1 到
    ///   `MAX_PAGE_CELLS` 之间，且与向上取整公式一致。
    #[test]
    fn layout_constants_are_consistent() {
        assert_eq!(mem::size_of::<BlockHeader>(), PTR);
        assert_eq!(Page::CELLS, mem::size_of::<Page>().div_ceil(PTR));
        assert_eq!(mem::size_of::<PageState>(), 8);
        assert_eq!(ORDER_MASK, 0x7FFF);
        assert_eq!(FREE_BIT, 0x8000);
    }
}
