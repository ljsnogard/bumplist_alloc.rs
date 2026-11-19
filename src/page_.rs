//! Page 头、BlockHeader、页锁与页内块链。
//!
//! # 布局
//!
//! ```text
//! Page base
//! ├── Page { links, lock, state }
//! ├── ... padding 到 cell 边界 ...
//! ├── BlockHeader cell (order_flags, next_free, reserved)
//! ├── user block
//! ├── ...
//! ```
//!
//! Page 内以指针宽度 `P = size_of::<usize>()` 为一个 cell。`BlockHeader` 占一个
//! cell；用户块从下一个 cell 开始。`BlockHeader.order_flags` 的低 15 位是自身 cell
//! 下标，最高位表示该块已经释放；`BlockHeader.next_free` 是页内块链的后继 cell
//! 下标（`u16::MAX` 表示没有后继）。
//!
//! 块链在分配时通过回填前一块头的 `next_free` 建立。释放块只设置 `FREE_BIT`，不改变
//! 链，因此可以从 `PageState.first_` 走完整页的块链，统计存活/已释放块，供 debug 与
//! `BumpListAlloc` 析构时检查未释放内存。

use core::{
    alloc::Layout,
    cell::UnsafeCell,
    hint::spin_loop,
    mem,
    ptr::NonNull,
    sync::atomic::{AtomicPtr, Ordering},
};

use crate::{alloc_::BumpListCore, page_size_::MAX_PAGE_CELLS};

/// 指针宽度，也是 Page 内一个 cell 的字节数。
pub(crate) const PTR: usize = mem::size_of::<usize>();

/// 没有后继/没有首块时使用的哨兵。
pub(crate) const SENTINEL: u16 = u16::MAX;

/// `order_flags` 的最高位：该块已释放。
pub(crate) const FREE_BIT: u16 = 1 << 15;

/// `order_flags` 中保存 cell 下标的低位掩码。
pub(crate) const ORDER_MASK: u16 = FREE_BIT - 1;

/// 页内块头。
///
/// 字段布局在 64 位平台占 8 字节、32 位平台占 4 字节；`_reserved` 在 64 位平台保留
/// 4 字节，在 32 位平台长度为 0。块头位于 cell 边界，但只使用开头的两个 `u16`。
#[repr(C)]
pub(crate) struct BlockHeader {
    /// 低 15 位：自身 cell 下标；最高位：已释放标记。
    pub(crate) order_flags_: u16,
    /// 页内块链的后继 cell 下标；`SENTINEL` 表示没有后继。
    pub(crate) next_free_: u16,
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
#[repr(C)]
pub(crate) struct PageState {
    /// 下一个可 bump 的 cell 下标；空页复用时会重置。
    pub(crate) used_: u16,
    /// 本页仍存活的块数。
    pub(crate) live_: u16,
    /// 本页第一个块头的 cell 下标；没有块时为 `SENTINEL`。
    pub(crate) first_: u16,
    /// 本页最后一个块头的 cell 下标；没有块时为 `SENTINEL`。
    pub(crate) last_: u16,
}

/// Page 头。
///
/// 字段按大小降序手动布置：`links_`（2 个指针）、`state_`（8 字节）、`owner_`
/// （指向非泛型 [`BumpListCore`] 的 owner 指针，低 1 位兼作页锁）。64 位下 32 字节，
/// 32 位下 20 字节，分别占 4/5 个 cell。
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
    ) -> NonNull<Self> {
        let base = mem.cast::<Self>();
        let owner_ptr = owner.as_ptr();
        debug_assert_eq!(
            owner_ptr.addr() & LOCK_BIT,
            0,
            "owner 必须按至少 2 字节对齐"
        );
        // SAFETY: 调用方保证 mem 是足够的、对齐的独占内存；Page 头写在起始处。
        unsafe {
            base.as_ptr().write(Self {
                links_: UnsafeCell::new(PageLinks {
                    prev_: None,
                    next_: None,
                }),
                state_: UnsafeCell::new(PageState {
                    used_: Self::CELLS as u16,
                    live_: 0,
                    first_: SENTINEL,
                    last_: SENTINEL,
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

    /// owner 指针（去掉页锁标记）。
    pub(crate) fn owner_ptr_(&self) -> NonNull<BumpListCore> {
        let raw = self
            .owner_
            .load(Ordering::Acquire)
            .map_addr(|addr| addr & !LOCK_BIT);
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

    /// 取得某个 cell 下标处的 BlockHeader 指针。
    ///
    /// # Safety
    ///
    /// `page` 必须是有效的 Page 首指针；`cell` 必须小于 Page 的 cell 总数，且不等于
    /// `SENTINEL`。
    fn header_at_(page: NonNull<Page>, cell: u16) -> *mut BlockHeader {
        let base = page.cast::<u8>().as_ptr();
        // SAFETY: 由调用方保证 cell 落在页内；page 指针带整页 provenance。
        unsafe { base.add(cell as usize * PTR).cast::<BlockHeader>() }
    }

    /// 获取页锁后尝试在页内分配。
    ///
    /// 返回用户块首址；空间不足或尺寸为零时返回 [`None`]。
    pub(crate) fn try_alloc_(
        &self,
        page: NonNull<Page>,
        layout: Layout,
        page_cells: u16,
    ) -> Option<NonNull<u8>> {
        let _guard = self.lock_();
        // SAFETY: 当前线程持有页锁。
        unsafe { self.try_alloc_locked_(page, layout, page_cells) }
    }

    /// 在已持有页锁的前提下尝试分配。
    ///
    /// # Safety
    ///
    /// 调用方必须持有本 Page 的页锁，且 `page` 是有效的 Page 首指针。
    pub(crate) unsafe fn try_alloc_locked_(
        &self,
        page: NonNull<Page>,
        layout: Layout,
        page_cells: u16,
    ) -> Option<NonNull<u8>> {
        if layout.size() == 0 {
            return None;
        }
        // SAFETY: 调用方持有页锁，state_ 的访问是独占的。
        let state = unsafe { &mut *self.state_.get() };

        let base_ptr = page.cast::<u8>().as_ptr();
        let base_addr = base_ptr as usize;
        let used = state.used_ as usize;
        let header_start = base_addr.checked_add(used.checked_add(1)?.checked_mul(PTR)?)?;
        let align = layout.align().max(PTR);
        let user_addr = align_up_(header_start, align)?;
        let user_off = user_addr.checked_sub(base_addr)?;
        let header_off = user_off.checked_sub(PTR)?;
        if !header_off.is_multiple_of(PTR) {
            return None;
        }
        let header_cell = header_off / PTR;
        if header_cell < used || header_cell >= MAX_PAGE_CELLS {
            return None;
        }

        let data_cells = layout.size().div_ceil(PTR);
        let end_cell = header_cell.checked_add(1)?.checked_add(data_cells)?;
        if end_cell > page_cells as usize {
            return None;
        }

        // SAFETY: header_off 已校验落在 page_cells 范围内；从页首指针偏移保留页内
        // provenance，且本线程持有页锁。
        let header = unsafe { base_ptr.add(header_off).cast::<BlockHeader>() };
        // SAFETY: header 落在页内，且本线程持有页锁。
        unsafe {
            (*header).order_flags_ = header_cell as u16;
            (*header).next_free_ = SENTINEL;
        }

        // 回填前一个块头的后继；由此形成整页的块链。
        if state.first_ == SENTINEL {
            state.first_ = header_cell as u16;
        } else {
            if state.last_ == SENTINEL {
                return None;
            }
            let last_header = Self::header_at_(page, state.last_);
            // SAFETY: last_header 属于本页，且持有页锁。
            unsafe { (*last_header).next_free_ = header_cell as u16 };
        }
        state.last_ = header_cell as u16;
        state.used_ = end_cell as u16;
        state.live_ = state.live_.checked_add(1)?;

        // SAFETY: user_off 已校验落在页内，user_addr 非零；从页首偏移保留 provenance。
        Some(unsafe { NonNull::new_unchecked(base_ptr.add(user_off)) })
    }

    /// 归还一个块；返回该页是否刚刚变为空页。
    ///
    /// 它是关联函数而不是 `&self` 方法：页刚变空时会经 owner 回调把「空页回收」升级给
    /// [`BumpListAlloc`]，清理可能释放 Page 本身。所有 `&Page` 借用都只活在取页锁 /
    /// 读取 owner 的短作用域内，调用清理前已经结束，避免在已释放页上保留引用。
    pub(crate) fn release_page_(page: NonNull<Page>, header_cell: u16) -> bool {
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
    /// `header_cell` 是 BlockHeader 自身在页内的 cell 下标。函数会把块头的
    /// `FREE_BIT` 置真；页变空时把 `used_` / `first_` / `last_` 重置，使这个 Page
    /// 可以直接复用（不设退休状态）。
    ///
    /// # Safety
    ///
    /// 调用方必须持有本 Page 的页锁，且 `page` 是有效的 Page 首指针。
    pub(crate) unsafe fn release_locked_(&self, page: NonNull<Page>, header_cell: u16) -> bool {
        // SAFETY: 调用方持有页锁。
        let state = unsafe { &mut *self.state_.get() };
        let Some(live) = state.live_.checked_sub(1) else {
            return false;
        };
        if header_cell >= MAX_PAGE_CELLS as u16 {
            return false;
        }
        state.live_ = live;
        let header = Self::header_at_(page, header_cell);
        // SAFETY: header_cell 落在页内，且持有页锁。
        unsafe { (*header).order_flags_ |= FREE_BIT };
        if live == 0 {
            // 空页回到初始 bump 状态；若是唯一空页，BumpListAlloc 会把它留作 tail 复用。
            state.used_ = Self::CELLS as u16;
            state.first_ = SENTINEL;
            state.last_ = SENTINEL;
            true
        } else {
            false
        }
    }

    /// 读取当前活跃块数。
    ///
    /// # Safety
    ///
    /// 调用方必须持有本 Page 的页锁。
    pub(crate) unsafe fn live_locked_(&self) -> u16 {
        // SAFETY: 由调用方保证页锁。
        unsafe { (*self.state_.get()).live_ }
    }

    /// 扫描整页块链，返回 `(live, freed)`。
    ///
    /// 该函数持页锁，并验证链的 cell 范围、order 一致性、live 计数与首尾指针；用于
    /// debug 与 `BumpListAlloc` 析构时的未释放内存警告。
    ///
    /// # Errors
    ///
    /// 链损坏、order 不一致、live 计数不匹配或首尾指针不一致时返回错误描述。
    pub(crate) fn scan_chain_(
        &self,
        page: NonNull<Page>,
        page_cells: u16,
    ) -> Result<(u16, u16), &'static str> {
        let _guard = self.lock_();
        // SAFETY: 持有页锁。
        let state = unsafe { &*self.state_.get() };
        if page_cells > MAX_PAGE_CELLS as u16 {
            return Err("page_cells exceeds MAX_PAGE_CELLS");
        }

        let mut cell = state.first_;
        let mut live = 0u16;
        let mut freed = 0u16;
        let mut steps = 0usize;
        let mut last_seen = SENTINEL;

        while cell != SENTINEL {
            if cell >= page_cells {
                return Err("block cell out of range");
            }
            if steps > page_cells as usize {
                return Err("block chain cycle");
            }
            // SAFETY: cell 已校验落在页内。
            let header = unsafe { &*Self::header_at_(page, cell) };
            if header.order_flags_ & ORDER_MASK != cell {
                return Err("block order mismatch");
            }
            if header.order_flags_ & FREE_BIT != 0 {
                freed = freed.checked_add(1).ok_or("freed count overflow")?;
            } else {
                live = live.checked_add(1).ok_or("live count overflow")?;
            }
            last_seen = cell;
            cell = header.next_free_;
            steps += 1;
        }

        if live != state.live_ {
            return Err("live count mismatch");
        }
        if last_seen != state.last_ {
            return Err("last block mismatch");
        }
        Ok((live, freed))
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
        page_cells: u16,
        // owner 的堆地址在 TestPage 生命周期内保持稳定；Drop 时重建 Box 释放。
        bump_ptr_: NonNull<crate::BumpListAlloc>,
    }

    impl TestPage {
        fn new_(bytes: usize) -> Self {
            let layout =
                Layout::from_size_align(bytes, mem::align_of::<Page>()).expect("测试页布局合法");
            // SAFETY: 测试中直接申请一块独立内存。
            let ptr = unsafe { alloc(layout) };
            let page_ptr = NonNull::new(ptr).expect("测试页分配成功");
            let mem = NonNull::slice_from_raw_parts(page_ptr, bytes);
            // 用 Box::into_raw 固定 owner 地址；避免把 Box 移进结构体时对 pointee
            // 做 Unique retag 而让 Page 里的 owner 指针失效。
            let bump_ptr_ = NonNull::new(Box::into_raw(Box::new(
                crate::BumpListAlloc::new(crate::PageSize::P4K).expect("4 KiB 页应当受支持"),
            )))
            .expect("Box::into_raw 不返回空指针");
            // SAFETY: bump_ptr_ 指向 Box 里的稳定分配器；Core 自身在另一个 Box 中，
            // 地址不随外层 BumpListAlloc 移动。
            let core_ptr_ = unsafe { NonNull::from(&bump_ptr_.as_ref().core_) };
            // SAFETY: 内存足够、按 Page 对齐且独占；owner 在 TestPage 生命周期内稳定。
            let page = unsafe { Page::init_in_place_(mem, core_ptr_) };
            Self {
                page,
                layout,
                page_cells: (bytes / PTR) as u16,
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

    fn header_cell_(ptr: NonNull<u8>) -> u16 {
        // SAFETY: 指针来自本测试页的分配，BlockHeader 紧邻用户块之前。
        let header = unsafe { ptr.as_ptr().sub(PTR) };
        // SAFETY: BlockHeader 至少包含首个 u16。
        let flags = unsafe { (header as *const u16).read() };
        flags & ORDER_MASK
    }

    fn page_of_block_(ptr: NonNull<u8>) -> NonNull<Page> {
        // SAFETY: 同 header_cell_。
        let header = unsafe { ptr.as_ptr().sub(PTR) };
        // SAFETY: 同 header_cell_。
        let flags = unsafe { (header as *const u16).read() };
        let cell = (flags & ORDER_MASK) as usize;
        // SAFETY: 减去自身位置下标回到页首，仍在同一页分配内。
        let page = unsafe { header.sub(cell * PTR) };
        // SAFETY: 页首非空。
        unsafe { NonNull::new_unchecked(page as *mut Page) }
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
        let page_cells = page.page_cells;

        let layout_a = Layout::from_size_align(8, 8).expect("布局合法");
        let layout_b = Layout::from_size_align(24, 8).expect("布局合法");
        let layout_c = Layout::from_size_align(16, 8).expect("布局合法");

        let a = page_ref
            .try_alloc_(page.page, layout_a, page_cells)
            .expect("a");
        let b = page_ref
            .try_alloc_(page.page, layout_b, page_cells)
            .expect("b");
        let c = page_ref
            .try_alloc_(page.page, layout_c, page_cells)
            .expect("c");
        let (cell_a, cell_b, cell_c) = (header_cell_(a), header_cell_(b), header_cell_(c));

        assert_eq!(page_of_block_(a), page.page);
        assert_eq!(page_of_block_(b), page.page);
        assert_eq!(page_of_block_(c), page.page);
        assert_eq!(
            page_ref.scan_chain_(page.page, page_cells).expect("链有效"),
            (3, 0)
        );

        assert!(!Page::release_page_(page.page, cell_a));
        assert_eq!(
            page_ref.scan_chain_(page.page, page_cells).expect("链有效"),
            (2, 1)
        );

        assert!(!Page::release_page_(page.page, cell_b));
        assert_eq!(
            page_ref.scan_chain_(page.page, page_cells).expect("链有效"),
            (1, 2)
        );

        assert!(Page::release_page_(page.page, cell_c));
        // 最后一个块释放后 Page 回到初始 bump 状态，旧块链被清空。
        assert_eq!(
            page_ref.scan_chain_(page.page, page_cells).expect("链有效"),
            (0, 0)
        );

        // 空页可以立即复用，不需要先归还。
        let d = page_ref
            .try_alloc_(page.page, layout_a, page_cells)
            .expect("空页应当可复用");
        assert_eq!(
            page_ref.scan_chain_(page.page, page_cells).expect("链有效"),
            (1, 0)
        );
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
        let page_cells = page.page_cells;
        let aligns = [1usize, 2, 4, 8, 16, 64, 4096];
        let mut cells = Vec::new();

        for align in aligns {
            let layout = Layout::from_size_align(1, align).expect("布局合法");
            let ptr = page_ref
                .try_alloc_(page.page, layout, page_cells)
                .expect("应当放得下");
            assert_eq!(ptr.as_ptr() as usize % align, 0);
            assert_eq!(ptr.as_ptr() as usize % PTR, 0);
            cells.push(header_cell_(ptr));
        }

        assert_eq!(
            page_ref.scan_chain_(page.page, page_cells).expect("链有效"),
            (aligns.len() as u16, 0)
        );

        for (index, cell) in cells.iter().enumerate() {
            let empty = Page::release_page_(page.page, *cell);
            assert_eq!(empty, index + 1 == cells.len());
        }
        assert_eq!(
            page_ref.scan_chain_(page.page, page_cells).expect("链有效"),
            (0, 0)
        );
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
        let page_cells = page.page_cells;
        let layout = Layout::from_size_align(8, 8).expect("布局合法");

        let a = page_ref
            .try_alloc_(page.page, layout, page_cells)
            .expect("a");
        let b = page_ref
            .try_alloc_(page.page, layout, page_cells)
            .expect("b");
        let c = page_ref
            .try_alloc_(page.page, layout, page_cells)
            .expect("c");
        let cell_a = header_cell_(a);
        let cell_b = header_cell_(b);
        let cell_c = header_cell_(c);

        assert!(!Page::release_page_(page.page, cell_b));
        assert_eq!(
            page_ref.scan_chain_(page.page, page_cells).expect("链有效"),
            (2, 1)
        );

        let d = page_ref
            .try_alloc_(page.page, layout, page_cells)
            .expect("d");
        assert_eq!(page_of_block_(d), page.page);
        assert_eq!(
            page_ref.scan_chain_(page.page, page_cells).expect("链有效"),
            (3, 1)
        );

        assert!(!Page::release_page_(page.page, cell_a));
        assert!(!Page::release_page_(page.page, cell_c));
        assert!(Page::release_page_(page.page, header_cell_(d)));
        assert_eq!(
            page_ref.scan_chain_(page.page, page_cells).expect("链有效"),
            (0, 0)
        );
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
        let page_cells = page.page_cells;
        let layout = Layout::from_size_align(8, 8).expect("布局合法");
        let mut cells = Vec::new();

        while let Some(ptr) = page_ref.try_alloc_(page.page, layout, page_cells) {
            cells.push(header_cell_(ptr));
            assert!(cells.len() <= page_cells as usize);
        }

        let live = cells.len() as u16;
        assert!(live > 0);
        assert_eq!(
            page_ref.scan_chain_(page.page, page_cells).expect("链有效"),
            (live, 0)
        );

        for (index, cell) in cells.iter().enumerate() {
            let empty = Page::release_page_(page.page, *cell);
            assert_eq!(empty, index + 1 == cells.len());
        }
        assert_eq!(
            page_ref.scan_chain_(page.page, page_cells).expect("链有效"),
            (0, 0)
        );
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
                .try_alloc_(page.page, Layout::new::<()>(), page.page_cells)
                .is_none()
        );
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
        assert_eq!(SENTINEL, u16::MAX);
        assert_eq!(ORDER_MASK, 0x7FFF);
        assert_eq!(FREE_BIT, 0x8000);
    }
}
