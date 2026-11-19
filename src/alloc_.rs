//! `BumpListAlloc`：链表锁 + 每页页锁的固定页 bump 分配器。

use alloc::alloc::Global;
use core::{
    alloc::{AllocError, Allocator, AllocatorClone, Layout},
    cell::UnsafeCell,
    hint::spin_loop,
    marker::PhantomData,
    mem,
    num::NonZero,
    ptr::NonNull,
};

use crate::{
    error_::BumpListAllocError,
    init_page_::InitPage,
    page_::{CellIndex, PTR, Page},
    page_size_::PageSize,
    state_::{CoreLockGuard, CoreState},
};

/// Page 链表的头尾；由 Core 的 list lock 保护。
struct PageList {
    head_: Option<NonNull<Page>>,
    tail_: Option<NonNull<Page>>,
}

impl PageList {
    const fn with_root_(root: NonNull<Page>) -> Self {
        Self {
            head_: Some(root),
            tail_: Some(root),
        }
    }
}

/// 与 `A` 无关的分配器核心：页大小、共享状态、页链表与页 backing。
///
/// Core 位于第一页内存中的 [`InitPage<A>`] 里，Page 的 owner 指向它。它与底层分配器
/// `A` 的相对位置在构造后固定：backing 字段保存一个指向 `A` 的 `NonNull<dyn Allocator>`
/// （fat 指针，所占空间与原来的两个函数指针相同），不再需要“整分配器自指针”和
/// `offset_of!`，也不需要单态化函数指针。
///
/// [`CoreState`] 用一个 `AtomicUsize` 同时保存句柄引用计数与两把锁的标记：最高位是
/// list lock，次高位是 backing lock。`A: Send` 只保证 A 可以独占地在任意线程使用；
/// backing lock 串行化所有对 A 的 allocate / deallocate 调用，因此 A 不需要 `Sync`。
#[repr(C)]
pub(crate) struct BumpListCore {
    /// 第一页的 Page 首地址；它始终留在链表中，永远不会被归还。
    root_page_: NonNull<Page>,
    /// 指向第一页 `MaybeUninit<A>` 槽的 fat 指针；构造完成后有效，析构时先于 `A`
    /// 被移出而失效。
    backing_: NonNull<dyn Allocator>,
    inner_: UnsafeCell<PageList>,
    /// 低位引用计数；最高两位分别是 list / backing lock 标记。
    state_: CoreState,
    page_size_: PageSize,
}

impl BumpListCore {
    /// 用页大小、第一页指针与 backing fat 指针构造核心。
    ///
    /// 调用方必须已经写入 `A`，并且 `backing` 指向该 `A`；`root_page` 必须位于
    /// `backing` 所在的第一页起始处。容量校验在 [`InitPage::new_`] 中完成。
    pub(crate) fn new_(
        page_size: PageSize,
        root_page: NonNull<Page>,
        backing: NonNull<dyn Allocator>,
    ) -> Self {
        Self {
            root_page_: root_page,
            backing_: backing,
            inner_: UnsafeCell::new(PageList::with_root_(root_page)),
            state_: CoreState::new(),
            page_size_: page_size,
        }
    }

    /// 当前实例的页大小。
    pub(crate) fn page_size_(&self) -> PageSize {
        self.page_size_
    }

    fn lock_list_(&self) -> CoreLockGuard<'_> {
        self.state_.lock_list_()
    }

    fn lock_backing_(&self) -> CoreLockGuard<'_> {
        self.state_.lock_backing_()
    }

    /// 增加一个 `BumpListAlloc` 句柄引用。
    pub(crate) fn clone_ref_(&self) {
        self.state_.clone_ref_()
    }

    /// 减少一个句柄引用；返回是否已经是最后一个句柄。
    pub(crate) fn release_ref_(&self) -> bool {
        self.state_.release_ref_()
    }

    /// 申请一个普通 Page 并初始化 Page 头；不挂链。
    ///
    /// 普通页按 `align_of::<Page>()` 对齐，首个可分配 cell 是 `Page::CELLS`；第一页
    /// 的元数据更多，由 [`InitPage`] 单独初始化。
    fn create_page_(&self) -> Result<NonNull<Page>, AllocError> {
        let page_bytes = self.page_size_.bytes();
        let layout =
            Layout::from_size_align(page_bytes, mem::align_of::<Page>()).map_err(|_| AllocError)?;
        // SAFETY: backing 在第一页构造后一直指向仍然存活、位于第一页内的 A；
        // backing lock 串行化所有对 A 的调用，因此 A 只需要 `Send`。
        let mem = {
            let _guard = self.lock_backing_();
            unsafe { self.backing_.as_ref() }.allocate(layout)
        }?;
        let page_mem = NonNull::slice_from_raw_parts(mem.cast::<u8>(), page_bytes);
        // SAFETY: page_mem 是刚申请、恰好 page_bytes 字节、按 Page 对齐的独占内存；
        // Page 头占用不超过 page_bytes。
        Ok(
            unsafe {
                Page::init_in_place_(page_mem, NonNull::from(self), Page::CELLS as CellIndex)
            },
        )
    }

    /// 归还一个已摘链、已 drain 的普通 Page。
    ///
    /// 第一页（root）永远不经过这里：它包含 Core 与 A，只能在
    /// [`InitPage::drop_in_place_`] 中用移出的 A 归还。
    fn dealloc_page_(&self, page: NonNull<Page>) {
        debug_assert_ne!(
            page, self.root_page_,
            "the root page must never be deallocated here"
        );
        // SAFETY: page 由 create_page_ 以同一布局申请，且调用方保证已摘链、无页锁持有者。
        let layout = unsafe {
            Layout::from_size_align_unchecked(self.page_size_.bytes(), mem::align_of::<Page>())
        };
        // SAFETY: page/layout 来自本分配器，A 先于 Core 存续；
        // backing lock 串行化所有对 A 的调用，因此 A 只需要 `Send`。
        {
            let _guard = self.lock_backing_();
            let backing = unsafe { self.backing_.as_ref() };
            // SAFETY: 由调用方保证 page/layout 契约。
            unsafe { backing.deallocate(page.cast::<u8>(), layout) };
        }
    }

    /// 把新页追加到 `tail` 之后，并更新 `head_` / `tail_`。
    ///
    /// 调用方必须持有链表锁。
    fn link_after_(&self, inner: &mut PageList, tail: Option<NonNull<Page>>, page: NonNull<Page>) {
        // SAFETY: 调用方持有链表锁；page 是新页，尚未共享。
        let page_ref = unsafe { page.as_ref() };
        unsafe {
            page_ref.set_prev_(tail);
            page_ref.set_next_(None);
        }
        match tail {
            // SAFETY: tail 仍在链表上；链表锁保证没有并发摘链。
            Some(tail_page) => unsafe { tail_page.as_ref().set_next_(Some(page)) },
            None => inner.head_ = Some(page),
        }
        inner.tail_ = Some(page);
    }

    /// 把 Page 从链表摘下。
    ///
    /// 调用方必须持有链表锁，且 `page` 必须在链表上。
    fn unlink_(&self, inner: &mut PageList, page: NonNull<Page>) {
        // SAFETY: 调用方持有链表锁；page 在链表上。
        let page_ref = unsafe { page.as_ref() };
        // SAFETY: 链表锁保证 links_ 没有并发写。
        let (prev, next) = unsafe { (page_ref.prev_(), page_ref.next_()) };
        match prev {
            // SAFETY: 邻居仍在链表上。
            Some(prev_page) => unsafe { prev_page.as_ref().set_next_(next) },
            None => inner.head_ = next,
        }
        match next {
            // SAFETY: 邻居仍在链表上。
            Some(next_page) => unsafe { next_page.as_ref().set_prev_(prev) },
            None => inner.tail_ = prev,
        }
        // SAFETY: 链表锁保护；摘链后清空本页链接，避免误用。
        unsafe {
            page_ref.set_prev_(None);
            page_ref.set_next_(None);
        }
    }

    /// 在链表锁内创建新页、完成本次分配并挂到 `tail` 之后。
    fn create_page_and_alloc_locked_(
        &self,
        inner: &mut PageList,
        tail: Option<NonNull<Page>>,
        layout: Layout,
    ) -> Result<NonNull<u8>, AllocError> {
        let page = self.create_page_()?;
        // SAFETY: page 刚创建、尚未挂链，独占访问。
        let page_ref = unsafe { page.as_ref() };
        let Some(ptr) = page_ref.try_alloc_(page, layout) else {
            self.dealloc_page_(page);
            return Err(AllocError);
        };
        self.link_after_(inner, tail, page);
        Ok(ptr)
    }

    /// 单块是否可能放进一个普通空页（只按数据 cell 做保守下界检查）。
    ///
    /// 第一页因为要容纳 `InitPage<A>`，实际容量更小；它在分配失败时走
    /// “容量不足则追加普通页”的专门判断，不依赖这里。
    fn fits_page_(&self, layout: Layout) -> bool {
        let max_data = self.page_size_.cells().saturating_sub(Page::CELLS + 1);
        layout.size().div_ceil(PTR) <= max_data
    }

    /// 零尺寸分配：不创建 Page，返回按对齐的悬垂地址。
    fn zero_sized_(align: usize) -> Result<NonNull<[u8]>, AllocError> {
        let Some(align) = NonZero::new(align) else {
            return Err(AllocError);
        };
        let ptr = NonNull::<u8>::without_provenance(align);
        Ok(NonNull::slice_from_raw_parts(ptr, 0))
    }

    fn slice_(ptr: NonNull<u8>, len: usize) -> NonNull<[u8]> {
        NonNull::slice_from_raw_parts(ptr, len)
    }

    /// 分配主路径。
    pub(crate) fn allocate_(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return Self::zero_sized_(layout.align());
        }
        if !self.fits_page_(layout) {
            return Err(AllocError);
        }

        loop {
            let list_guard = self.lock_list_();
            // SAFETY: 持有链表锁。
            let tail = unsafe { (*self.inner_.get()).tail_ };
            let Some(tail_page) = tail else {
                // SAFETY: 持有链表锁。
                let inner = unsafe { &mut *self.inner_.get() };
                let ptr = self.create_page_and_alloc_locked_(inner, None, layout)?;
                drop(list_guard);
                return Ok(Self::slice_(ptr, layout.size()));
            };

            // SAFETY: 持有链表锁，tail 是链上的有效 Page。
            let page_ptr = tail_page.as_ptr();
            // SAFETY: 临时借用只用于 try_lock；失败后不保留引用，释放链表锁后重试。
            let Some(page_guard) = (unsafe { &*page_ptr }).try_lock_() else {
                drop(list_guard);
                spin_loop();
                continue;
            };
            drop(list_guard);

            // SAFETY: 持有页锁，页不会被摘链释放。
            let page_ref = unsafe { &*page_ptr };
            // SAFETY: 持有页锁。
            if let Some(ptr) = unsafe { page_ref.try_alloc_locked_(tail_page, layout) } {
                drop(page_guard);
                return Ok(Self::slice_(ptr, layout.size()));
            }
            // 空 tail 页放不下时，普通页的同大小新页也不会容纳这个请求；保持空页复用，
            // 直接返回错误，避免把唯一空 tail 变成非 tail。
            //
            // 第一页是例外：它要容纳 InitPage<A>，容量比普通页小。如果请求的
            // “块头 + 数据 cell”已经超过第一页的 cells_，普通页可能放得下，必须追加
            // 新页；若只是对齐/地址原因放不下，则仍按普通空 tail 的规则返回错误。
            let was_empty = unsafe { page_ref.empty_locked_() };
            let root_capacity_limited = tail_page == self.root_page_
                && (unsafe { page_ref.capacity_locked_() } as usize)
                    < layout.size().div_ceil(PTR).saturating_add(1);
            drop(page_guard);
            if was_empty && !root_capacity_limited {
                return Err(AllocError);
            }

            let list_guard = self.lock_list_();
            // SAFETY: 持有链表锁。
            let current_tail = unsafe { (*self.inner_.get()).tail_ };
            if current_tail != tail {
                drop(list_guard);
                continue;
            }
            // SAFETY: 持有链表锁；current_tail 与观察到的 tail 相同。
            let inner = unsafe { &mut *self.inner_.get() };
            let ptr = self.create_page_and_alloc_locked_(inner, tail, layout)?;
            drop(list_guard);
            return Ok(Self::slice_(ptr, layout.size()));
        }
    }

    /// 释放主路径。
    ///
    /// # Safety
    ///
    /// 必须满足 [`Allocator::deallocate`] 的契约：`ptr` 是通过本 allocator 与 `layout`
    /// 分配的当前块。
    unsafe fn deallocate_(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() == 0 {
            return;
        }

        // SAFETY: 块仍然活跃；由用户块首址反推 Page 与块头（含对齐 padding 的情况）。
        let (page, header_cell) = unsafe { Page::locate_block_(ptr) };
        // `release_page_` 在页变空时会通过 Page 的 owner 指针把回收升级给本分配器。
        let _ = Page::release_page_(page, header_cell);
    }

    /// 回收一个刚变空的 Page。
    ///
    /// 规则：第一页（root）永远保留；除此之外最多保留一个空页。是否为空由
    /// `PageState.returned_ == used_` 判断。
    ///
    /// - root 页变空：如果 tail 已经是一个空页，先归还 tail，再把 root 挪到 tail；
    ///   这样 root 可以在下一次分配中立即复用，同时不会留下两个空页；
    /// - 普通页变空且 root 页已经为空：不能再保留第二个空页，直接摘链并归还；
    /// - 普通页变空且 root 页非空：沿用“只保留一个空 tail”的策略。
    ///
    /// 锁顺序为链表锁 → 页锁（可能再锁一次 tail 页或 root 页）。分配路径只会在持有
    /// 链表锁时用 `try_lock` 取页锁；归还路径在调用本函数前已经释放页锁，因此不会形成
    /// ABBA 死锁。未挂链的 Page（例如单元测试自己构造的 Page）不会被回收。
    pub(crate) fn cleanup_empty_page_(&self, page: NonNull<Page>) {
        let list_guard = self.lock_list_();
        // SAFETY: 调用方保证 page 仍然有效（活跃块刚刚归还或旧 tail）。
        let page_ref = unsafe { page.as_ref() };
        let page_guard = page_ref.lock_();
        // 期间可能被 tail 分配复用，需要重新确认。
        // SAFETY: 持有页锁。
        if unsafe { !page_ref.empty_locked_() } {
            drop(page_guard);
            drop(list_guard);
            return;
        }

        // SAFETY: 持有链表锁。
        let inner = unsafe { &mut *self.inner_.get() };

        if page == self.root_page_ {
            // root 必须保留。如果当前 tail 是另一个空页，先把它摘掉，再让 root 成为
            // tail；否则直接追加到 tail。
            let to_dealloc = if inner.tail_ == Some(page) {
                Option::None
            } else {
                let mut to_dealloc = Option::None;
                if let Some(tail) = inner.tail_ {
                    // SAFETY: tail 仍在链表上，链表锁保证其存活。
                    let tail_ref = unsafe { tail.as_ref() };
                    let tail_guard = tail_ref.lock_();
                    // SAFETY: 持有 tail 的页锁。
                    let tail_empty = unsafe { tail_ref.empty_locked_() };
                    drop(tail_guard);
                    if tail_empty {
                        self.unlink_(inner, tail);
                        to_dealloc = Option::Some(tail);
                    }
                }
                self.unlink_(inner, page);
                self.link_after_(inner, inner.tail_, page);
                to_dealloc
            };
            drop(page_guard);
            drop(list_guard);
            if let Option::Some(tail) = to_dealloc {
                self.dealloc_page_(tail);
            }
            return;
        }

        // SAFETY: 链表锁保证 links_ 没有并发写。
        let linked = inner.head_ == Some(page)
            || unsafe { page_ref.prev_() }.is_some()
            || unsafe { page_ref.next_() }.is_some();
        if !linked {
            // 不在本 allocator 链表上的 Page（测试辅助构造）不参与空页回收。
            drop(page_guard);
            drop(list_guard);
            return;
        }

        // root 页已经为空时不能再留下第二个空页；直接归还这个普通页。
        let root_empty = {
            // SAFETY: root_page_ 始终在链表中，链表锁保证其存活。
            let root_ref = unsafe { self.root_page_.as_ref() };
            let root_guard = root_ref.lock_();
            // SAFETY: 持有 root 的页锁。
            let empty = unsafe { root_ref.empty_locked_() };
            drop(root_guard);
            empty
        };
        if root_empty {
            self.unlink_(inner, page);
            drop(page_guard);
            drop(list_guard);
            self.dealloc_page_(page);
            return;
        }

        if inner.tail_ == Some(page) {
            // 已经是 tail 空页：保留并复用，不归还。
            drop(page_guard);
            drop(list_guard);
            return;
        }

        self.unlink_(inner, page);
        // 检查原 tail 是否为空。链表锁保证 tail 不会被摘链/释放，页锁保证状态稳定。
        let tail_is_empty = match inner.tail_ {
            Some(tail) => {
                // SAFETY: tail 仍在链表上，链表锁保证其存活。
                let tail_ref = unsafe { tail.as_ref() };
                let tail_guard = tail_ref.lock_();
                // SAFETY: 持有 tail 的页锁。
                let empty = unsafe { tail_ref.empty_locked_() };
                drop(tail_guard);
                empty
            }
            None => false,
        };
        if tail_is_empty {
            // 已经有一个空 tail 页；这个额外的空页归还。
            drop(page_guard);
            drop(list_guard);
            self.dealloc_page_(page);
        } else {
            // 还没有空页：把这个空页挪到 tail 保留复用，避免归还唯一空页。
            let old_tail = inner.tail_;
            self.link_after_(inner, old_tail, page);
            drop(page_guard);
            drop(list_guard);
        }
    }

    /// 析构第一步：归还所有普通页，保留 root 页。
    ///
    /// 只由 [`InitPage::drop_in_place_`] 在独占访问时调用。调用前后 A 必须仍然有效；
    /// 这里通过 `backing_` 归还普通页，root 页则留给调用方在移出 A 后归还。
    pub(crate) fn free_non_root_pages_(&mut self, root: NonNull<Page>) {
        debug_assert_eq!(root, self.root_page_, "root page mismatch");
        let head = {
            let inner = self.inner_.get_mut();
            let head = inner.head_.take();
            inner.tail_ = Option::None;
            head
        };
        let mut cursor = head;
        while let Some(page) = cursor {
            // 先把 `next` 与扫描结果取出来，随后释放 `page.as_ref()` 的借用，
            // 最后才可能归还 page 本身。
            let (next, scan) = {
                // SAFETY: page 在链表上且还没有被归还；独占析构期间没有并发访问。
                let page_ref = unsafe { page.as_ref() };
                // SAFETY: 同上。
                let next = unsafe { page_ref.next_() };
                (next, page_ref.scan_chain_(page))
            };
            #[cfg(any(test, feature = "std"))]
            {
                match scan {
                    Ok((live, _freed)) if live > 0 => Self::warn_live_blocks_(live),
                    Ok(_) => {}
                    Err(reason) => Self::warn_chain_corrupt_(reason),
                }
            }
            #[cfg(not(any(test, feature = "std")))]
            {
                let _ = scan;
            }

            if page != root {
                self.dealloc_page_(page);
            }
            cursor = next;
        }

        // root 的 links 还指向已经归还的页；清空它们，避免留下悬垂指针。
        // SAFETY: root 仍然有效，独占析构期间没有并发访问。
        let root_ref = unsafe { root.as_ref() };
        // SAFETY: 独占析构期间，links_ 没有并发写。
        unsafe {
            root_ref.set_prev_(Option::None);
            root_ref.set_next_(Option::None);
        }
    }

    #[cfg(any(test, feature = "std"))]
    fn warn_live_blocks_(live: u16) {
        use std::io::Write;
        let _ = writeln!(
            std::io::stderr(),
            "bumplist-alloc: dropping allocator with {live} live block(s)"
        );
    }

    #[cfg(any(test, feature = "std"))]
    fn warn_chain_corrupt_(reason: &'static str) {
        use std::io::Write;
        let _ = writeln!(
            std::io::stderr(),
            "bumplist-alloc: page block chain is corrupt: {reason}"
        );
    }
}

/// 以固定大小 Page 链表组织的 bump 分配器。
///
/// # 生命周期与 Clone
///
/// [`Allocator`] 同时为 `BumpListAlloc` 与 `&BumpListAlloc` 实现：
///
/// - 按值使用（如 `Vec<u32, BumpListAlloc>`）时，容器拥有一个句柄，块的生命周期与
///   容器 / arena 句柄一致；
/// - 按借用使用（如 `Vec<u32, &BumpListAlloc>`）时，分配块携带借期。
///
/// ```
/// use bumplist_alloc::{BumpListAlloc, PageSize};
/// use std::vec::Vec;
///
/// let bump = BumpListAlloc::new(PageSize::P4K)?;
/// let mut values: Vec<u32, &BumpListAlloc> = Vec::new_in(&bump);
/// values.extend_from_slice(&[1, 2, 3]);
/// assert_eq!(&values[..], &[1, 2, 3]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// `BumpListAlloc::clone` 只增加第一页 Core 里的引用计数，不复制 Page，也不复制底层
/// 分配器 `A`；所有 clone 共享同一个 arena。最后一个句柄析构时才归还所有 Page 与
/// 第一页内存。需要 `Arc` / `Rc` 等外部共享句柄时仍可使用 [`BumpListAllocRef`](crate::BumpListAllocRef)。
///
/// # 线程安全
///
/// 类型参数要求 `A: Send`。所有对底层分配器 `A` 的调用（页申请 / 页归还）都由
/// Core 的 backing lock 串行化，因此 `BumpListAlloc<A>` 是 `Send + Sync`，不要求
/// `A: Sync`。`A: !Send` 的分配器不能用于本类型；例如需要 trait object 时，应使用
/// `&(dyn Allocator + Sync)` 而不是 `&dyn Allocator`。
///
/// # 单指针句柄与可移动性
///
/// 所有状态都放在第一页的 `InitPage<A>` 中：Page 头、非泛型 `BumpListCore` 与
/// 底层分配器 `A`。`BumpListAlloc` 只是一个 `NonNull<InitPage<A>>`（加一个零大小的
/// 所有权标记），因此与指针同宽。移动句柄只移动这个指针，不会移动第一页；Page 的
/// owner 指针始终指向第一页里稳定的 Core，所以创建 Page 之后仍然可以安全移动。
///
/// [`BumpListAlloc::new_in`] 会立即申请第一页，因此构造本身可能失败（页大小不足以
/// 容纳元数据，或底层分配器申请失败）。
#[repr(transparent)]
pub struct BumpListAlloc<A: Allocator + Send = Global> {
    /// 第一页：`Page`、`BumpListCore` 与 `MaybeUninit<A>` 都在它里面。
    page_: NonNull<InitPage<A>>,

    /// 表示本句柄拥有 `A`（用于 dropck 与方差）；不占空间。
    _owns_a_: PhantomData<A>,
}

// SAFETY: `PageList` 与 Page 链表中的 `NonNull` 指针由 list lock 保护；Page 的
// `state_` 由页锁保护；第一页在构造后地址稳定，移动句柄只移动指针。`A` 的 `Send`
// 允许整个 allocator（包括最终从第一页移出的 A）转移线程。
unsafe impl<A: Allocator + Send> Send for BumpListAlloc<A> {}

// SAFETY: 共享 `&BumpListAlloc` 时，链表由 list lock 保护，页状态由页锁保护；
// 所有对底层分配器 `A` 的调用都由 backing lock 串行化，`A: Send` 足以让这把锁把
// A 的独占访问安全地借给任意线程。Page 的 `links_` / `state_` 使用 `UnsafeCell`，
// 分别由两把锁保护不相交字段；第一页在句柄共享期间不会移动。
unsafe impl<A: Allocator + Send> Sync for BumpListAlloc<A> {}

impl<A: Allocator + Send> Clone for BumpListAlloc<A> {
    /// 克隆句柄：增加第一页 Core 中的引用计数，不申请 Page，不复制 `A`。
    ///
    /// # Panics
    ///
    /// 引用计数达到可表示上限时 panic；该上限为低位掩码（64 位平台 `2^62 - 1`），
    /// 实际使用中不可达。
    fn clone(&self) -> Self {
        self.core_().clone_ref_();
        Self {
            page_: self.page_,
            _owns_a_: PhantomData,
        }
    }
}

impl<A: Allocator + Send> Drop for BumpListAlloc<A> {
    fn drop(&mut self) {
        let last = self.core_().release_ref_();
        if last {
            // SAFETY: 引用计数减到 0，说明没有其它句柄、引用或正在进行的 Core 操作；
            // `page_` 来自 `new_in` 且尚未被归还。
            unsafe { InitPage::drop_in_place_(self.page_) };
        }
    }
}

impl<A: Allocator + Send> BumpListAlloc<A> {
    /// 使用指定的底层分配器与页大小创建一个空 allocator。
    ///
    /// 底层分配器 `A` 必须满足 `Send`。构造时会立即申请第一页，并在其中原地初始化
    /// Page 头、Core 与 `alloc`；第一页同时是该 allocator 的第一个可分配页。
    ///
    /// # Errors
    ///
    /// - 页大小在当前平台不受支持，或普通页内连 Page 头、一个块头与一个数据 cell 都
    ///   放不下时返回 [`BumpListAllocError::UnsupportedPageSize`]；
    /// - 页大小放不下第一页的 `Page`、Core、`A` 与一个最小分配块时返回
    ///   [`BumpListAllocError::PageSizeTooSmallForAllocator`]；
    /// - 底层分配器无法提供第一页时返回
    ///   [`BumpListAllocError::InitPageAllocationFailed`]。
    ///
    /// # Examples
    ///
    /// ```
    /// use bumplist_alloc::{BumpListAlloc, PageSize};
    /// use std::alloc::Global;
    ///
    /// let bump = BumpListAlloc::new_in(Global, PageSize::P2K)?;
    /// assert_eq!(bump.page_size(), PageSize::P2K);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new_in(alloc: A, page_size: PageSize) -> Result<Self, BumpListAllocError> {
        let page_ = InitPage::new_(alloc, page_size)?;
        Ok(Self {
            page_,
            _owns_a_: PhantomData,
        })
    }

    /// 当前实例的页大小。
    ///
    /// # Examples
    ///
    /// ```
    /// use bumplist_alloc::{BumpListAlloc, PageSize};
    ///
    /// let bump = BumpListAlloc::new(PageSize::P1K)?;
    /// assert_eq!(bump.page_size(), PageSize::P1K);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn page_size(&self) -> PageSize {
        self.core_().page_size_()
    }

    /// 第一页的只读视图。
    fn init_page_(&self) -> &InitPage<A> {
        // SAFETY: `page_` 由 `new_in` 分配，指向一块按 `InitPage<A>` 对齐的独占内存；
        // 它在 `Drop` 之前一直有效，且 `BumpListAlloc` 只保存这个指针。
        unsafe { self.page_.as_ref() }
    }

    /// 非泛型核心的只读视图。
    pub(crate) fn core_(&self) -> &BumpListCore {
        &self.init_page_().core_
    }
}

impl BumpListAlloc<Global> {
    /// 使用 `Global` 作为底层分配器创建 allocator。
    ///
    /// # Errors
    ///
    /// 同 [`BumpListAlloc::new_in`]。
    ///
    /// # Examples
    ///
    /// ```
    /// use bumplist_alloc::{BumpListAlloc, PageSize};
    ///
    /// let bump = BumpListAlloc::new(PageSize::P1K)?;
    /// assert_eq!(bump.page_size(), PageSize::P1K);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(page_size: PageSize) -> Result<Self, BumpListAllocError> {
        Self::new_in(Global, page_size)
    }
}

unsafe impl<A: Allocator + Send> Allocator for BumpListAlloc<A> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.core_().allocate_(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: 由 Allocator::deallocate 的调用方保证契约。
        unsafe { self.core_().deallocate_(ptr, layout) };
    }
}

// SAFETY: `Clone` 只增加共享 Core 的引用计数；clone 与原来的句柄指向同一个 arena，
// 因此任意 clone 都能释放另一 clone 分配的块。
unsafe impl<A: Allocator + Send> AllocatorClone for BumpListAlloc<A> {}

#[cfg(test)]
mod tests_ {
    use super::*;
    use crate::page_::Page;
    use std::vec::Vec;

    /// 测试 `TrArena` 的 `Box` 析构后把块归还给页。
    /// Test that dropping a `TrArena` `Box` releases its block back to the page.
    ///
    /// - 手段：用 `TrArena::alloc_value_async` 分配一个 `u64`，检查 live 为 1；drop 句柄后再检查。
    /// - 判定：drop 后 live 回到 0，且唯一空页仍被保留为 tail（页数为 1）。
    ///
    /// - Means: allocate a `u64` through `TrArena::alloc_value_async`, observe live == 1,
    ///   then drop the handle.
    /// - Verdict: live returns to 0 and the single empty page is kept as the tail (page
    ///   count stays 1).
    #[cfg(feature = "denco_arena")]
    #[test]
    fn denco_arena_box_release_returns_block_() {
        use core::{
            future::Future,
            pin::pin,
            task::{Context, Poll, Waker},
        };

        use abs_denco::de::TrArena;

        fn block_on_<TyFut>(future: TyFut) -> TyFut::Output
        where
            TyFut: Future,
        {
            let mut cx = Context::from_waker(Waker::noop());
            let mut future = pin!(future);
            loop {
                match future.as_mut().poll(&mut cx) {
                    Poll::Ready(output) => return output,
                    Poll::Pending => core::hint::spin_loop(),
                }
            }
        }

        let bump = BumpListAlloc::new(crate::PageSize::P4K).expect("4 KiB 页应当受支持");
        let value = block_on_(bump.alloc_value_async(7u64)).expect("arena 分配应当成功");
        assert_eq!(count_live_(&bump), 1u16);
        drop(value);
        assert_eq!(count_live_(&bump), 0u16);
        assert_eq!(count_pages_(&bump), 1usize);
    }

    fn count_pages_(bump: &BumpListAlloc<Global>) -> usize {
        // SAFETY: 单线程测试，没有并发访问。
        let inner = unsafe { &*bump.core_().inner_.get() };
        let mut cursor = inner.head_;
        let mut count = 0;
        while let Some(page) = cursor {
            count += 1;
            // SAFETY: 单线程测试，链表有效。
            cursor = unsafe { page.as_ref().next_() };
        }
        count
    }

    fn count_live_(bump: &BumpListAlloc<Global>) -> u16 {
        // SAFETY: 单线程测试，没有并发访问。
        let inner = unsafe { &*bump.core_().inner_.get() };
        let mut cursor = inner.head_;
        let mut total = 0u16;
        while let Some(page) = cursor {
            // SAFETY: 单线程测试，Page 有效。
            let page_ref = unsafe { page.as_ref() };
            let (live, _freed) = page_ref.scan_chain_(page).expect("链应当有效");
            total = total.checked_add(live).expect("live 计数不溢出");
            // SAFETY: 单线程测试，链表有效。
            cursor = unsafe { page_ref.next_() };
        }
        total
    }

    fn page_of_block_(ptr: NonNull<u8>) -> NonNull<Page> {
        // SAFETY: 块仍然活跃，由 locator/块头反推页首。
        unsafe { Page::locate_block_(ptr).0 }
    }

    fn block_ptr_(block: NonNull<[u8]>) -> NonNull<u8> {
        block.cast::<u8>()
    }

    /// 测试目标：构造时创建第一页；第一次分配复用它，释放最后一个块后仍保留这一页。
    /// Test goal: Construction creates the first page; the first allocation reuses it and
    /// releasing the last block keeps it.
    /// - 手段 / Method: 构造 4 KiB 页后检查页数，分配一个 8 字节块并释放，再分配一次。
    /// - 判定 / Assertion: 页数始终为 1，live 数 1→0→1；块头反查出的页首等于链头。
    #[test]
    fn first_allocation_reuses_root_page() {
        let bump = BumpListAlloc::new(PageSize::P4K).expect("4 KiB 页");
        assert_eq!(count_pages_(&bump), 1);

        let layout = Layout::from_size_align(8, 8).expect("布局合法");
        let block = bump.core_().allocate_(layout).expect("分配成功");
        let ptr = block_ptr_(block);
        assert_eq!(count_pages_(&bump), 1);
        assert_eq!(count_live_(&bump), 1);

        // SAFETY: 单线程测试，链头有效。
        let head = unsafe { (*bump.core_().inner_.get()).head_ }.expect("有页");
        assert_eq!(page_of_block_(ptr), head);

        // SAFETY: 指针来自同一 allocator 与 layout。
        unsafe { bump.core_().deallocate_(ptr, layout) };
        // 唯一空 Page 被保留为 tail，不归还；它应该可以直接复用。
        assert_eq!(count_pages_(&bump), 1);
        assert_eq!(count_live_(&bump), 0);

        let block_again = bump.core_().allocate_(layout).expect("空页应当可复用");
        assert_eq!(count_pages_(&bump), 1);
        assert_eq!(count_live_(&bump), 1);
        assert_eq!(page_of_block_(block_ptr_(block_again)), head);

        // SAFETY: 指针来自同一 allocator 与 layout。
        unsafe { bump.core_().deallocate_(block_ptr_(block_again), layout) };
        assert_eq!(count_pages_(&bump), 1);
        assert_eq!(count_live_(&bump), 0);
    }

    /// 测试目标：多页分配、乱序释放时页不会提前消失，全部释放后只保留 root 页。
    /// Test goal: With multiple pages and out-of-order release, pages stay alive until
    /// all their blocks are freed; only the root page remains at the end.
    /// - 手段 / Method: 1 KiB 页上分配 12 个 200 字节块，先释放偶数下标，再释放奇数。
    /// - 判定 / Assertion: 页数至少为 2；中间 live=6；最后 live=0，且只保留 root 页。
    #[test]
    fn multiple_pages_and_out_of_order_release() {
        let bump = BumpListAlloc::new(PageSize::P1K).expect("1 KiB 页");
        let layout = Layout::from_size_align(200, 8).expect("布局合法");
        let mut blocks = Vec::new();

        for _ in 0..12 {
            let block = bump.core_().allocate_(layout).expect("块应当放得下");
            blocks.push(block_ptr_(block));
        }
        assert!(count_pages_(&bump) >= 2);
        assert_eq!(count_live_(&bump), 12);

        for index in (0..12).step_by(2) {
            // SAFETY: 指针来自同一 allocator 与 layout。
            unsafe { bump.core_().deallocate_(blocks[index], layout) };
        }
        assert_eq!(count_live_(&bump), 6);

        for index in (1..12).step_by(2) {
            // SAFETY: 指针来自同一 allocator 与 layout。
            unsafe { bump.core_().deallocate_(blocks[index], layout) };
        }
        assert_eq!(count_live_(&bump), 0);
        // 多页都空掉后只保留 root 页，其余归还。
        assert_eq!(count_pages_(&bump), 1);
    }

    /// 测试目标：root 空页因对齐放不下时直接返回 `AllocError`，不追加新页，也不破坏
    /// 空页的复用状态。
    /// Test goal: When the empty root page cannot satisfy an alignment, return `AllocError`
    /// without appending a new page or losing the cached empty page.
    /// - 手段 / Method: 1 KiB 页先分配/释放一个块得到空 root，再请求 1 MiB 对齐的块。
    /// - 判定 / Assertion: 请求失败；页数仍为 1，live 仍为 0，空页仍可复用。
    #[test]
    fn empty_tail_alignment_failure_does_not_append() {
        let bump = BumpListAlloc::new(PageSize::P1K).expect("1 KiB 页");
        let layout = Layout::from_size_align(8, 8).expect("布局合法");
        let block = bump.core_().allocate_(layout).expect("分配成功");
        let ptr = block_ptr_(block);
        // SAFETY: 指针来自同一 allocator 与 layout。
        unsafe { bump.core_().deallocate_(ptr, layout) };
        assert_eq!(count_pages_(&bump), 1);
        assert_eq!(count_live_(&bump), 0);

        let over_aligned = Layout::from_size_align(8, 1 << 20).expect("布局合法");
        assert!(bump.core_().allocate_(over_aligned).is_err());
        assert_eq!(count_pages_(&bump), 1);
        assert_eq!(count_live_(&bump), 0);

        // 空页仍然可用。
        let again = bump.core_().allocate_(layout).expect("空页应当仍可复用");
        assert_eq!(count_live_(&bump), 1);
        // SAFETY: 指针来自同一 allocator 与 layout。
        unsafe { bump.core_().deallocate_(block_ptr_(again), layout) };
        assert_eq!(count_pages_(&bump), 1);
    }

    /// 测试目标：root 页容量不足、但请求能放进普通页时会追加普通页。
    /// Test goal: When the root page is too small but the request fits a normal page, a
    /// normal page is appended.
    /// - 手段 / Method: 在 1 KiB 页上分配一个 960 字节块（超过 root 可用容量，但小于
    ///   普通页容量）。
    /// - 判定 / Assertion: 分配成功且页数为 2；释放后普通页被归还，只保留 root 页。
    #[test]
    fn root_capacity_spills_to_normal_page() {
        let bump = BumpListAlloc::new(PageSize::P1K).expect("1 KiB 页");
        let layout = Layout::from_size_align(960, 8).expect("布局合法");
        let block = bump
            .core_()
            .allocate_(layout)
            .expect("普通页应当放得下 960 字节");
        assert_eq!(count_pages_(&bump), 2);
        assert_eq!(count_live_(&bump), 1);
        // SAFETY: 指针来自同一 allocator 与 layout。
        unsafe { bump.core_().deallocate_(block_ptr_(block), layout) };
        assert_eq!(count_live_(&bump), 0);
        assert_eq!(count_pages_(&bump), 1);
    }

    /// 测试目标：超过页容量的请求直接返回 `AllocError`，不追加 Page。
    /// Test goal: A request larger than the page returns `AllocError` without appending a
    /// page.
    /// - 手段 / Method: 1 KiB 页上请求 8 KiB。
    /// - 判定 / Assertion: `allocate_` 返回错误，页数仍为构造时的 1。
    #[test]
    fn oversized_request_is_rejected_without_page() {
        let bump = BumpListAlloc::new(PageSize::P1K).expect("1 KiB 页");
        let layout = Layout::from_size_align(8 * 1024, 8).expect("布局合法");
        assert!(bump.core_().allocate_(layout).is_err());
        assert_eq!(count_pages_(&bump), 1);
        assert_eq!(count_live_(&bump), 0);
    }

    /// 测试目标：零尺寸分配不追加 Page，释放也不改变链表。
    /// Test goal: A zero-sized allocation appends no page and its deallocation leaves the
    /// list unchanged.
    /// - 手段 / Method: 1 KiB 页上分配/释放 `Layout::new::<()>()`。
    /// - 判定 / Assertion: 页数始终为 1（root 页），返回地址按对齐。
    #[test]
    fn zero_sized_allocation_has_no_page() {
        let bump = BumpListAlloc::new(PageSize::P1K).expect("1 KiB 页");
        let layout = Layout::new::<()>();
        let block = bump.core_().allocate_(layout).expect("零尺寸分配成功");
        let ptr = block_ptr_(block);
        assert_eq!(count_pages_(&bump), 1);
        assert_eq!(ptr.as_ptr() as usize % layout.align(), 0);

        // SAFETY: 同一 allocator 与 layout 的零尺寸分配。
        unsafe { bump.core_().deallocate_(ptr, layout) };
        assert_eq!(count_pages_(&bump), 1);
    }

    /// 测试目标：64 位下 256 KiB 页可用，页内 cell 数达到上限；32 位下构造失败。
    /// Test goal: The 256 KiB page is usable on 64-bit and rejected on 32-bit.
    /// - 手段 / Method: 按目标指针宽度构造 `PageSize::P256K`。
    /// - 判定 / Assertion: 64 位页 cell 数为 32768，且可完成一次分配/释放；32 位返回错误。
    #[test]
    fn p256k_page_boundary() {
        #[cfg(target_pointer_width = "64")]
        {
            let bump = BumpListAlloc::new(PageSize::P256K).expect("64 位支持 256 KiB 页");
            assert_eq!(PageSize::P256K.cells(), 1usize << 15);
            let layout = Layout::from_size_align(8, 8).expect("布局合法");
            let block = bump.core_().allocate_(layout).expect("分配成功");
            let ptr = block_ptr_(block);
            assert_eq!(count_live_(&bump), 1);
            // SAFETY: 指针来自同一 allocator 与 layout。
            unsafe { bump.core_().deallocate_(ptr, layout) };
            // 256 KiB 页也遵循“只保留 root 空页”的策略。
            assert_eq!(count_pages_(&bump), 1);
            assert_eq!(count_live_(&bump), 0);
        }

        #[cfg(target_pointer_width = "32")]
        {
            assert!(BumpListAlloc::new(PageSize::P256K).is_err());
        }
    }

    /// 测试目标：`BumpListAlloc` 句柄与一个指针同宽。
    /// Test goal: The `BumpListAlloc` handle is exactly one pointer wide.
    /// - 手段 / Method: 编译期/运行期比较 `size_of::<BumpListAlloc<Global>>()` 与
    ///   `size_of::<usize>()`。
    /// - 判定 / Assertion: 两者相等。
    #[test]
    fn allocator_handle_is_one_pointer() {
        assert_eq!(
            core::mem::size_of::<BumpListAlloc<Global>>(),
            core::mem::size_of::<usize>()
        );
    }

    /// 测试目标：持有活跃块时移动 `BumpListAlloc`，块仍能正确归还。
    /// Test goal: Moving `BumpListAlloc` while a block is live keeps the block and its
    /// page valid.
    /// - 手段 / Method: 通过 `&bump` 分配并写入一个 64 字节块，结束借用后把句柄移动到
    ///   新变量，再从新位置读取数据并通过 `&moved` 归还。
    /// - 判定 / Assertion: 移动后读回原值，归还成功且不崩溃；这依赖 Page owner 指向
    ///   堆上稳定的第一页而不是句柄地址。
    #[test]
    fn moving_allocator_keeps_pages_valid() {
        let bump = BumpListAlloc::new(PageSize::P4K).expect("4 KiB 页");
        let layout = Layout::from_size_align(64, 8).expect("布局合法");
        let ptr = {
            let alloc: &BumpListAlloc = &bump;
            let block = Allocator::allocate(&alloc, layout).expect("分配成功");
            block.cast::<u8>()
        };
        // SAFETY: 块至少 8 字节，写入一个 u64 不越界。
        unsafe { ptr.cast::<u64>().write(0xA5A5_5A5A_1234_5678) };

        let moved = bump;
        let alloc: &BumpListAlloc = &moved;
        // SAFETY: 块仍然活跃，读取之前写入的 u64。
        assert_eq!(unsafe { ptr.cast::<u64>().read() }, 0xA5A5_5A5A_1234_5678);
        // SAFETY: 指针来自同一 allocator 与 layout。
        unsafe { Allocator::deallocate(&alloc, ptr, layout) };
        drop(moved);
    }

    /// 2 KiB 的底层分配器，用来触发“第一页放不下全部元数据”。
    struct BigAlloc {
        _padding: [u8; 2048],
    }

    unsafe impl Allocator for BigAlloc {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            Global.allocate(layout)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            // SAFETY: 由 Allocator::deallocate 的调用方保证契约。
            unsafe { Global.deallocate(ptr, layout) }
        }
    }

    /// 测试目标：第一页无法容纳 `A` 时返回 `PageSizeTooSmallForAllocator`。
    /// Test goal: When the first page cannot hold `A`, `PageSizeTooSmallForAllocator` is
    /// returned.
    /// - 手段 / Method: 用 2 KiB 的 `BigAlloc` 在 1 KiB 页上构造 allocator。
    /// - 判定 / Assertion: 返回 `PageSizeTooSmallForAllocator { page_size: P1K }`，且
    ///   不会尝试申请第一页。
    #[test]
    fn page_size_too_small_for_big_allocator() {
        let result = BumpListAlloc::new_in(
            BigAlloc {
                _padding: [0u8; 2048],
            },
            PageSize::P1K,
        );
        assert!(matches!(
            result,
            Err(crate::BumpListAllocError::PageSizeTooSmallForAllocator {
                page_size: PageSize::P1K
            })
        ));
    }

    /// 测试目标：拥有型 `BumpListAlloc` 可以直接使用方法语法调用 `Allocator`。
    /// Test goal: An owned `BumpListAlloc` can call `Allocator` methods with method
    /// syntax.
    /// - 手段 / Method: `bump.allocate(layout)` 分配一个 `u8`，再用
    ///   `bump.deallocate(...)` 归还。
    /// - 判定 / Assertion: 两次调用都成功，说明方法解析没有因 `&BumpListAlloc` 的
    ///   通用 impl 产生歧义。
    #[test]
    fn owned_allocator_method_syntax() {
        let bump = BumpListAlloc::new(PageSize::P4K).expect("4 KiB 页");
        let layout = Layout::new::<u8>();
        let block = bump.allocate(layout).expect("分配成功");
        // SAFETY: 指针来自同一 allocator 与 layout。
        unsafe { bump.deallocate(block.cast(), layout) };
    }

    /// 测试目标：`BumpListAlloc` 与 `&BumpListAlloc` 都实现 `Clone` 与
    /// `AllocatorClone`。
    /// Test goal: Both `BumpListAlloc` and `&BumpListAlloc` implement `Clone` and
    /// `AllocatorClone`.
    /// - 手段 / Method: 编译期 trait 断言。
    /// - 判定 / Assertion: 断言通过。
    #[test]
    fn allocator_is_clone_and_allocator_clone() {
        fn assert_allocator_clone<T: Allocator + AllocatorClone + Clone>() {}
        assert_allocator_clone::<BumpListAlloc<Global>>();
        assert_allocator_clone::<&BumpListAlloc<Global>>();
    }

    /// 测试目标：`BumpListAlloc` 及其共享引用满足 `Send + Sync`。
    /// Test goal: `BumpListAlloc` and its shared reference are `Send + Sync`.
    /// - 手段 / Method: 编译期 trait 断言。
    /// - 判定 / Assertion: 断言通过。
    #[test]
    fn allocator_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BumpListAlloc<Global>>();
        assert_send_sync::<&BumpListAlloc<Global>>();
    }
}
