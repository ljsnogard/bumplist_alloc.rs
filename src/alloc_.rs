//! `BumpListAlloc`：链表锁 + 每页页锁的固定页 bump 分配器。

use alloc::alloc::Global;
use core::{
    alloc::{AllocError, Allocator, AllocatorClone, Layout},
    cell::UnsafeCell,
    hint::spin_loop,
    mem,
    num::NonZero,
    ptr::NonNull,
    sync::atomic::{AtomicPtr, Ordering},
};

use crate::{
    error_::BumpListAllocError,
    lock_::{SpinGuard, SpinLock},
    page_::{CellIndex, PTR, Page},
    page_size_::PageSize,
};

/// Page 链表的头尾；由 `BumpListAlloc::list_lock_` 保护。
struct PageList {
    head_: Option<NonNull<Page>>,
    tail_: Option<NonNull<Page>>,
}

impl PageList {
    const fn empty_() -> Self {
        Self {
            head_: None,
            tail_: None,
        }
    }
}

/// 页申请函数指针；参数是整个 `BumpListAlloc<A>` 的地址。
type PageAllocFn = unsafe fn(*const (), Layout) -> Result<NonNull<[u8]>, AllocError>;

/// 页归还函数指针；参数是整个 `BumpListAlloc<A>` 的地址。
type PageDeallocFn = unsafe fn(*const (), NonNull<u8>, Layout);

/// 从整个 `BumpListAlloc<A>` 的地址取出 `alloc_: A`。
///
/// # Safety
///
/// `base` 必须指向一个真实的 `BumpListAlloc<A>`，且其地址在调用期间有效。
unsafe fn allocator_ptr_<A: Allocator + Send>(base: *const ()) -> *const A {
    // SAFETY: 字段偏移由编译器给出；base 带整个结构体的 provenance。
    unsafe {
        base.cast::<u8>()
            .add(core::mem::offset_of!(BumpListAlloc<A>, alloc_))
            .cast::<A>()
    }
}

/// # Safety
///
/// `base` 必须指向一个真实且仍然存活的 `BumpListAlloc<A>`。
unsafe fn alloc_page_<A: Allocator + Send>(
    base: *const (),
    layout: Layout,
) -> Result<NonNull<[u8]>, AllocError> {
    // SAFETY: 由调用方保证 base 指向 BumpListAlloc<A>。
    let alloc = unsafe { &*allocator_ptr_::<A>(base) };
    <A as Allocator>::allocate(alloc, layout)
}

/// # Safety
///
/// `base` 必须指向一个真实且仍然存活的 `BumpListAlloc<A>`，且 `page`/`layout` 来自该分配器。
unsafe fn dealloc_page_<A: Allocator + Send>(base: *const (), page: NonNull<u8>, layout: Layout) {
    // SAFETY: 由调用方保证 base 指向 BumpListAlloc<A>。
    let alloc = unsafe { &*allocator_ptr_::<A>(base) };
    // SAFETY: 由调用方保证 page/layout 契约。
    unsafe { <A as Allocator>::deallocate(alloc, page, layout) };
}

/// 与 `A` 无关的分配器核心：页大小、链表锁、页链表与页 backing。
///
/// 它直接内联在 `BumpListAlloc<A>` 里；Page 的 owner 指向它。真正需要 `A` 的建页 /
/// 还页使用 Core 里的两个单态化函数指针；Core 保存一个指向整个 `BumpListAlloc<A>` 的
/// 原子指针，函数用 `offset_of!` 从它取出 `alloc_`。因此不需要 Box<A>，也不需要每个
/// Page 保存函数指针。该指针在每次分配 / 归还 / 析构前刷新，所以移动 `BumpListAlloc`
/// 后下一次操作会重新指向新地址。
///
/// 但 Page 的 owner 指向内联 Core，因此创建第一个 Page 之后仍不能移动 `BumpListAlloc`。
pub(crate) struct BumpListCore {
    page_size_: PageSize,
    page_cells_: u16,
    page_bytes_: usize,
    list_lock_: SpinLock,
    /// 串行化对底层分配器 `A` 的所有 allocate / deallocate 调用。
    ///
    /// `A: Send` 只保证 A 可以独占地在任意线程使用；并发调用仍需要本锁。
    backing_lock_: SpinLock,
    inner_: UnsafeCell<PageList>,
    backing_alloc_: PageAllocFn,
    backing_dealloc_: PageDeallocFn,
    self_ptr_: AtomicPtr<()>,
}

impl BumpListCore {
    /// 用底层分配器类型与页大小构造核心；数据指针在第一次操作前刷新。
    fn new_<A: Allocator + Send>(page_size: PageSize) -> Result<Self, BumpListAllocError> {
        if !page_size.is_supported() {
            return Err(BumpListAllocError::UnsupportedPageSize { page_size });
        }
        let cells = page_size.cells();
        if cells < Page::CELLS + 2 {
            return Err(BumpListAllocError::UnsupportedPageSize { page_size });
        }
        Ok(Self {
            page_size_: page_size,
            page_cells_: cells as u16,
            page_bytes_: page_size.bytes(),
            list_lock_: SpinLock::new(),
            backing_lock_: SpinLock::new(),
            inner_: UnsafeCell::new(PageList::empty_()),
            backing_alloc_: alloc_page_::<A>,
            backing_dealloc_: dealloc_page_::<A>,
            self_ptr_: AtomicPtr::new(core::ptr::null_mut()),
        })
    }

    /// 当前实例的页大小。
    pub(crate) fn page_size_(&self) -> PageSize {
        self.page_size_
    }

    /// 把整分配器地址写入 Core；`offset_of!` 之后从这里取 `alloc_`。
    pub(crate) fn set_self_ptr_<A: Allocator + Send>(&self, this: &BumpListAlloc<A>) {
        self.self_ptr_.store(
            this as *const BumpListAlloc<A> as *mut (),
            Ordering::Release,
        );
    }

    fn self_ptr_(&self) -> *const () {
        self.self_ptr_.load(Ordering::Acquire).cast()
    }

    fn lock_list_(&self) -> SpinGuard<'_> {
        self.list_lock_.lock_()
    }

    /// 申请一个 Page 并初始化 Page 头；不挂链。
    fn create_page_(&self) -> Result<NonNull<Page>, AllocError> {
        let layout = Layout::from_size_align(self.page_bytes_, mem::align_of::<Page>())
            .map_err(|_| AllocError)?;
        let base = self.self_ptr_();
        // SAFETY: base 在最近一次操作前刷新为整个 BumpListAlloc<A>；A 仍然存活。
        // `backing_lock_` 串行化所有对 A 的调用，因此 A 只需要 `Send`。
        let mem = {
            let _guard = self.backing_lock_.lock_();
            unsafe { (self.backing_alloc_)(base, layout) }
        }?;
        // SAFETY: mem 是刚申请、至少 page_bytes 字节、按 Page 对齐的独占内存；
        // Page 头占用不超过 page_bytes。
        Ok(unsafe {
            Page::init_in_place_(mem, NonNull::from(self), Page::CELLS as CellIndex)
        })
    }

    /// 归还一个已摘链、已 drain 的 Page。
    fn dealloc_page_(&self, page: NonNull<Page>) {
        // SAFETY: page 由 create_page_ 以同一布局申请，且调用方保证已摘链、无页锁持有者。
        let layout = unsafe { Layout::from_size_align_unchecked(
                self.page_bytes_,
                mem::align_of::<Page>(),
        ) };
        let base = self.self_ptr_();
        // SAFETY: 同上；page/layout 来自本分配器，A 先于 Core 存续。
        // `backing_lock_` 串行化所有对 A 的调用，因此 A 只需要 `Send`。
        {
            let _guard = self.backing_lock_.lock_();
            unsafe { (self.backing_dealloc_)(base, page.cast::<u8>(), layout) };
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

    /// 单块是否可能放进一个空页（只按数据 cell 做保守下界检查）。
    fn fits_page_(&self, layout: Layout) -> bool {
        let max_data = (self.page_cells_ as usize).saturating_sub(Page::CELLS + 1);
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
            // 空 tail 页放不下时，同大小的新页也不会容纳这个请求；保持空页复用，
            // 直接返回错误，避免把唯一空 tail 变成非 tail。
            let was_empty = unsafe { page_ref.empty_locked_() };
            drop(page_guard);
            if was_empty {
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
    /// 规则：只保留 `tail_` 这一个空页供复用；其它空页摘链并归还。是否为空由
    /// `PageState.returned_ == used_` 判断，不再需要 `PageList` 里的 bool。
    ///
    /// 锁顺序为链表锁 → 页锁（可能再锁一次 tail 页）。分配路径只会在持有链表锁时
    /// 用 `try_lock` 取页锁；归还路径在调用本函数前已经释放页锁，因此不会形成
    /// ABBA 死锁。
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
        if inner.tail_ == Some(page) {
            // 已经是 tail 空页：保留并复用，不归还。
            drop(page_guard);
            drop(list_guard);
            return;
        }
        // SAFETY: 链表锁保证 links_ 没有并发写。
        let linked = inner.head_ == Some(page)
            || unsafe { page_ref.prev_() }.is_some()
            || unsafe { page_ref.next_() }.is_some();
        if !linked {
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
/// # 生命周期
///
/// 只为 `&BumpListAlloc` 实现 [`Allocator`]，因此交给容器的分配器类型携带借期：
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
/// `BumpListAlloc` 本身不实现 `Clone`；需要共享时使用 `&` 或 `Arc`。
///
/// # 线程安全
///
/// 类型参数要求 `A: Send`。所有对底层分配器 `A` 的调用（页申请 / 页归还）都由
/// `BumpListCore` 的 `backing_lock_` 串行化，因此 `BumpListAlloc<A>` 是
/// `Send + Sync`，不要求 `A: Sync`。`A: !Send` 的分配器不能用于本类型；例如需要
/// trait object 时，应使用 `&(dyn Allocator + Sync)` 而不是 `&dyn Allocator`。
///
/// # 地址稳定性
///
/// `BumpListCore` 直接内联在 `BumpListAlloc` 里；Page 的 owner 指向它。Core 保存一个
/// 指向整个分配器的原子指针，单态化函数用 `offset_of!` 从它取出 `alloc_`，因此不需要
/// `Box<A>`；该指针在每次分配 / 归还 / 析构前刷新。
///
/// **地址稳定性**：Page 的 owner 指向内联 Core，因此创建第一个 Page 之后
/// `BumpListAlloc` 不能被移动；需要移动请先 `Box::new` / `pin` 固定地址再分配。
pub struct BumpListAlloc<A: Allocator + Send = Global> {
    /// 非泛型核心；Page 的 owner 指向它。声明在 `alloc_` 之前，因此先于 A 析构。
    pub(crate) core_: BumpListCore,

    /// 底层分配器；Core 的单态化函数从 Core 地址反推它。
    alloc_: A,
}

// SAFETY: `PageList` 与 Page 链表中的 `NonNull` 指针由 `list_lock_` 保护；Page 的
// `state_` 由页锁保护；Page 地址在分配后不再移动。`A` 的 `Send` 允许整个 allocator
// 转移线程。
unsafe impl<A: Allocator + Send> Send for BumpListAlloc<A> {}

// SAFETY: 共享 `&BumpListAlloc` 时，链表由链表锁保护，页状态由页锁保护；
// 所有对底层分配器 `A` 的调用都由 `backing_lock_` 串行化，`A: Send` 足以让
// 这把锁把 A 的独占访问安全地借给任意线程。Page 的 `links_` / `state_` 使用
// `UnsafeCell`，分别由两把锁保护不相交字段。
unsafe impl<A: Allocator + Send> Sync for BumpListAlloc<A> {}

impl<A: Allocator + Send> Drop for BumpListAlloc<A> {
    fn drop(&mut self) {
        // 刷新自指针到当前地址；随后 `core_` 字段先析构并使用它归还所有页。
        self.core_.set_self_ptr_(self);
    }
}

impl<A: Allocator + Send> BumpListAlloc<A> {
    /// 使用指定的底层分配器与页大小创建一个空 allocator。
    ///
    /// 底层分配器 `A` 必须满足 `Send`。构造时不申请 Page，第一次分配才会申请。
    ///
    /// # Errors
    ///
    /// 页大小在当前平台不受支持，或页内连 Page 头、一个块头与一个数据 cell 都放不下
    /// 时返回 [`BumpListAllocError::UnsupportedPageSize`]。
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
        let core_ = BumpListCore::new_::<A>(page_size)?;
        Ok(Self {
            core_,
            alloc_: alloc,
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
        self.core_.page_size_()
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

impl Drop for BumpListCore {
    fn drop(&mut self) {
        let head = {
            let inner = self.inner_.get_mut();
            let head = inner.head_.take();
            inner.tail_ = None;
            head
        };
        let mut cursor = head;
        while let Some(page) = cursor {
            // SAFETY: Drop 独占 self；没有并发访问。
            let page_ref = unsafe { page.as_ref() };
            // SAFETY: 同上；链表 links_ 没有并发写。
            let next = unsafe { page_ref.next_() };

            let scan = page_ref.scan_chain_(page);
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

            self.dealloc_page_(page);
            cursor = next;
        }
    }
}

unsafe impl<A: Allocator + Send> Allocator for &BumpListAlloc<A> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.core_.set_self_ptr_(self);
        self.core_.allocate_(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.core_.set_self_ptr_(self);
        // SAFETY: 由 Allocator::deallocate 的调用方保证契约。
        unsafe { self.core_.deallocate_(ptr, layout) };
    }
}

unsafe impl<A: Allocator + Send> AllocatorClone for &BumpListAlloc<A>
{}

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

    /// 测试直接调用 Core 时先刷新整分配器自指针。
    fn set_self_(bump: &BumpListAlloc<Global>) {
        bump.core_.set_self_ptr_(bump);
    }

    fn count_pages_(bump: &BumpListAlloc<Global>) -> usize {
        // SAFETY: 单线程测试，没有并发访问。
        let inner = unsafe { &*bump.core_.inner_.get() };
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
        let inner = unsafe { &*bump.core_.inner_.get() };
        let mut cursor = inner.head_;
        let mut total = 0u16;
        while let Some(page) = cursor {
            // SAFETY: 单线程测试，Page 有效。
            let page_ref = unsafe { page.as_ref() };
            let (live, _freed) = page_ref
                .scan_chain_(page)
                .expect("链应当有效");
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

    /// 测试目标：第一次分配创建一个 Page；释放最后一个块后保留这一个空 Page 并复用。
    /// Test goal: The first allocation creates one page; releasing the last block keeps
    /// that single empty page for reuse.
    /// - 手段 / Method: 4 KiB 页上分配一个 8 字节块并释放，再分配一次。
    /// - 判定 / Assertion: 页数始终为 1，live 数 1→0→1；块头反查出的页首等于链头。
    #[test]
    fn first_allocation_creates_and_frees_page() {
        let bump = BumpListAlloc::new(PageSize::P4K).expect("4 KiB 页");
        assert_eq!(count_pages_(&bump), 0);

        let layout = Layout::from_size_align(8, 8).expect("布局合法");
        let block = {
            set_self_(&bump);
            bump.core_.allocate_(layout)
        }
        .expect("分配成功");
        let ptr = block_ptr_(block);
        assert_eq!(count_pages_(&bump), 1);
        assert_eq!(count_live_(&bump), 1);

        // SAFETY: 单线程测试，链头有效。
        let head = unsafe { (*bump.core_.inner_.get()).head_ }.expect("有页");
        assert_eq!(page_of_block_(ptr), head);

        // SAFETY: 指针来自同一 allocator 与 layout。
        set_self_(&bump);
        unsafe { bump.core_.deallocate_(ptr, layout) };
        // 唯一空 Page 被保留为 tail，不归还；它应该可以直接复用。
        assert_eq!(count_pages_(&bump), 1);
        assert_eq!(count_live_(&bump), 0);

        let block_again = {
            set_self_(&bump);
            bump.core_.allocate_(layout)
        }
        .expect("空页应当可复用");
        assert_eq!(count_pages_(&bump), 1);
        assert_eq!(count_live_(&bump), 1);
        assert_eq!(page_of_block_(block_ptr_(block_again)), head);

        // SAFETY: 指针来自同一 allocator 与 layout。
        set_self_(&bump);
        unsafe { bump.core_.deallocate_(block_ptr_(block_again), layout) };
        assert_eq!(count_pages_(&bump), 1);
        assert_eq!(count_live_(&bump), 0);
    }

    /// 测试目标：多页分配、乱序释放时页不会提前消失，全部释放后链表为空。
    /// Test goal: With multiple pages and out-of-order release, pages stay alive until
    /// all their blocks are freed; the list is empty at the end.
    /// - 手段 / Method: 1 KiB 页上分配 12 个 200 字节块，先释放偶数下标，再释放奇数。
    /// - 判定 / Assertion: 页数至少为 2；中间 live=6；最后 live=0，且只保留一个空 tail
    ///   页。
    #[test]
    fn multiple_pages_and_out_of_order_release() {
        let bump = BumpListAlloc::new(PageSize::P1K).expect("1 KiB 页");
        let layout = Layout::from_size_align(200, 8).expect("布局合法");
        let mut blocks = Vec::new();

        for _ in 0..12 {
            let block = {
                set_self_(&bump);
                bump.core_.allocate_(layout)
            }
            .expect("块应当放得下");
            blocks.push(block_ptr_(block));
        }
        assert!(count_pages_(&bump) >= 2);
        assert_eq!(count_live_(&bump), 12);

        for index in (0..12).step_by(2) {
            // SAFETY: 指针来自同一 allocator 与 layout。
            set_self_(&bump);
            unsafe { bump.core_.deallocate_(blocks[index], layout) };
        }
        assert_eq!(count_live_(&bump), 6);

        for index in (1..12).step_by(2) {
            // SAFETY: 指针来自同一 allocator 与 layout。
            set_self_(&bump);
            unsafe { bump.core_.deallocate_(blocks[index], layout) };
        }
        assert_eq!(count_live_(&bump), 0);
        // 多页都空掉后只保留 tail 这一个空页，其余归还。
        assert_eq!(count_pages_(&bump), 1);
    }

    /// 测试目标：空 tail 页因对齐放不下时直接返回 `AllocError`，不追加新页，也不破坏
    /// 唯一空页的复用状态。
    /// Test goal: When the empty tail page cannot satisfy an alignment, return
    /// `AllocError` without appending a new page or losing the cached empty page.
    /// - 手段 / Method: 1 KiB 页先分配/释放一个块得到空 tail，再请求 1 MiB 对齐的块。
    /// - 判定 / Assertion: 请求失败；页数仍为 1，live 仍为 0，空页仍可复用。
    #[test]
    fn empty_tail_alignment_failure_does_not_append() {
        let bump = BumpListAlloc::new(PageSize::P1K).expect("1 KiB 页");
        let layout = Layout::from_size_align(8, 8).expect("布局合法");
        let block = {
            set_self_(&bump);
            bump.core_.allocate_(layout)
        }
        .expect("分配成功");
        let ptr = block_ptr_(block);
        // SAFETY: 指针来自同一 allocator 与 layout。
        set_self_(&bump);
        unsafe { bump.core_.deallocate_(ptr, layout) };
        assert_eq!(count_pages_(&bump), 1);
        assert_eq!(count_live_(&bump), 0);

        let over_aligned = Layout::from_size_align(8, 1 << 20).expect("布局合法");
        assert!(bump.core_.allocate_(over_aligned).is_err());
        assert_eq!(count_pages_(&bump), 1);
        assert_eq!(count_live_(&bump), 0);

        // 空页仍然可用。
        let again = {
            set_self_(&bump);
            bump.core_.allocate_(layout)
        }
        .expect("空页应当仍可复用");
        assert_eq!(count_live_(&bump), 1);
        // SAFETY: 指针来自同一 allocator 与 layout。
        unsafe { bump.core_.deallocate_(block_ptr_(again), layout) };
        assert_eq!(count_pages_(&bump), 1);
    }

    /// 测试目标：超过页容量的请求直接返回 `AllocError`，不创建 Page。
    /// Test goal: A request larger than the page returns `AllocError` without creating a
    /// page.
    /// - 手段 / Method: 1 KiB 页上请求 8 KiB。
    /// - 判定 / Assertion: `allocate_` 返回错误，页数为 0。
    #[test]
    fn oversized_request_is_rejected_without_page() {
        let bump = BumpListAlloc::new(PageSize::P1K).expect("1 KiB 页");
        let layout = Layout::from_size_align(8 * 1024, 8).expect("布局合法");
        assert!(
            {
                set_self_(&bump);
                bump.core_.allocate_(layout)
            }
            .is_err()
        );
        assert_eq!(count_pages_(&bump), 0);
        assert_eq!(count_live_(&bump), 0);
    }

    /// 测试目标：零尺寸分配不创建 Page，释放也不改变链表。
    /// Test goal: A zero-sized allocation creates no page and its deallocation leaves
    /// the list unchanged.
    /// - 手段 / Method: 1 KiB 页上分配/释放 `Layout::new::<()>()`。
    /// - 判定 / Assertion: 页数始终为 0，返回地址按对齐。
    #[test]
    fn zero_sized_allocation_has_no_page() {
        let bump = BumpListAlloc::new(PageSize::P1K).expect("1 KiB 页");
        let layout = Layout::new::<()>();
        let block = {
            set_self_(&bump);
            bump.core_.allocate_(layout)
        }
        .expect("零尺寸分配成功");
        let ptr = block_ptr_(block);
        assert_eq!(count_pages_(&bump), 0);
        assert_eq!(ptr.as_ptr() as usize % layout.align(), 0);

        // SAFETY: 同一 allocator 与 layout 的零尺寸分配。
        set_self_(&bump);
        unsafe { bump.core_.deallocate_(ptr, layout) };
        assert_eq!(count_pages_(&bump), 0);
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
            assert_eq!(bump.core_.page_cells_, 1u16 << 15);
            let layout = Layout::from_size_align(8, 8).expect("布局合法");
            let block = {
                set_self_(&bump);
                bump.core_.allocate_(layout)
            }
            .expect("分配成功");
            let ptr = block_ptr_(block);
            assert_eq!(count_live_(&bump), 1);
            // SAFETY: 指针来自同一 allocator 与 layout。
            set_self_(&bump);
            unsafe { bump.core_.deallocate_(ptr, layout) };
            // 256 KiB 页也遵循“保留一个空 tail 页”的策略。
            assert_eq!(count_pages_(&bump), 1);
            assert_eq!(count_live_(&bump), 0);
        }

        #[cfg(target_pointer_width = "32")]
        {
            assert!(BumpListAlloc::new(PageSize::P256K).is_err());
        }
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
