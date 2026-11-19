//! 第一页上的分配器元数据。
//!
//! # 布局
//!
//! ```text
//! 第一页内存（page_size 字节，按 align_of::<InitPage<A>>() 对齐）
//! ├── Page                     // 页头；位于偏移 0
//! ├── ... padding ...
//! ├── BumpListCore             // 非泛型核心；Page.owner_ 指向它
//! ├── MaybeUninit<A>           // 底层分配器；构造完成后才初始化
//! └── ... 用户可分配区域 ...
//! ```
//!
//! `InitPage<A>` 描述前三个字段。它只存在于第一页，并且从构造到析构都不会移动：
//! [`BumpListAlloc`](crate::BumpListAlloc) 只保存 `NonNull<InitPage<A>>`，移动句柄
//! 不会移动这里的内存。普通页的页头仍然只有 [`Page`]；两类页的“首个可分配 cell”
//! 分别由各自的 `PageState.base_` 记录。
//!
//! # 自引用与析构
//!
//! `BumpListCore` 的 backing 字段是指向 `MaybeUninit<A>` 的原始 fat 指针。它只能在
//! `A` 写入之后建立，因此本模块按“分配内存 → 写 Page → 写 A → 写 Core”的顺序在原地
//! 初始化。
//!
//! 析构顺序更关键：先用 `A` 归还所有**非第一页**的 Page，再销毁 Core，然后把 `A`
//! 从第一页中移出，最后用移出的 `A` 归还第一页本身。这样 `A` 不会在被自己释放的内存
//! 上继续存活：归还第一页时，`A` 已经在栈上。

use core::{
    alloc::{Allocator, Layout},
    mem::{self, MaybeUninit},
    ptr::{self, NonNull},
};

use crate::{
    alloc_::BumpListCore,
    error_::BumpListAllocError,
    page_::{CellIndex, PTR, Page},
    page_size_::PageSize,
};

/// 第一页的元数据布局。
///
/// 字段顺序由 `#[repr(C)]` 固定，`Page` 必须在偏移 0，因此
/// `NonNull<InitPage<A>>` 可以同时当作 `NonNull<Page>` 使用。`alloc_` 在构造完成后
/// 才写入；只有 [`InitPage::drop_in_place_`] 会把它移出。
#[repr(C)]
pub(crate) struct InitPage<A: Allocator + Send> {
    /// 第一页的 Page 头。
    pub(crate) page_: Page,
    /// 非泛型核心；第一页的 `Page.owner_` 指向它。
    pub(crate) core_: BumpListCore,
    /// 底层分配器；构造完成后才初始化。
    alloc_: MaybeUninit<A>,
}

// `Page` 必须是第一个字段：`Page::owner_` 与所有页内偏移都以 Page 首地址为基准。
const _: () = {
    assert!(mem::offset_of!(InitPage<alloc::alloc::Global>, page_) == 0);
    assert!(
        mem::offset_of!(InitPage<alloc::alloc::Global>, core_)
            >= mem::size_of::<crate::page_::Page>()
    );
};

impl<A: Allocator + Send> InitPage<A> {
    /// `InitPage` 占用的 cell 数，也就是第一页的首个可分配 cell 下标。
    pub(crate) const CELLS: usize = mem::size_of::<Self>().div_ceil(PTR);

    /// 申请第一页并在原地写入 `Page`、`A` 与 [`BumpListCore`]。
    ///
    /// 第一页使用 `page_size` 字节、按 `align_of::<InitPage<A>>()` 对齐的布局申请；
    /// 它同时是分配器的元数据页和第一个可分配页。调用成功后，返回值由
    /// [`BumpListAlloc`](crate::BumpListAlloc) 独占。
    ///
    /// # Errors
    ///
    /// - 页大小受平台支持，但容纳不下 `Page`、Core、`A` 与一个最小分配块时返回
    ///   [`BumpListAllocError::PageSizeTooSmallForAllocator`]；
    /// - 底层分配器无法提供第一页时返回
    ///   [`BumpListAllocError::InitPageAllocationFailed`]；
    /// - 页大小本身不受平台支持时返回
    ///   [`BumpListAllocError::UnsupportedPageSize`]。
    pub(crate) fn new_(alloc: A, page_size: PageSize) -> Result<NonNull<Self>, BumpListAllocError> {
        let page_bytes = page_size.bytes();
        let page_cells = page_size.cells();
        if !page_size.is_supported() || page_cells < Page::CELLS + 2 {
            return Err(BumpListAllocError::UnsupportedPageSize { page_size });
        }

        let init_cells = Self::CELLS;
        let Some(required_cells) = init_cells.checked_add(2) else {
            return Err(BumpListAllocError::PageSizeTooSmallForAllocator { page_size });
        };
        if required_cells > page_cells {
            return Err(BumpListAllocError::PageSizeTooSmallForAllocator { page_size });
        }

        // 第一页可能比普通页要求更高的对齐（由 `A` 决定），因此布局的 align 取
        // `InitPage<A>` 的对齐；大小仍严格等于配置的页大小，后续按同一布局归还。
        let layout = Layout::from_size_align(page_bytes, mem::align_of::<Self>())
            .map_err(|_| BumpListAllocError::InitPageAllocationFailed { page_size })?;
        let mem = alloc
            .allocate(layout)
            .map_err(|_| BumpListAllocError::InitPageAllocationFailed { page_size })?;
        let page_mem = NonNull::slice_from_raw_parts(mem.cast::<u8>(), page_bytes);
        let base = page_mem.cast::<Self>();

        // SAFETY:
        // - `mem` 至少 page_bytes 字节，并按 `InitPage<A>` 对齐；
        // - 这块内存刚申请、独占，尚未共享；
        // - `Page::init_in_place_` 的断言已由上面的容量检查保证；
        // - 所有写入都落在 InitPage 的三个不重叠字段内，之后才返回给调用方。
        unsafe {
            let base_ptr = base.as_ptr();
            let core_ptr = NonNull::new_unchecked(ptr::addr_of_mut!((*base_ptr).core_));
            let page = Page::init_in_place_(page_mem, core_ptr, init_cells as CellIndex);
            debug_assert_eq!(page, base.cast::<Page>());

            // 先写入 A；backing 的 fat 指针只能在 A 初始化之后建立。
            let alloc_slot = ptr::addr_of_mut!((*base_ptr).alloc_).cast::<A>();
            alloc_slot.write(alloc);

            // `A` 可能带有非 `'static` 的借用，而 `BumpListCore` 是非泛型类型，无法
            // 在类型上表达这个生命周期。这里把引用生命周期擦除成 `'static` 再存成原始
            // fat 指针：Core 与 A 在同一个 InitPage 内，Core 总是先于 A 被销毁，且这个
            // 指针不会离开本 crate 的内部实现。原始指针不要求指针目标始终有效，因此
            // 析构时把 A 移出第一页也不会留下悬垂引用。
            let backing_ref: &dyn Allocator = &*alloc_slot;
            // SAFETY: 见上；生命周期仅用于存储，不会被用于延长 A 的实际存活期。
            let backing_ref: &'static dyn Allocator = mem::transmute(backing_ref);
            let backing = NonNull::from(backing_ref);

            let core = BumpListCore::new_(page_size, page, backing);
            ptr::addr_of_mut!((*base_ptr).core_).write(core);
        }

        Ok(base)
    }

    /// 原地销毁第一页：归还所有普通页、销毁 Core、移出 `A` 并归还第一页本身。
    ///
    /// # Safety
    ///
    /// `base` 必须是 [`InitPage::new_`] 返回、当前没有被任何引用借用、且仍然独占的
    /// 第一页指针。调用后该内存被归还给底层分配器，`base` 立即失效。
    pub(crate) unsafe fn drop_in_place_(base: NonNull<Self>) {
        let base_ptr = base.as_ptr();
        // SAFETY: 由调用方保证 base 独占且仍然有效；整个过程没有并发访问。
        unsafe {
            let core_ptr = ptr::addr_of_mut!((*base_ptr).core_);
            let page_bytes = (*core_ptr).page_size_().bytes();

            // 1. 用仍然存活、仍在第一页内的 A 归还所有普通页。
            (*core_ptr).free_non_root_pages_(base.cast::<Page>());

            // 2. 销毁 Core，结束 backing 对 A 的自引用；之后不再访问 core_ptr。
            ptr::drop_in_place(core_ptr);

            // 3. 把 A 移出第一页。此后再也不读取第一页里的 alloc_ 槽。
            let alloc = ptr::read(ptr::addr_of!((*base_ptr).alloc_).cast::<A>());

            // 4. A 已在栈上，可以安全地用它归还第一页本身。
            // SAFETY: page_bytes 与 align_of::<Self>() 与 new_ 中申请第一页的布局完全一致。
            let layout = Layout::from_size_align_unchecked(page_bytes, mem::align_of::<Self>());
            alloc.deallocate(base.cast::<u8>(), layout);

            // 5. 最后销毁移出的 A。
            drop(alloc);
        }
    }
}
