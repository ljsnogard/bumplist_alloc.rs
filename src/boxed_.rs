use core::{
    alloc::Layout,
    borrow::{Borrow, BorrowMut},
    cmp::Ordering,
    error::Error,
    fmt,
    future::Future,
    hash::{Hash, Hasher},
    iter::FusedIterator,
    marker::{PhantomData, Unpin, Unsize},
    mem,
    ops::{CoerceUnsized, Deref, DerefMut},
    pin::Pin,
    ptr::{self, NonNull},
    task::{Context, Poll},
};

use abs_mm::res_man::{TrBoxed, TrUnique};

use crate::page_::{CellIndex, Page};

/// [`BumpListAlloc`](crate::BumpListAlloc) 的独占句柄。
///
/// 它拥有 arena 里的一块 `T`：析构时运行 `T` 的析构函数，并通过块头反推所属 `Page`、
/// 调用 `Page::release_` 把块归还到页内。它不需要知道 `BumpListAlloc` 的类型，也不
/// 持有分配器引用。
///
/// # 能力
///
/// - `T: ?Sized`，支持 `Box<'a, Concrete>` 到 `Box<'a, dyn Trait>` 的 unsize 强制转换；
/// - `Deref` / `DerefMut` / `AsRef` / `AsMut` / `Borrow` / `BorrowMut`；
/// - `Debug` / `Display` / `Pointer` 与各进制格式化；
/// - `PartialEq` / `Eq` / `PartialOrd` / `Ord` / `Hash` / `Hasher`；
/// - 裸指针转换、[`Box::leak`]、[`Box::into_pin`]；
/// - `Iterator` 家族、`Future`、[`Error`] 的转发；
/// - `T: Clone`（`T: Sized`）时通过原 arena 的 `Page` owner 重新分配一份（零尺寸 `T`
///   复用对齐悬垂地址）；
/// - `T: Send` / `T: Sync` 时 `Box` 分别是 `Send` / `Sync`。arena 的底层分配器要求
///   `A: Send`，而 Core 的 backing lock 串行化所有对 `A` 的调用，因此
///   不需要 `A: Sync`。
///
/// # 限制
///
/// - 没有 `Box::new` / `Box::pin` / `Default` / `From<T>`：`Box` 只有块指针，不知道
///   应该使用哪个 arena；这些 API 需要调用方提供 arena。
///
/// # Examples
///
/// ```
/// use core::alloc::{Allocator, Layout};
/// use bumplist_alloc::{BumpListAlloc, Box, PageSize};
///
/// let arena = BumpListAlloc::new(PageSize::P4K).expect("4 KiB 页应当受支持");
/// let layout = Layout::new::<u32>();
/// let block = <&BumpListAlloc as Allocator>::allocate(&&arena, layout)
///     .expect("分配应当成功");
/// let raw = block.as_ptr().cast::<u32>();
/// unsafe { raw.write(7) };
/// let boxed = unsafe { Box::from_raw(raw) };
/// assert_eq!(*boxed, 7);
/// ```
pub struct Box<'a, T: ?Sized> {
    /// arena 里的值。
    ptr_: NonNull<T>,

    /// 只把 `'alloc` 带进类型，不持有分配器。
    _marker_: PhantomData<&'a T>,
}

impl<'a, T: ?Sized> Box<'a, T> {
    /// 从 arena 分配出的裸指针接管所有权。
    #[cfg(feature = "denco_arena")]
    pub(crate) fn from_raw_(raw: NonNull<T>) -> Self {
        Box {
            ptr_: raw,
            _marker_: PhantomData,
        }
    }

    /// 从 arena 块的裸指针接管所有权。
    ///
    /// # Safety
    ///
    /// `raw` 必须指向一个由某个 `BumpListAlloc` 分配、已经初始化、且当前没有其它
    /// `Box` 拥有的块。非零尺寸时块前必须紧邻一个 `BlockHeader`；`'a` 必须不长于该
    /// arena 的存活期。零尺寸 `T` 可以传入任意非空且按 `T` 对齐的地址。
    ///
    /// # Examples
    ///
    /// ```
    /// use core::alloc::{Allocator, Layout};
    /// use bumplist_alloc::{BumpListAlloc, Box, PageSize};
    ///
    /// let arena = BumpListAlloc::new(PageSize::P4K).expect("4 KiB 页应当受支持");
    /// let block = <&BumpListAlloc as Allocator>::allocate(&&arena, Layout::new::<u32>())
    ///     .expect("分配应当成功");
    /// let raw = block.as_ptr().cast::<u32>();
    /// unsafe { raw.write(7) };
    /// let boxed = unsafe { Box::from_raw(raw) };
    /// assert_eq!(*boxed, 7);
    /// ```
    #[must_use = "调用 `drop(Box::from_raw(ptr))` 才会析构并归还块"]
    pub unsafe fn from_raw(raw: *mut T) -> Self {
        // SAFETY: 由调用方保证 raw 非空。
        let ptr = unsafe { NonNull::new_unchecked(raw) };
        Self {
            ptr_: ptr,
            _marker_: PhantomData,
        }
    }

    /// 从 arena 块的 [`NonNull`] 指针接管所有权。
    ///
    /// # Safety
    ///
    /// 同 [`Box::from_raw`]。
    ///
    /// # Examples
    ///
    /// ```
    /// use core::alloc::{Allocator, Layout};
    /// use bumplist_alloc::{BumpListAlloc, Box, PageSize};
    ///
    /// let arena = BumpListAlloc::new(PageSize::P4K).expect("4 KiB 页应当受支持");
    /// let block = <&BumpListAlloc as Allocator>::allocate(&&arena, Layout::new::<u32>())
    ///     .expect("分配应当成功");
    /// let raw = block.as_ptr().cast::<u32>();
    /// unsafe { raw.write(7) };
    /// let boxed = unsafe { Box::from_raw(raw) };
    /// let ptr = Box::into_non_null(boxed);
    /// let boxed = unsafe { Box::from_non_null(ptr) };
    /// assert_eq!(*boxed, 7);
    /// ```
    #[must_use = "调用 `drop(Box::from_non_null(ptr))` 才会析构并归还块"]
    pub unsafe fn from_non_null(ptr: NonNull<T>) -> Self {
        Self {
            ptr_: ptr,
            _marker_: PhantomData,
        }
    }

    /// 消费 `Box`，返回底层裸指针。
    ///
    /// 调用后 `T` 不再被自动析构，块也不会被自动归还。
    ///
    /// # Examples
    ///
    /// ```
    /// use core::alloc::{Allocator, Layout};
    /// use bumplist_alloc::{BumpListAlloc, Box, PageSize};
    ///
    /// let arena = BumpListAlloc::new(PageSize::P4K).expect("4 KiB 页应当受支持");
    /// let raw = {
    ///     let block = <&BumpListAlloc as Allocator>::allocate(&&arena, Layout::new::<u32>())
    ///         .expect("分配应当成功");
    ///     let raw = block.as_ptr().cast::<u32>();
    ///     unsafe { raw.write(7) };
    ///     Box::into_raw(unsafe { Box::from_raw(raw) })
    /// };
    /// let boxed = unsafe { Box::from_raw(raw) };
    /// assert_eq!(*boxed, 7);
    /// ```
    #[must_use = "调用 `Box::from_raw` 或自行释放块，否则块会泄漏"]
    pub fn into_raw(self) -> *mut T {
        let ptr = self.ptr_;
        mem::forget(self);
        ptr.as_ptr()
    }

    /// 消费 `Box`，返回底层 [`NonNull`] 指针。
    ///
    /// 调用后 `T` 不再被自动析构，块也不会被自动归还。
    ///
    /// # Examples
    ///
    /// ```
    /// use core::alloc::{Allocator, Layout};
    /// use bumplist_alloc::{BumpListAlloc, Box, PageSize};
    ///
    /// let arena = BumpListAlloc::new(PageSize::P4K).expect("4 KiB 页应当受支持");
    /// let block = <&BumpListAlloc as Allocator>::allocate(&&arena, Layout::new::<u32>())
    ///     .expect("分配应当成功");
    /// let raw = block.as_ptr().cast::<u32>();
    /// unsafe { raw.write(7) };
    /// let boxed = unsafe { Box::from_raw(raw) };
    /// let ptr = Box::into_non_null(boxed);
    /// let boxed = unsafe { Box::from_non_null(ptr) };
    /// assert_eq!(*boxed, 7);
    /// ```
    #[must_use = "调用 `Box::from_non_null` 或自行释放块，否则块会泄漏"]
    pub fn into_non_null(self) -> NonNull<T> {
        let ptr = self.ptr_;
        mem::forget(self);
        ptr
    }

    /// 借出底层裸指针。
    ///
    /// # Examples
    ///
    /// ```
    /// use core::alloc::{Allocator, Layout};
    /// use bumplist_alloc::{BumpListAlloc, Box, PageSize};
    ///
    /// let arena = BumpListAlloc::new(PageSize::P4K).expect("4 KiB 页应当受支持");
    /// let block = <&BumpListAlloc as Allocator>::allocate(&&arena, Layout::new::<u32>())
    ///     .expect("分配应当成功");
    /// let raw = block.as_ptr().cast::<u32>();
    /// unsafe { raw.write(7) };
    /// let boxed = unsafe { Box::from_raw(raw) };
    /// assert_eq!(unsafe { *Box::as_ptr(&boxed) }, 7);
    /// ```
    pub fn as_ptr(this: &Self) -> *const T {
        this.ptr_.as_ptr()
    }

    /// 借出底层可变裸指针。
    ///
    /// # Examples
    ///
    /// ```
    /// use core::alloc::{Allocator, Layout};
    /// use bumplist_alloc::{BumpListAlloc, Box, PageSize};
    ///
    /// let arena = BumpListAlloc::new(PageSize::P4K).expect("4 KiB 页应当受支持");
    /// let block = <&BumpListAlloc as Allocator>::allocate(&&arena, Layout::new::<u32>())
    ///     .expect("分配应当成功");
    /// let raw = block.as_ptr().cast::<u32>();
    /// unsafe { raw.write(7) };
    /// let mut boxed = unsafe { Box::from_raw(raw) };
    /// unsafe { *Box::as_mut_ptr(&mut boxed) = 8 };
    /// assert_eq!(*boxed, 8);
    /// ```
    pub fn as_mut_ptr(this: &mut Self) -> *mut T {
        this.ptr_.as_ptr()
    }

    /// 借出底层 [`NonNull`] 指针。
    ///
    /// # Examples
    ///
    /// ```
    /// use core::alloc::{Allocator, Layout};
    /// use bumplist_alloc::{BumpListAlloc, Box, PageSize};
    ///
    /// let arena = BumpListAlloc::new(PageSize::P4K).expect("4 KiB 页应当受支持");
    /// let block = <&BumpListAlloc as Allocator>::allocate(&&arena, Layout::new::<u32>())
    ///     .expect("分配应当成功");
    /// let raw = block.as_ptr().cast::<u32>();
    /// unsafe { raw.write(7) };
    /// let mut boxed = unsafe { Box::from_raw(raw) };
    /// let ptr = Box::as_non_null(&mut boxed);
    /// unsafe { *ptr.as_ptr() = 9 };
    /// assert_eq!(*boxed, 9);
    /// ```
    pub fn as_non_null(this: &mut Self) -> NonNull<T> {
        this.ptr_
    }

    /// 泄漏 `Box`，返回 arena 中 `T` 的可变借用。
    ///
    /// 块不会被归还，会随 arena 一起释放。返回引用的生命周期是 `'a`，因此不能长于
    /// 创建本句柄时借用的 arena。
    ///
    /// # Examples
    ///
    /// ```
    /// use core::alloc::{Allocator, Layout};
    /// use bumplist_alloc::{BumpListAlloc, Box, PageSize};
    ///
    /// let arena = BumpListAlloc::new(PageSize::P4K).expect("4 KiB 页应当受支持");
    /// let block = <&BumpListAlloc as Allocator>::allocate(&&arena, Layout::new::<u32>())
    ///     .expect("分配应当成功");
    /// let raw = block.as_ptr().cast::<u32>();
    /// unsafe { raw.write(7) };
    /// let boxed = unsafe { Box::from_raw(raw) };
    /// let leaked = Box::leak(boxed);
    /// *leaked += 1;
    /// assert_eq!(*leaked, 8);
    /// ```
    pub fn leak(self) -> &'a mut T {
        let ptr = self.ptr_;
        mem::forget(self);
        // SAFETY: 块由本句柄独占且已初始化；'a 借住 arena，块在 'a 内保持有效。
        unsafe { &mut *ptr.as_ptr() }
    }

    /// 把 `Box` 转成 [`Pin<Box>`]。
    ///
    /// `Box` 本身无条件实现 [`Unpin`]，因此这个转换是安全的，等价于 [`Pin::new`]。
    ///
    /// # Examples
    ///
    /// ```
    /// use core::alloc::{Allocator, Layout};
    /// use bumplist_alloc::{BumpListAlloc, Box, PageSize};
    ///
    /// let arena = BumpListAlloc::new(PageSize::P4K).expect("4 KiB 页应当受支持");
    /// let block = <&BumpListAlloc as Allocator>::allocate(&&arena, Layout::new::<u32>())
    ///     .expect("分配应当成功");
    /// let raw = block.as_ptr().cast::<u32>();
    /// unsafe { raw.write(7) };
    /// let boxed = unsafe { Box::from_raw(raw) };
    /// let pinned = Box::into_pin(boxed);
    /// assert_eq!(*pinned, 7);
    /// ```
    pub fn into_pin(self) -> Pin<Self> {
        // SAFETY: 移动 `Box` 不会移动它指向的 `T`；`T: !Unpin` 时 `Pin<Box<T>>`
        // 也不会再暴露 `&mut T`，因此可以安全地把这个句柄直接钉住。
        unsafe { Pin::new_unchecked(self) }
    }

    /// 从用户块反推所属 `Page` 与块头在本页中的 cell 下标。
    ///
    /// # Safety
    ///
    /// `ptr_` 必须指向一个仍然活跃、非零尺寸的 arena 块。
    unsafe fn locate_page_(&self) -> (NonNull<Page>, CellIndex) {
        // SAFETY: 句柄独占一个活跃块；`T: ?Sized` 时先丢弃元数据，只保留数据地址。
        unsafe {
            Page::locate_block_(NonNull::new_unchecked(self.ptr_.as_ptr() as *mut u8))
        }
    }
}

impl<T: ?Sized> Deref for Box<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        // SAFETY: `ptr_` 指向一块已写入、由本句柄独占的 `T`。
        unsafe { self.ptr_.as_ref() }
    }
}

impl<T: ?Sized> DerefMut for Box<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: `ptr_` 指向一块已写入、由本句柄独占的 `T`。
        unsafe { self.ptr_.as_mut() }
    }
}

impl<'a, T: ?Sized> TrBoxed for Box<'a, T> {
    type Item = T;
}

impl<T: ?Sized> TrUnique for Box<'_, T> {}

// SAFETY: `Box` 只是指向 arena 内 `T` 的指针；移动 `Box` 不会移动 `T`。
impl<T: ?Sized> Unpin for Box<'_, T> {}

// SAFETY: `Box` 独占一个 `T`，跨线程转移时 `T: Send` 允许在新线程使用/析构它。
// 它所属的 `BumpListAlloc<A>` 在类型上要求 `A: Send`，且 Core 用 backing lock
// 串行化对 `A` 的访问；`'a` 保证 arena 在句柄存活期间有效，因此跨线程的块归还
// （以及 Clone 中的页申请）是安全的。
unsafe impl<T: ?Sized + Send> Send for Box<'_, T> {}

// SAFETY: 共享 `&Box` 只会交出 `&T`，因此要求 `T: Sync`；`Clone` 会在同一 arena 里
// 申请新块，其 `A` 访问同样由 backing lock 串行化。
unsafe impl<T: ?Sized + Sync> Sync for Box<'_, T> {}

impl<T: ?Sized> AsRef<T> for Box<'_, T> {
    fn as_ref(&self) -> &T {
        self
    }
}

impl<T: ?Sized> AsMut<T> for Box<'_, T> {
    fn as_mut(&mut self) -> &mut T {
        self
    }
}

impl<T: ?Sized> Borrow<T> for Box<'_, T> {
    fn borrow(&self) -> &T {
        self
    }
}

impl<T: ?Sized> BorrowMut<T> for Box<'_, T> {
    fn borrow_mut(&mut self) -> &mut T {
        self
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for Box<'_, T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, formatter)
    }
}

impl<T: ?Sized + fmt::Display> fmt::Display for Box<'_, T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, formatter)
    }
}

impl<T: ?Sized + fmt::Binary> fmt::Binary for Box<'_, T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Binary::fmt(&**self, formatter)
    }
}

impl<T: ?Sized + fmt::Octal> fmt::Octal for Box<'_, T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Octal::fmt(&**self, formatter)
    }
}

impl<T: ?Sized + fmt::LowerHex> fmt::LowerHex for Box<'_, T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(&**self, formatter)
    }
}

impl<T: ?Sized + fmt::UpperHex> fmt::UpperHex for Box<'_, T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::UpperHex::fmt(&**self, formatter)
    }
}

impl<T: ?Sized + fmt::LowerExp> fmt::LowerExp for Box<'_, T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerExp::fmt(&**self, formatter)
    }
}

impl<T: ?Sized + fmt::UpperExp> fmt::UpperExp for Box<'_, T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::UpperExp::fmt(&**self, formatter)
    }
}

impl<T: ?Sized> fmt::Pointer for Box<'_, T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ptr: *const T = &**self;
        fmt::Pointer::fmt(&ptr, formatter)
    }
}

impl<T: ?Sized + PartialEq> PartialEq for Box<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        PartialEq::eq(&**self, &**other)
    }
}

impl<T: ?Sized + Eq> Eq for Box<'_, T> {}

impl<T: ?Sized + PartialOrd> PartialOrd for Box<'_, T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
}

impl<T: ?Sized + Ord> Ord for Box<'_, T> {
    fn cmp(&self, other: &Self) -> Ordering {
        Ord::cmp(&**self, &**other)
    }
}

impl<T: ?Sized + Hash> Hash for Box<'_, T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Hash::hash(&**self, state);
    }
}

impl<T: ?Sized + Hasher> Hasher for Box<'_, T> {
    fn finish(&self) -> u64 {
        (**self).finish()
    }

    fn write(&mut self, bytes: &[u8]) {
        (**self).write(bytes);
    }

    fn write_u8(&mut self, value: u8) {
        (**self).write_u8(value);
    }

    fn write_u16(&mut self, value: u16) {
        (**self).write_u16(value);
    }

    fn write_u32(&mut self, value: u32) {
        (**self).write_u32(value);
    }

    fn write_u64(&mut self, value: u64) {
        (**self).write_u64(value);
    }

    fn write_u128(&mut self, value: u128) {
        (**self).write_u128(value);
    }

    fn write_usize(&mut self, value: usize) {
        (**self).write_usize(value);
    }

    fn write_i8(&mut self, value: i8) {
        (**self).write_i8(value);
    }

    fn write_i16(&mut self, value: i16) {
        (**self).write_i16(value);
    }

    fn write_i32(&mut self, value: i32) {
        (**self).write_i32(value);
    }

    fn write_i64(&mut self, value: i64) {
        (**self).write_i64(value);
    }

    fn write_i128(&mut self, value: i128) {
        (**self).write_i128(value);
    }

    fn write_isize(&mut self, value: isize) {
        (**self).write_isize(value);
    }
}

impl<I: ?Sized + Iterator> Iterator for Box<'_, I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        (**self).next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (**self).size_hint()
    }
}

impl<I: ?Sized + DoubleEndedIterator> DoubleEndedIterator for Box<'_, I> {
    fn next_back(&mut self) -> Option<Self::Item> {
        (**self).next_back()
    }
}

impl<I: ?Sized + ExactSizeIterator> ExactSizeIterator for Box<'_, I> {
    fn len(&self) -> usize {
        (**self).len()
    }
}

impl<I: ?Sized + FusedIterator> FusedIterator for Box<'_, I> {}

impl<F: ?Sized + Future + Unpin> Future for Box<'_, F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        F::poll(Pin::new(&mut **this), context)
    }
}

impl<E: Error> Error for Box<'_, E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        (**self).source()
    }
}

/// 通过 `Page` 的 owner（`BumpListCore`）在同一 arena 重新分配一份。
///
/// 零尺寸 `T` 没有 `Page`，因此复用原句柄的对齐悬垂地址；克隆出的值由新句柄的
/// `drop_in_place` 负责析构。
impl<'a, T: Clone> Clone for Box<'a, T> {
    fn clone(&self) -> Self {
        let value = T::clone(&**self);
        if mem::size_of::<T>() == 0 {
            // `value` 不能再在栈上析构一次；它对应的 Drop 由新 `Box` 完成。
            mem::forget(value);
            return Self {
                ptr_: self.ptr_,
                _marker_: PhantomData,
            };
        }

        let layout = Layout::new::<T>();
        // SAFETY: 当前句柄仍然拥有块，所以块所在 Page 有效；`'a` 借住 arena，
        // 期间 arena 不会被析构，Page 的 owner（Core）保持有效。
        let (page, _) = unsafe { self.locate_page_() };
        // SAFETY: 同上。
        let core = unsafe { page.as_ref().owner_ptr_() };
        // Core 位于 arena 第一页的堆内存中，地址不随 `BumpListAlloc` 句柄移动；
        // `'a` 期间 arena 不会被析构，因此 Core 与其 backing 分配器一直有效。
        let block = match unsafe { core.as_ref() }.allocate_(layout) {
            Result::Ok(block) => block,
            Result::Err(_) => alloc::alloc::handle_alloc_error(layout),
        };
        // SAFETY: `block` 非空，且按 `Layout::new::<T>()` 对齐。
        let ptr = unsafe { NonNull::new_unchecked(block.as_ptr() as *mut T) };
        // SAFETY: 新块是未初始化的 `T` 空间；这里只写入一次。
        unsafe { ptr.as_ptr().write(value) };
        Self {
            ptr_: ptr,
            _marker_: PhantomData,
        }
    }
}

impl<'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<Box<'a, U>> for Box<'a, T> {}

impl<'a, T: ?Sized> From<Box<'a, T>> for Pin<Box<'a, T>> {
    fn from(boxed: Box<'a, T>) -> Self {
        Box::into_pin(boxed)
    }
}

impl<T: ?Sized> Drop for Box<'_, T> {
    fn drop(&mut self) {
        // 尺寸必须在析构 `T` 之前计算；DST 的元数据来自 `ptr_`。
        // SAFETY: `ptr_` 指向一块已写入、由本句柄独占的 `T`。
        let size = unsafe { mem::size_of_val(self.ptr_.as_ref()) };
        // SAFETY: `ptr_` 来自 arena、按 `Layout::new::<T>()` 分配且只写入一次；
        // `drop_in_place` 之后这块内存不再被读取。
        unsafe { ptr::drop_in_place(self.ptr_.as_ptr()) };
        if size == 0 {
            // 零尺寸分配没有 BlockHeader，也没有 Page 需要归还。
            return;
        }
        // SAFETY: 非零尺寸块由 arena 分配，用户块前紧邻一个 `BlockHeader`；头里的
        // cell 下标可 O(1) 反推 Page 首地址。块此刻仍然活跃，Page 在归还前有效。
        unsafe {
            let (page, header_cell) = self.locate_page_();
            Page::release_page_(page, header_cell);
        }
    }
}
