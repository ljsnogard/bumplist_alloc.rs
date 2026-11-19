//! 把「稳定 `Deref<Target = BumpListAlloc>` 的句柄」适配成 [`Allocator`]。
//!
//! [`BumpListAlloc`] 在创建第一个 Page 之后不能移动，因此本 crate 只为
//! `&BumpListAlloc` 实现 [`Allocator`]。`Arc<BumpListAlloc, _>` / `Rc<...>` /
//! `Box<...>` 拥有稳定的目标地址，却不能直接实现 [`Allocator`]：`Allocator` 是
//! core 的 trait，而 `Arc` / `Rc` / `Box` 都不是本 crate 的本地类型，孤儿规则不允许
//! 为它们写 impl。
//!
//! [`BumpListAllocRef`] 是本地 newtype：它只转发 [`Allocator`] / [`AllocatorClone`]，
//! 在 `denco_arena` 特性下再转发 `TrArena` / `TrArenaBytes` / `TrAllocSource`。
//! 它不是新的 arena，也没有新的页管理；真正的分配仍然由内部的 [`BumpListAlloc`] 完成。
//!
//! # `alloc_ref_`: a local adapter for stable `Deref` handles
//!
//! `BumpListAlloc` cannot move after its first page, so `Allocator` is implemented for
//! `&BumpListAlloc`. `Arc` / `Rc` / `Box` own a stable target but are foreign types, so the
//! orphan rule forbids implementing `Allocator` for them directly. `BumpListAllocRef` is the
//! local newtype that forwards the allocator (and denco arena) traits.

use alloc::alloc::Global;
use core::{
    alloc::{AllocError, Allocator, AllocatorClone, Layout},
    fmt,
    marker::PhantomData,
    ops::Deref,
    ptr::NonNull,
};

use crate::BumpListAlloc;

/// 把稳定 `Deref` 句柄适配成 [`Allocator`] / [`AllocatorClone`] 的薄包装。
///
/// 典型用法是拥有 per-request / per-session arena：
///
/// ```ignore
/// let root = Box::new(BumpListAlloc::new(PageSize::P2K)?);
/// let inner = BumpListAlloc::new(PageSize::P2K)?;
/// let handle = BumpListAllocRef::new(Arc::new_in(inner, &*root));
/// let mut headers: Headers<BumpListAllocRef<Arc<BumpListAlloc, &BumpListAlloc>>> =
///     Headers::new_in(handle.clone());
/// ```
///
/// 包装本身不分配、不持有 Page；它只是把 `&*inner` 的 [`Allocator`] 实现投影到
/// 句柄类型上。`Clone` 克隆的是句柄，不是 arena；所有克隆共享同一个
/// [`BumpListAlloc`]。
///
/// # Examples
///
/// ```
/// use bumplist_alloc::{BumpListAlloc, BumpListAllocRef, PageSize};
///
/// let bump = BumpListAlloc::new(PageSize::P2K)?;
/// let handle = BumpListAllocRef::new(&bump);
/// let mut values: Vec<u32, _> = Vec::new_in(handle);
/// values.extend_from_slice(&[1, 2, 3]);
/// assert_eq!(&values[..], &[1, 2, 3]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct BumpListAllocRef<D, A = Global>
where
    D: Deref<Target = BumpListAlloc<A>>,
    A: Allocator + Send,
{
    inner_: D,
    _use_a_: PhantomData<fn() -> A>,
}

impl<D, A> BumpListAllocRef<D, A>
where
    D: Deref<Target = BumpListAlloc<A>>,
    A: Allocator + Send,
{
    /// 由稳定句柄构造。
    ///
    /// 安全性由 [`TrStableDeref`] 的实现者承担：`D` 必须保证目标地址稳定且 clone 等价。
    pub const fn new(inner: D) -> Self {
        BumpListAllocRef {
            inner_: inner,
            _use_a_: PhantomData,
        }
    }

    /// 借用内部的稳定句柄。
    pub const fn inner(&self) -> &D {
        &self.inner_
    }

    /// 取回内部的稳定句柄。
    pub fn into_inner(self) -> D {
        self.inner_
    }
}

impl<D, A> Clone for BumpListAllocRef<D, A>
where
    D: Clone + Deref<Target = BumpListAlloc<A>>,
    A: Allocator + Send,
{
    fn clone(&self) -> Self {
        BumpListAllocRef {
            inner_: self.inner_.clone(),
            _use_a_: PhantomData,
        }
    }
}

impl<D, A> Deref for BumpListAllocRef<D, A>
where
    D: Deref<Target = BumpListAlloc<A>>,
    A: Allocator + Send,
{
    type Target = D::Target;

    fn deref(&self) -> &Self::Target {
        self.inner_.deref()
    }
}

impl<D, A> fmt::Debug for BumpListAllocRef<D, A>
where
    D: Deref<Target = BumpListAlloc<A>> + fmt::Debug,
    A: Allocator + Send,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("BumpListAllocRef")
            .field(&self.inner_)
            .finish()
    }
}

// SAFETY: 所有方法都转发给 `&*self.inner_` 的 `Allocator` 实现；
// `TrStableDeref` 保证该 `BumpListAlloc` 在句柄存活期间地址稳定。
unsafe impl<D, A> Allocator for BumpListAllocRef<D, A>
where
    D: Deref<Target = BumpListAlloc<A>>,
    A: Allocator + Send,
{
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let bump: &BumpListAlloc<A> = self.inner_.deref();
        <&BumpListAlloc<A> as Allocator>::allocate(&bump, layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let bump: &BumpListAlloc<A> = self.inner_.deref();
        // SAFETY: 调用方保证 ptr/layout 来自本分配器；本实现转发给同一个底层分配器。
        unsafe { <&BumpListAlloc<A> as Allocator>::deallocate(&bump, ptr, layout) }
    }
}

// SAFETY: `D: Clone` 的克隆必须指向同一个稳定目标（见 `TrStableDeref` 的安全契约）；
// 克隆出来的包装共享同一个底层 arena，因此任意 clone 都能释放另一 clone 分配的块。
unsafe impl<D, A> AllocatorClone for BumpListAllocRef<D, A>
where
    D: Clone + Deref<Target = BumpListAlloc<A>>,
    A: AllocatorClone + Send,
{}

#[cfg(feature = "denco_arena")]
mod denco_impls_ {
    //! `denco_arena` 特性下的 arena 转发。

    use abs_denco::de::{TrAllocSource, TrArena, TrArenaBytes};

    use super::*;

    impl<'alloc, D, A> TrArena<'alloc> for BumpListAllocRef<D, A>
    where
        D: 'alloc + Deref<Target = BumpListAlloc<A>>,
        A: 'alloc + Allocator + Send,
    {
        type Error = <BumpListAlloc<A> as TrArena<'alloc>>::Error;

        type Ref<T: 'alloc> = <BumpListAlloc<A> as TrArena<'alloc>>::Ref<T>
        where
            Self: 'alloc;

        type AllocValueAsync<'a, T> = <BumpListAlloc<A> as TrArena<'alloc>>::AllocValueAsync<'a, T>
        where
            'alloc: 'a,
            T: 'alloc + 'a,
            Self: 'alloc + 'a;

        fn alloc_value_async<'a, T>(&'alloc self, value: T) -> Self::AllocValueAsync<'a, T>
        where
            'alloc: 'a,
            T: 'alloc + 'a,
            Self: 'alloc + 'a,
        {
            let bump: &'alloc BumpListAlloc<A> = self.inner_.deref();
            TrArena::alloc_value_async(bump, value)
        }
    }

    impl<'alloc, D, A> TrArenaBytes<'alloc> for BumpListAllocRef<D, A>
    where
        D: 'alloc + Deref<Target = BumpListAlloc<A>>,
        A: 'alloc + Allocator + Send,
    {
        type AllocBytesAsync<'a> = <BumpListAlloc<A> as TrArenaBytes<'alloc>>::AllocBytesAsync<'a>
        where
            'alloc: 'a,
            Self: 'alloc + 'a;

        fn alloc_bytes_async<'a>(&'alloc self, src: &'a [u8]) -> Self::AllocBytesAsync<'a>
        where
            'alloc: 'a,
            Self: 'alloc + 'a,
        {
            let bump: &'alloc BumpListAlloc<A> = self.inner_.deref();
            TrArenaBytes::alloc_bytes_async(bump, src)
        }
    }

    impl<'alloc, D, A> TrAllocSource<'alloc, BumpListAllocRef<D, A>> for BumpListAllocRef<D, A>
    where
        D: 'alloc + Clone + Deref<Target = BumpListAlloc<A>>,
        A: 'alloc + Allocator + Send,
    {
        fn alloc_from(arena: &'alloc BumpListAllocRef<D, A>) -> Self {
            arena.clone()
        }
    }
}

#[cfg(test)]
mod tests_ {
    use alloc::{rc::Rc, sync::Arc, vec::Vec};

    use super::*;
    use crate::PageSize;

    /// 测试 `BumpListAllocRef<&BumpListAlloc>` 可以作为容器 allocator。
    /// Test that `BumpListAllocRef<&BumpListAlloc>` works as a container allocator.
    ///
    /// - 手段 / Means: 用 `BumpListAllocRef::new(&bump)` 构造句柄，创建带该 allocator 的
    ///   `Vec` 并写入三个值。
    /// - 判定 / Verdict: `Vec` 的内容与写入值完全一致。
    #[test]
    fn alloc_ref_borrowed_supports_containers_() {
        let bump = BumpListAlloc::new(PageSize::P2K).expect("P2K 应当受支持");
        let handle = BumpListAllocRef::new(&bump);
        let mut values: Vec<u32, BumpListAllocRef<&BumpListAlloc>> = Vec::new_in(handle);
        values.extend_from_slice(&[1u32, 2u32, 3u32]);
        assert_eq!(&values[..], &[1u32, 2u32, 3u32]);
    }

    /// 测试 `Arc` / `Rc` 句柄的 `Clone` 仍共享同一个 arena。
    /// Test that cloned `Arc` / `Rc` handles still share the same arena.
    ///
    /// - 手段 / Means: 分别用 `Arc<BumpListAlloc>` 与 `Rc<BumpListAlloc>` 构造句柄，
    ///   克隆句柄后创建 `Vec` 并写入数据。
    /// - 判定 / Verdict: 两种容器都保留写入的数据，说明 clone 后仍能正确使用同一 arena。
    #[test]
    fn alloc_ref_arc_and_rc_clone_share_arena_() {
        let arc_handle = BumpListAllocRef::new(Arc::new(
            BumpListAlloc::new(PageSize::P2K).expect("P2K 应当受支持"),
        ));
        let arc_clone = arc_handle.clone();
        let mut arc_values: Vec<u32, _> = Vec::new_in(arc_handle);
        arc_values.extend_from_slice(&[4u32, 5u32]);
        drop(arc_clone);
        assert_eq!(&arc_values[..], &[4u32, 5u32]);

        let rc_handle = BumpListAllocRef::new(Rc::new(
            BumpListAlloc::new(PageSize::P2K).expect("P2K 应当受支持"),
        ));
        let rc_clone = rc_handle.clone();
        let mut rc_values: Vec<u32, _> = Vec::new_in(rc_handle);
        rc_values.extend_from_slice(&[6u32, 7u32]);
        drop(rc_clone);
        assert_eq!(&rc_values[..], &[6u32, 7u32]);
    }
    /// 测试 denco 的 `TrAllocSource` / `TrArena` 转发到内部 `BumpListAlloc`。
    /// Test that denco's `TrAllocSource` / `TrArena` forward to the inner `BumpListAlloc`.
    ///
    /// - 手段 / Means: 用 `BumpListAllocRef` 克隆出 `TrAllocSource`，再通过 `TrArena`
    ///   分配一个 `u32` 并立即驱动 future 完成。
    /// - 判定 / Verdict: 克隆句柄可用；分配回来的 `Box` deref 后等于原值。
    #[cfg(feature = "denco_arena")]
    #[test]
    fn alloc_ref_denco_traits_forward_() {
        use core::{
            future::Future,
            pin::pin,
            task::{Context, Poll, Waker},
        };

        use abs_denco::de::{TrAllocSource, TrArena};

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

        let bump = BumpListAlloc::new(PageSize::P2K).expect("P2K 应当受支持");
        let handle = BumpListAllocRef::new(&bump);
        let copy = <BumpListAllocRef<&BumpListAlloc> as TrAllocSource<
            '_,
            BumpListAllocRef<&BumpListAlloc>,
        >>::alloc_from(&handle);
        let value =
            block_on_(TrArena::alloc_value_async(&copy, 42u32)).expect("arena 分配应当成功");
        assert_eq!(*value, 42u32);
    }
}
