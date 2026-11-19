# 引用计数、Clone 与共享 Core 锁位

- 状态：**已实施（2026-11-19 19:55）**
- 分支：`feat/move-safety`
- 相关代码：[`src/state_.rs`](../src/state_.rs)、[`src/alloc_.rs`](../src/alloc_.rs)
- 目标：让 `BumpListAlloc<A>` 支持 `Clone` 与 `AllocatorClone`，并把句柄引用计数与
  Core 的两把锁放进同一个 `AtomicUsize`。

## 背景

上一提交已经让 `BumpListAlloc<A>` 成为指向第一页 `InitPage<A>` 的单指针句柄。要支持
`Clone`，必须知道“还有多少句柄共享这块 arena”：最后一个句柄析构时才归还 Page 与第一页
内存。计数天然适合放在第一页的 `BumpListCore` 里，因为 Core 的生命周期恰好与 arena
一致。

Core 还已经有两把独立自旋锁：

- list lock：保护 Page 链表与 `Page.links_`；
- backing lock：串行化对底层分配器 `A` 的调用。

把它们做成两个独立的 `SpinLock` 字段，与“新增一个原子引用计数”是三次原子状态更新。
既然引用计数必须在同一个 Core 里，且现有的 `Page` 锁已经采用“原子字段中的一位作为锁”
的位标记技巧，本次把三者合并到一个 `AtomicUsize`：

```text
64 位示例：
  bit 63          list lock
  bit 62          backing lock
  bit 61..0       BumpListAlloc 句柄引用计数（初始为 1）
```

32 位平台同理，只是计数位变成 30 位，两把锁仍占最高两位。

## `CoreState`

新增 [`src/state_.rs`](../src/state_.rs)，提供 `CoreState`：

```rust
#[repr(transparent)]
pub(crate) struct CoreState {
    state_: AtomicUsize,
}
```

- `CoreState::new()`：计数 1，锁位 0；
- `clone_ref_()`：`try_update` 增加低位计数。低位数到达掩码上限时返回 `None`，随后
  panic（`Clone` 没有 fallible 版本）；
- `release_ref_()`：`fetch_sub(1, AcqRel)` 减少计数，返回是否减到 0；锁位在高位，
  减法不会借位到这里；
- `lock_list_()` / `lock_backing_()`：在同一个原子上用 CAS 置对应锁位；guard 析构时
  `fetch_and(!bit, Release)` 清位。

关键不变量：

1. 计数增加只在低位，且先检查是否达到 `REF_COUNT_MASK`，因此不会进位覆盖锁位；
2. 计数减少只减 1，且计数在操作期间至少为 1，不会借位；
3. 锁位的 CAS / fetch_and 保留所有其它位，因此计数与另一把锁不受影响；
4. 最后一个句柄 Drop 时，计数为 1 且不可能有其它线程正在操作 Core，所以锁位必须为 0
   （debug 断言）；
5. 释放锁的 `Release` 与获取锁的 `Acquire` 配对，维持原自旋锁的同步语义。

原来的 `src/lock_.rs`（`SpinLock` / `SpinGuard`）不再需要，已删除。

## `BumpListAlloc` 的 Clone / Drop

```rust
impl<A: Allocator + Send> Clone for BumpListAlloc<A> {
    fn clone(&self) -> Self {
        self.core_().clone_ref_();
        Self { page_: self.page_, _owns_a_: PhantomData }
    }
}

impl<A: Allocator + Send> Drop for BumpListAlloc<A> {
    fn drop(&mut self) {
        if self.core_().release_ref_() {
            unsafe { InitPage::drop_in_place_(self.page_) };
        }
    }
}
```

- `Clone` 不申请 Page、不复制 `A`，只让新句柄指向同一第一页；
- 非最后 Drop 只减计数；
- 最后 Drop 才执行原有 `InitPage::drop_in_place_`（归还普通页、移出 A、归还第一页）。

`Clone` 的 panic 只在计数达到 `2^62 - 1`（64 位）时发生，实际不可达。

## `Allocator` 与 `AllocatorClone`

本次给 `BumpListAlloc<A>` 本身实现：

```rust
unsafe impl<A: Allocator + Send> Allocator for BumpListAlloc<A> { ... }
unsafe impl<A: Allocator + Send> AllocatorClone for BumpListAlloc<A> {}
```

`AllocatorClone` 的安全契约是：clone 出来的 allocator 能释放原 allocator 分配的块。
这里 Clone 共享同一个 arena / Core，因此该契约成立。

`&BumpListAlloc<A>` 的 `Allocator` 与 `AllocatorClone` 不再需要手写：`core` 已经提供

```rust
impl<A: Allocator + ?Sized> Allocator for &A
```

以及对应的 `AllocatorClone`。一旦 `BumpListAlloc` 自身实现 `Allocator`，显式给
`&BumpListAlloc` 写 impl 会与其冲突，因此删除了旧的两个引用 impl。对使用者而言
`&BumpListAlloc` 仍然是 allocator，只是来源变成 core 的通用 impl。

这样一来：

- `Vec<T, BumpListAlloc<A>>` 可以使用拥有型句柄；
- `Vec::clone` 通过 `AllocatorClone` 克隆句柄，在同一 arena 中分配新缓冲区；
- `Vec<T, &BumpListAlloc<A>>` 的借用期语义保持不变；
- `BumpListAllocRef` 仍然服务于 `Arc` / `Rc` / `Box` 等外部 `Deref` 句柄的 trait 转发。

## 并发与锁

- `Clone` / Drop 只碰低位计数；分配 / 归还路径只碰锁位；两者可以在同一个原子字上并发；
- 锁顺序保持不变：list lock → backing lock；Page 锁是独立字段（owner 指针最低位）；
- 最后一个 Drop 不获取任何 Core 锁，因为此时不可能还有合法操作在进行；
- backing lock 仍保证 `A: Send` 足够（不需要 `A: Sync`）。

## 验证

新增 / 更新的测试：

- `state_::tests_::ref_count_and_lock_bits_share_word_`：持锁 clone 不破坏锁位；
- `state_::tests_::list_and_backing_locks_are_independent_`：两把锁标记互不干扰；
- `allocator_is_clone_and_allocator_clone`：编译期断言 `Clone + AllocatorClone`；
- `cloned_handles_share_arena`：Clone 不申请 Page，clone 与原句柄可互相释放块；
- `owned_allocator_supports_vec_clone`：`Vec<T, BumpListAlloc>` 与 `Vec::clone`；
- `concurrent_clones_keep_arena_alive`：多线程并发 clone + 拥有型 `Vec`。

全部测试、Miri（`-Zmiri-strict-provenance`）与 clippy 通过。详见
[`init-page-20261119-1920.md`](init-page-20261119-1920.md) 了解第一页布局与析构顺序。
