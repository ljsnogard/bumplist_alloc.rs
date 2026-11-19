# bumplist-alloc

以**固定大小 Page 链表**组织的 bump 分配器，实现标准
[`core::alloc::Allocator`](https://doc.rust-lang.org/core/alloc/trait.Allocator.html)，
但只对 `&BumpListAlloc` 实现，因此每块分配都带有借用生命周期。

设计目标是“不带 `Retain<T>` 的 sw-scope”：适合小对象 arena，不实现 `Clone`；
底层分配器 `A: Send` 时 `BumpListAlloc` 是 `Send + Sync`（对 `A` 的访问由内部锁串行化，
不要求 `A: Sync`），空 Page 可以归还给底层分配器。

## 快速开始

```rust
use bumplist_alloc::{BumpListAlloc, PageSize};
use std::vec::Vec;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bump = BumpListAlloc::new(PageSize::P4K)?;

    let mut values: Vec<u32, &BumpListAlloc> = Vec::new_in(&bump);
    values.extend_from_slice(&[1, 2, 3, 4]);
    assert_eq!(&values[..], &[1, 2, 3, 4]);

    Ok(())
}
```

`Vec` 的分配器类型是 `&BumpListAlloc`，因此 `values` 存活期间 `bump` 不能被移动或
析构；这正是“每块分配带生命周期”的来源。

## 固定页大小

同一个 `BumpListAlloc` 实例的所有 Page 大小相同，页大小来自有限枚举：

| 选项 | 字节数 |
| --- | ---: |
| `PageSize::P1K` | 1 KiB |
| `PageSize::P2K` | 2 KiB |
| `PageSize::P4K` | 4 KiB |
| `PageSize::P8K` | 8 KiB |
| `PageSize::P16K` | 16 KiB |
| `PageSize::P32K` | 32 KiB |
| `PageSize::P64K` | 64 KiB |
| `PageSize::P128K` | 128 KiB |
| `PageSize::P256K` | 256 KiB（仅 64 位） |

`BlockHeader` 用 15 位记录自身在 Page 内的 cell 下标，因此单个 Page 最多
`1 << 15` 个 cell：

- 64 位平台上限 256 KiB；
- 32 位平台上限 128 KiB，`PageSize::P256K` 会返回
  `BumpListAllocError::UnsupportedPageSize`。

一个分配块必须完整落在一个 Page 内。**超过页可容纳空间的请求返回 `AllocError`**，
不会跨页，也不会临时申请大页。这适合小对象池；`Vec` / `String` 等容器在增长到超过
页容量时通常会走 `handle_alloc_error`，请选择足够大的页或只用小块。

空 `tail_` 页若因对齐要求放不下某个请求，会直接返回 `AllocError`，不会为它追加
同大小的新页；这样能保证 allocator 最多缓存一个空 Page。

## 结构

```text
BumpListAlloc<A>
├── core_: BumpListCore          // 非泛型，直接内联
│   ├── page_size_ / page_cells_ / page_bytes_
│   ├── list_lock_               // 只保护 Page 链表
│   ├── backing_alloc_ / backing_dealloc_  // 两个单态化函数指针
│   ├── self_ptr_                // 整分配器地址；offset_of! 用它取 alloc_
│   └── inner_
│       ├── head_ ──> Page A <──> Page B <──> Page C
│       ├── tail_                     // 分配时优先尝试的页
│       └── tail_empty_: bool         // tail_ 是否为空页
└── alloc_: A                    // 底层分配器；Core 用 offset_of! 反推它
```

> **地址稳定性**：`core_` 直接内联在 `BumpListAlloc` 里，Page 的 owner 指向它。
> 创建第一个 Page 之后，`BumpListAlloc` 不能再移动；需要移动时先 `Box::new(bump)` 固定
> 地址，再进行第一次分配。

每个 Page 有：

- 页首的 Page 头：链表指针、`used_` / `live_` / `first_` / `last_` 与 owner 指针
  （最低位兼作页锁）；`Page` / `PageLinks` / `PageState` / `BlockHeader` / `SpinLock`
  都是 `#[repr(C)]`，字段按大小手动编排；Page 头在 64 位下 32 字节、32 位下 20 字节；
- 一个内嵌的块链：每个 `BlockHeader` 保存自身 cell 下标（低 15 位）、已释放标记
  （最高位）与下一个块头的 cell 下标；
- 页锁：owner 指针的最低位，保护本页的 bump 游标、活跃块计数与块链状态；owner 指向
  非泛型的 `BumpListCore`，它至少按指针宽度对齐，最低位必然为 0，因此不需要额外的锁字段。

普通 `deallocate` 只需要页锁。`live_` 归零时，先把 `used_` / `first_` / `last_`
重置，再通过 Page 的 owner 指针把清理升级给 `BumpListCore`：清理路径取页锁、
`list_lock_`，把这个空页挪到 `tail_` 保留复用。如果已经有
一个空 tail，则把多出来的空页摘链归还。因此 allocator 始终最多缓存一个空 Page，
且它总是 `tail_`；不需要 `retiring` 标记，也不需要 drain。

## 页内块链与 debug

`BlockHeader` 的 `next_free` 是页内块链的后继槽。分配时会把前一块头的后继回填到新
块头，因此可以从 `PageState.first_` 扫描整页的块；释放块在页未空时只设置 `Free`
标记，不改变链；页变空时 `used_` / `first_` / `last_` 一起重置，供下一次复用。这让
Page 可以做内部一致性检查，也用于 `BumpListAlloc` 析构时检测未释放块。

默认构建是 `no_std`，不会打印警告；开启 `std` 特性后，析构时会扫描每个 Page 并把
未释放块数量写到 stderr：

```bash
cargo test --features std
```

## 特性

- 标准 `core::alloc::Allocator`，但只实现给 `&BumpListAlloc`；
- `denco_arena`：额外实现 `abs_denco::de::TrArena` / `TrArenaBytes`，让
  `MpDecoder::new_in` 可以把字符串 / 字节切片分配进同一个分配器；`TrArena::Ref<T>` 是
  `Box`，析构时运行 `T` 的析构函数并调用 `Page::release_` 把块归还到页内；
- 不实现 `Clone`；需要共享时使用 `&` 或 `Arc`；
- 默认 `A = Global`；`A: Send` 时 `BumpListAlloc` 是 `Send + Sync`（`A` 的访问由
  `backing_lock_` 串行化，不要求 `A: Sync`）；
- 所有 Page 同大小；最多缓存一个空 tail Page，其余空页归还；
- `no_std`，无外部依赖；锁为自旋锁。

## 测试

```bash
cargo test
cargo test --features std
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
python3 ../format_use_imports.py --check src tests
```

在 Miri 下（含 `-Zmiri-strict-provenance`）：

```bash
CARGO_HOME=../.tmp/miri-cargo \
XDG_CACHE_HOME=../.tmp/miri-cache \
MIRIFLAGS="-Zmiri-strict-provenance" \
cargo miri test --tests
```

Miri 覆盖页锁/链表锁的线程交互、`BlockHeader` 的 provenance、空页清理/复用与 Drop 路径。

## 开发日志

设计与决策记录在 [`dev-notes/`](dev-notes/)。

## License

AGPL-3.0-or-later，与 sw-scope 保持一致。
