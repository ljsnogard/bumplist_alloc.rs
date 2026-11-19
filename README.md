# bumplist-alloc

以**固定大小 Page 链表**组织的 bump 分配器，实现标准
[`core::alloc::Allocator`](https://doc.rust-lang.org/core/alloc/trait.Allocator.html)，
但只对 `&BumpListAlloc` 实现，因此每块分配都带有借用生命周期。

设计目标是“不带 `Retain<T>` 的 sw-scope”：适合小对象 arena，不实现 `Clone`；
底层分配器 `A: Send` 时 `BumpListAlloc` 是 `Send + Sync`（对 `A` 的访问由内部锁串行化，
不要求 `A: Sync`），空 Page 可以归还给底层分配器。

`BumpListAlloc<A>` 本身只是一个指向堆上第一页的指针（与 `usize` 同宽），因此可以安全
移动；第一页里同时保存 `Page` 头、非泛型 `BumpListCore` 与底层分配器 `A`。

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
析构；这仍然来自“每块分配带借用生命周期”，而不是 `BumpListAlloc` 自身的地址限制。
只要没有未结束的借用，`BumpListAlloc` 可以自由移动、放进结构体或返回。

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
同大小的新页。第一页因为要容纳 `InitPage<A>` 而容量更小；如果请求只是超过第一页的
容量、但仍能放进普通页，分配路径会追加一个普通页。

## 结构

```text
BumpListAlloc<A>                 // 只有一个指针（#[repr(transparent)]）
└── NonNull<InitPage<A>> ──> 第一页内存（page_size 字节）
    ├── Page                     // 页头，偏移 0
    ├── BumpListCore             // 非泛型核心；Page.owner_ 指向它
    │   ├── page_size_
    │   ├── root_page_           // 第一页；永不归还
    │   ├── backing_: NonNull<dyn Allocator>  // 指向下面的 A
    │   ├── list_lock_ / backing_lock_
    │   └── inner_: PageList
    │       ├── head_ ──> Page A <──> Page B <──> Page C
    │       └── tail_                     // 分配时优先尝试的页
    ├── MaybeUninit<A>           // 底层分配器
    └── ... 用户可分配区域 ...

普通页：Page 头 + 用户可分配区域
```

> **可移动性**：`BumpListAlloc` 只保存第一页指针；Page 的 owner 指向第一页里地址
> 稳定的 Core。移动句柄不会移动第一页，所以创建 Page 之后仍可自由移动；不需要
> `Box::new` / `pin`。

`BumpListCore` 不再保存 `page_cells_` / `page_bytes_`（由 `page_size_` 直接计算），
也不再有“整分配器自指针”和两个单态化函数指针。底层分配器访问统一通过
`backing_: NonNull<dyn Allocator>`，它占两个指针宽度，与原来的两个函数指针相同；
`backing_lock_` 串行化所有对 `A` 的 allocate / deallocate 调用。

每个 Page 有：

- 页首的 Page 头：链表指针、`base_` / `cells_` / `used_` / `returned_` 与 owner
  指针（最低位兼作页锁）；`Page` / `PageLinks` / `PageState` / `BlockHeader` /
  `SpinLock` 都是 `#[repr(C)]`，字段按大小手动编排；Page 头在 64 位下 32 字节、
  32 位下 20 字节；
- `PageState` 的 4 个 `u16`：`base_` 是首个可分配 cell 的页内绝对下标，`cells_` 是
  从 `base_` 起可分配的 cell 数，`used_` 是相对 `base_` 的 bump 偏移，`returned_`
  是相对 `base_` 的已归还 cell 数；普通页的 `base_` 是 `Page::CELLS`，第一页的
  `base_` 还要越过 Core 与 `A`，因此第一页容量更小；
- 一个内嵌的块链：每个 `BlockHeader` 保存自身 cell 下标（低 15 位）、已释放标记
  （最高位）与本块末尾（下一块头 / bump 末尾）的 cell 下标；有对齐 padding 时，
  紧邻用户块之前的 cell 是 locator，记录到块头的距离；
- 页锁：owner 指针的最低位，保护本页的 bump 游标、`PageState` 与块链状态；owner
  指向非泛型的 `BumpListCore`，它至少按指针宽度对齐，最低位必然为 0，因此不需要额外
  的锁字段。

普通 `deallocate` 只需要页锁。`returned_ == used_` 表示整页已经全部归还；此时通过
Page 的 owner 指针把清理升级给 `BumpListCore`：清理路径按「链表锁 → 页锁」取锁。
第一页永远保留；普通空页遵循“最多保留一个空页”的策略，root 页为空时会把多余的空页
归还。`PageList` 不需要 `tail_empty_` 标记，也不需要 `retiring` / drain。

第一页的构造与析构是自引用的关键：构造时先申请 `page_size` 字节、按
`align_of::<InitPage<A>>()` 对齐的内存，再原地写 Page、A、Core；析构时先归还所有普通
页，销毁 Core，把 A 从第一页移出到栈上，最后用移出的 A 归还第一页。这样 A 不会在
“正在被自己释放的内存”上被析构。

## 页内块链与 debug

`BlockHeader` 的 `next_free` 是本块末尾（也就是下一块头的位置）。分配时块头放在当前
bump 位置，`next_free = end`，因此块链天然连续，可以从 `PageState.base_` 一路走到
`base_ + used_`，不需要 `first_` / `last_`。释放块只设置 `Free` 标记，并把
`next_free - header_cell` 累加到 `returned_`；`returned_ == used_` 时把两者归零，
供下一次复用。这既让 Page 可以做内部一致性检查，也用于 `BumpListAlloc` 析构时检测
未释放块。

默认构建是 `no_std`，不会打印警告；开启 `std` 特性后，析构时会扫描每个 Page 并把
未释放块数量写到 stderr：

```bash
cargo test --features std
```

## 特性

- 标准 `core::alloc::Allocator`，但只实现给 `&BumpListAlloc`；`BumpListAlloc` 本身是
  单指针、可移动的句柄；
- `denco_arena`：额外实现 `abs_denco::de::TrArena` / `TrArenaBytes`，让
  `MpDecoder::new_in` 可以把字符串 / 字节切片分配进同一个分配器；`TrArena::Ref<T>` 是
  `Box`，析构时运行 `T` 的析构函数并调用 `Page::release_page_` 把块归还到页内；
- 不实现 `Clone`；需要共享时使用 `&` 或 `Arc`；
- 默认 `A = Global`；`A: Send` 时 `BumpListAlloc` 是 `Send + Sync`（`A` 的访问由
  `backing_lock_` 串行化，不要求 `A: Sync`）；
- 所有 Page 同大小；第一页永久保留，此外最多保留一个空 Page，其余空页归还；
- 构造时立即申请第一页，因此 `new` / `new_in` 可能因页大小或底层分配失败而返回错误；
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
