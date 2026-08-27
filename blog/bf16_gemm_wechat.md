# Blackwell 矩阵乘法优化分享（一）

这个系列的文章中，以我们在 B200 上优化矩阵乘法的经历为基础，讲一个以“流水线编排”为核心的故事，一种从“算法设计”的视角来看待矩阵乘法 kernel 的编写。除流水线编排之外，我们只使用两种必备的常规优化 —— 2-CTA MMA 与 CTA swizzling —— 便能在 4096 及以上的方阵上接近乃至超越 cuBLAS。事实上，本系列中介绍的第四版设计在 18 个尺寸的方阵下，有 12 个都跑赢了 cuBLAS，steady-state 性能最高可达 1450T 以上。

## 背景

### 分块矩阵乘法与算术强度
矩阵乘法的计算模式，天然适合于“分块”（Tiling）这样一种优化方式，即每次加载一小块输入到片上，也只计算一小块输出。这样的好处是提高数据局部性，充分使用每一小块的数据进行计算，减少对于全局内存的冗余访问。这里我们假定读者已对数据局部性、分块等基础背景具有相当的了解，便不再赘述其[基本原理](https://zhuanlan.zhihu.com/p/292539074)，直接探讨分块的大小如何影响流水线的编排。

![分块矩阵乘法：A 的一个行条与 B 的一个列条，产生 C 的一个 tile](https://raw.githubusercontent.com/tongzhou8086/mmc/main/blog/figures/tiled-gemm.png)

如上图所示，分块计算的矩阵乘法有三个维度，它们常常被称为 BM、BN、BK，即，每次加载 BMxBK 大小的 A（称为 A tile），以及 BKxBN 大小的 B（称为 B tile），以此计算 BMxBN 大小的 C 的部分结果（Partial accumulation）。BM、BN、BK 我们统称为分块大小（tile sizes），与此相关的另一个核心概念叫做算术强度（arithmetic intensity），它用来衡量每单位的 memory traffic ，譬如每字节，能够产生的计算量是多少。算术强度是一个软件本身的特征，不同的软件编写方式，会产生不同的算术强度。假如 A tile 和 B tile 都能够完全地存放在片上存储中，即 SMEM（Shared Memory，共享内存）或寄存器，那算术强度的计算方式即为

$$ A.I. = \frac{(2\times BM \times BN \times BK)}{2\times BM \times BK + 2\times BK \times BN}$$

通过简单的数学推导，我们可以看出，BM 和 BN 越大，算术强度就越大（BK 在分子分母里面被抵消掉了）。所以在实际的矩阵乘法算子的设计与实现中，我们会尽可能把 BM 和 BN 配得更大一点，只要片上存储资源能放得下。

### Blackwell 的异构设计 —— 流水线调度成为性能的关键

现代 GPU 的标配便是在 SM 中配置一个独立的 Tensor Core 单元，用来完成矩阵乘法计算 —— 相当于在通用硬件里面放置了一小块专用硬件，来提高计算效率。从 Hopper 架构开始，又出现了一种新的独立硬件单元，专门负责数据搬运，叫做 TMA。相比传统的 SIMT 式的数据搬运指令，即所有的线程都需要参与，用 TMA 完成数据搬运只需要一个线程发出指令，然后由专用硬件在背后异步完成。
Blackwell 架构将这种单线程发出指令、专用硬件背后完成异步计算的方式进一步推到了极致：无论是 Tensor Core 单元还是 TMA 单元，都只需要一个线程进行指令发射，然后背后的硬件异步完成运算。从软件的角度，这意味着矩阵乘法计算以及数据搬运不再采用传统的 SIMT 模型。当然，GPU 并没有摒弃通用计算单元，CUDA Cores、Integer Cores 等依然在那里。只不过现在做了分工 —— TMA 负责内存与 GPU 片上之间的数据搬运，异步运行；Tensor Core 负责矩阵乘法计算，也是异步运行；而 CUDA Cores、Integer Cores 等通用单元则负责其余的工作，譬如将计算结果在 SMEM 中进行重组、或者特定的 epilogue 计算，仍以传统的 SIMT 方式同步运行。

这种高度异步化的架构设计的结果就是，软件更多成为了一个调度者的角色，调度 TMA 指令和 MMA 指令什么时候、以何种顺序发射，以及何时进行同步的 epilogue 等等。除了指令的交织与调度，还有一种资源，便是片上的存储资源，即 SMEM 和寄存器如何进行分配？划分出多少用于辅助 TMA、MMA 或者是 epilogue？本文正是从流水线指令与资源调度这样一个视角来展开全文。鉴于篇幅限制，这里我们仅借做一个概述，更多的关于 Blackwell 的硬件特性请参考 [semianalysis 的文章](https://newsletter.semianalysis.com/p/dissecting-nvidia-blackwell-tensor)，硬件架构特性实际上会成为流水线设计的 constraints。

## 理论

### GEMM kernel 的宏观框架
在介绍流水线的资源调度之前，我们先看一下 GEMM kernel 代码的整体框架，以便对数据的生产与消费流程有一个宏观的认知。首先大小为 M x N 的输出矩阵被以 BMxBN 的块大小划分成 ceil(M/BM) x ceil(N/BN) 块，每一块（output tile）就是一个独立的计算任务。计算任何的 BM x BN 一小块都需要在 K 维度进行迭代，即每次处理 BK，分 ceil(K/BK) 步进行。与此同时，由于同一个 CTA 会处理多个 BM x BN 的块（即 persistent kernel，一个 CTA 会常驻在一个 SM 上），于是还会存在一个外层循环，来对不同的块进行迭代。两者循环嵌套起来，便可以得到如下的代码框架：

```text
num_k = ceil(K / BK)                         # 每个 output tile 需要多少次 k 迭代

# my_output_tiles：分配给当前 CTA 的那一批 output tile
for tile in my_output_tiles:                 # ── 外层循环：遍历 output tile ──

    acc = zeros(BM, BN)                      # 这个 output tile 的累加器

    for k in range(num_k):                   # ── 内层循环：遍历 k tile ──
        a_tile = load(A, tile.m, k)          #   BM x BK
        b_tile = load(B, k, tile.n)          #   BK x BN
        acc += a_tile @ b_tile               #   一次 BK 的部分累加

    store(C, tile.m, tile.n, acc)            # K 维度累加完毕，写回这块 BM x BN 的结果，又称为 epilogue
```

这段伪代码传达了这样一种流程：数据从内存被按块加载到 GPU 片上，之后会进入 Tensor Core 单元进行 MMA 操作。K 层循环结束，即代表一个输出块的结果计算完毕，这时便将结果写入内存。按照这样的流程依次计算所有分配给当前 CTA 的输出块。
在实际实现中，我们会开启的两种优化 2-CTA MMA 以及 CTA swizzle 会对如上的代码框架进行轻微调整，但是框架的本质、循环的结构并不会有任何变化。2-CTA MMA 可以使得两个 CTA 构成一个 cluster 协同计算大小为 2BM x BN 的输出块，如下图所示；而 CTA swizzle 则类似于一种 L2 缓存分块，通过改变输出块分配给 CTA 的顺序来提高 L2 缓存命中率。


![2-CTA MMA 开启前后：B tile 从读两遍变成只读一遍](https://raw.githubusercontent.com/tongzhou8086/mmc/main/blog/figures/two-cta-mma.png)


### 5 种操作和 5 种 buffer
我们上述的伪代码是比较宏观的和架构无关的，但在具体的流水线设计中，我们则会引入以下五种 Blackwell 架构特定的操作以及五种逻辑 buffer，每一种操作会从一种 buffer 里读入数据，然后结果会写入另一种 buffer。下图是一个图示，箭头代表操作而框框代表 buffer。每种 buffer 里也注明了它对应的物理存储介质，譬如是 SMEM、TMEM 还是寄存器（RMEM）等。

![5 种操作及其源和目的 buffer](https://raw.githubusercontent.com/tongzhou8086/mmc/main/data-flow-models/figures/operations-chain.png)

其中，TMA load、MMA 以及最后的 TMA store 操作可以直接对应到上述伪代码中的`load`，`a_tile @ b_tile`和`store(C, tile.m, tile.n, acc)`部分。而剩下的两种操作 tcgen05.ld 和 stage，则是 Blackwell 特定的、上述架构无关的伪代码中没有体现出来的。它们存在的原因是因为，根据 Blackwell 架构设计，MMA 的结果必须存储在 TMEM 中，而 TMEM 中的结果必须先通过 tcgen05.ld 系列指令读取到寄存器中以后才能进行后续操作，譬如 epilogue 或写回内存。于是，这里我们便多了一个 tcgen05.ld 操作。与此同时，由于 tcgen05.ld 是按列进行数据读取的，如果将它们的结果直接写入内存，就会导致 uncoalesced memory write。于是我们会先把寄存器中的数据，在 store buffer / SMEM 中做一个临时的缓冲与重组，当一行凑满连续的 128 个字节了，再进行内存写入，直接 issue 一个 TMA store 指令，便能达到 coalesced memory write 的效果。这便是 stage 操作的来历。

> 值得说明的是，我们不保证所有情况下使用 TMA store buffer 进行缓冲后再写入内存都是最高效的，另一种不同的设计完全可以为了节省 SMEM 空间而直接进行 uncoalesced memory write，本文这里探讨的是一种比较通用的设计框架，不保证在任何情况下都是最高性能，但是是比较通用的。

### Buffer 的状态翻转与操作之间的同步法则
上图中我们也可以看出，既然同一个 buffer 会被一个操作读，也会被另一个操作写，那为了保证操作的正确性，我们需要保证操作的原子性，即，同一时间，任何一个 buffer 无法同时既被读也被写。于是我们给所有的 buffer 都分配两种互斥的状态：可读和可写。一个操作的完成，即产生一个事件，可以翻转 buffer 的状态。假设一个 buffer 的初始状态为“可写”，然后写操作开始，buffer 里被灌入新的数据，当写操作结束时，buffer 的状态即翻转为“可读”，代表数据已经就绪。类似地，假如一个 buffer 的状态为“可读”，然后读操作开始，buffer 里的数据逐渐被消费。当读操作结束时，buffer 的状态即翻转为“可写”，代表数据已被消费完，内容可以被覆盖了。

于是可以自然地推导出，任何一个操作要能够开始必须同时满足的两个条件是：
* 源 Buffer 可读
* 目的 Buffer 可写

这是整个流水线调度设计正确性保障的根本原理。

> 对于 TMA load 和 store 操作而言，内存可以视为总是可读或者总是可写

延伸一下，我们可以推导出，任意两个操作要能够同时并行运行的条件：

* 其中一个操作的源 buffer 不是另一个操作的目的 buffer

### 单 buffer 的流水线时序图
根据上面的同步法则，我们可以推导出在所有 buffer 只配置一份的情况的流水线的时序图，即：

* TMA load 不能与 MMA 同时运行，但是可以和 tcgen05.ld 及其后续操作同时运行
* MMA 不能与 TMA load 或者 tcgen05.ld 同时运行，但是可以和 stage 以及 TMA store 同时运行
* 以此类推等等

假设每个 output tile 只需要两轮内层循环，即 K = 2*BK，也假设只有两个 output tile，我们可以画出如下的流水线时序图。

![单 buffer 配置下的流水线时序（K = 2·BK）](https://raw.githubusercontent.com/tongzhou8086/mmc/main/blog/figures/single-buffer-timeline.png)

让我们把视线专注在 MMA 的 issue 那一行，会发现它有两种类型的停顿，而它们恰好对应经典的两种数据依赖：

* **RAW 停顿**：同一个 output tile，不同的 k tile 之间存在停顿，需要等待 TMA load 的结束。MMA 要读的那份数据得先由 TMA load 写进去，这是一个 true dependence（read after write）。
* **WAR 停顿**：不同的 output tile 交接时，也存在空挡，需要等待 tcgen05.ld 的结束。MMA 要写的那块 accumulator 得先被 tcgen05.ld 读走，这是一个 anti dependence（write after read）。

这两种依赖，其实正是前面同步法则的两半：「源 buffer 可读」说的是 true dependence 已经满足，「目的 buffer 可写」说的是 anti dependence 已经满足。

### 解决 RAW 停顿

RAW 停顿来源于 read-after-write 数据依赖，减少停顿的方法则是预取（prefetch）数据，即 read 的时候同时开始下一轮的 write，这样可以减少下次 read 的时候的等待。这样的数据预取需要我们使用多个 TMA buffer，即当一个 MMA 开始时，与此同时，TMA load 可以同时开始往另一个 buffer 里面写入。
下图演示使用两个 TMA buffer 的情况，实际具体用几个 TMA buffer 最优取决于 MMA 和 TMA load 的时长比例。

![两份 TMA buffer 下的流水线时序](https://raw.githubusercontent.com/tongzhou8086/mmc/main/blog/figures/two-tma-buffer-timeline.png)

### 解决 WAR 停顿

WAR 停顿来源于 write-after-read 数据依赖，这种依赖并非真实的依赖，在编译原理以及计算机体系结构中的标准解决方案就是让 write 写入一个新的 buffer。于是这里我们通过增加一个 MMA buffer，便能够使得 MMA 操作和 tcgen05.ld 操作重叠起来 —— 各自操作不同的 MMA buffer，如下图所示：

![两份 TMA buffer 加两份 MMA buffer 下的流水线时序](https://raw.githubusercontent.com/tongzhou8086/mmc/main/blog/figures/two-accumulator-timeline.png)

这下 MMA 那一行从头到尾连成了一片，再没有空档 —— 而 MMA 能持续 issue，正是整个流水线编排追求的目标。

## 实现

### 各类 buffer 大小的计算方式
片上存储空间毕竟有限，BM 和 BN 不可能无限大。在计算各类 buffer 的大小之前，我们先把 Blackwell 的硬件限制集中列一下，它们是后面所有配置的边界条件：

* SMEM 容量：每个 SM 最多可用 227KB —— TMA buffer 和 store buffer 都从这里出，它们一共能配几个由这个总量决定
* TMEM 容量：128 行 x 512 列的 fp32，共 256KB —— MMA 的结果只能放在这里，而且必须先经 tcgen05.ld 读进寄存器才能进入后续操作
* 寄存器（RMEM）容量：每个 SM 共 256KB —— tcgen05.ld buffer 从这里出
* 单次 MMA 指令粒度：BM 固定为 128（正好对应 TMEM 的 128 行），N 只能取 64 / 128 / 256
* 内存访问粒度：对内存的读写最好以 128 个连续字节为单位，这既约束了 BK 的取值（读取内存时的连续），也约束了 store buffer 的列数（写入内存时的连续）
* 2-CTA MMA：一个 cluster 中的两个 CTA 各自只需加载半个 B tile，另一半 B tile 数据可以直接从隔壁 SM 读取，于是 TMA buffer 里 B 的那一半也随之减半

由这几条可以直接推出 tile sizes 的取值范围。BM 没有选择，只能是 128。BN 最大为 512，因为 TMEM 一行只有 512 个 fp32；当 BN 配为 256 时，一个 MMA buffer 是 128 行 x 256 列，整个 TMEM 正好放得下两个。BK 则不由容量决定，而是由内存访问的连续性决定：BK 是 A tile 的 inner dimension，在 BF16 下把 BK 配成 64 的倍数，一次访问正好凑满 128 个连续字节，不浪费任何内存带宽。

按上述取值范围，一个 A tile 和 B tile 分别的字节数的计算方式则分别是 BM * BK * 2 以及 BK * BN * 2。它们的大小将决定了 SMEM 中能放置几个 TMA buffer，以及 TMA buffer 和 Store buffer 分别应该配置多少个。与此同时，由于 2 CTA MMA 的开启，每个 CTA 实际需要读取的 B tiles 的字节数减半，变为 BK * BN。由此也可以看出 2 CTA MMA 的双重收益：即能提高算术强度（保持计算量不变的情况下，减少内存数据的加载），又能缩小 TMA buffer 的大小从而腾出更多的 SMEM 空间。

tcgen05.ld buffer 我们会默认配置为 128 x 64，即能装下 TMEM 数据的 64 列，有时会了加快 TMEM 结果的 draining（减少 WAR 停顿），我们会扩大 tcgen05.ld buffer，使用尽可能多的寄存器空间，这将是我们后续优化的一个重头戏。Store buffer 的大小我们也默认为 128 x 64， 即一个 buffer 大小为 16KB。与 BK 设为 64 的倍数的原因类似，这里每行凑满 64 列，便能保证 Coalesced Memory Write。

### Warp Specialization 配置
因为我们的 5 种流水线操作需要能够并行起来，于是我们会给不同的 Warp 分配不同的角色，这也称为 Warp Specialization。对于本文所探讨的设计方案而言，均采用如下配置：

* 配一个 Warp 进行异步的 TMA load issue，又称 TMA Warp
* 配一个 Warp 进行异步的 MMA issue，又称 MMA Warp
* 配 4 个或者 8 个 warps 进行同步的 tcgen05.ld 和 stage，这些 warps 也被称为 epilogue warps
* TMA Store 操作比较简单，也顺便由上面的 epilogue warps 完成异步指令 issue，减少 warp 调度

### 设计一：双 buffer BN256
现在我们介绍第一种流水线编排设计。尽管相对基础，它已经使用多个 TMA buffer 来减少 RAW 停顿，以及使用两个 MMA buffer 来减少 WAR 停顿，其 BM 配置为 128，BN 配为 256，BK 则有 64 和 128 两种不同的选配。以 BK 为 64 为例，我们可以计算出一个 A tile 和 B tile 占用的空间分别是 16KB 和 32KB，而由于 2 CTA MMA 的开启，一个 CTA 实际需要加载的数据量是 16 + 32/2 = 32KB，也就是一个 TMA buffer 的大小。而一个 store buffer 的大小是 16KB。这里，我们的思路是把 TMA buffer 配置多一点，6 个，来加深数据预取流水线深度，而 store buffer 仅仅只配两个。这样占用的 SMEM 空间正好是  `32*6 + 16*2 = 224KB`，刚好在 227KB 的容量范围内。

tcgen05.ld buffer 的话，我们遵循默认配置，大小为 128 行 x 64 列，即每次从 MMA buffer 中读取 64 列数据。确定了参数配置，我们来看 TMA warp、MMA warp 的调度逻辑，这里我们使用一套[调度原语](https://github.com/tongzhou8086/mmc/blob/main/docs/pipeline-primitives.md)来表达：

```text
# ── TMA Warp ─────────────────────────────────────────
gk = 0
for tile in my_output_tiles:                 # 持久化 kernel：每个 CTA 处理若干 output tile
    for k in range(num_k):
        s = (gk++) % NS                      # 在 NS 个 TMA buffer 上轮转
        wait_until_free(tma_buffers[s])
        tma_load_async(A, tile.m, k * BK, BM,     BK,       tma_buffers[s], 0)
        tma_load_async(B, k * BK, tile.n, BK,     BN / 2,   tma_buffers[s], 16KB)
        make_ready_on_tma_done(tma_buffers[s])

# ── MMA Warp ─────────────────────────────────────────
gk = 0
for tile in my_output_tiles:
    acc = tile % 2                           # 每换一个 output tile 就换一块 MMA buffer
    for k in range(num_k):
        s = (gk++) % NS
        wait_until_ready(tma_buffers[s])
        if k == 0:
            wait_until_free(mma_buffers[acc])
        issue_mma_chain_async(mma_buffers[acc], tma_buffers[s], accumulate = (k > 0))
        make_free_on_mma_done(tma_buffers[s])    # MMA 消费完这片数据，TMA buffer 就能重新装填
    make_ready_on_mma_done(mma_buffers[acc])     # num_k 次累加全部完成，可以 drain 了
```
可以看出这里 TMA warps 和 MMA warps 都有两层循环，外层循环对应不同的 output tiles，而内存循环对应同一个 output tiles 不同的 K tile 迭代。然后二者都使用一个全局的计数器`gk`在 6 个 TMA buffer（对于 BK=64 是 6 个，对于 BK=128 则是 3 个）上轮转。对于 TMA Warp 而言，它每次先等待对应的 TMA buffer 变为“可写”状态，之后发出加载数据的指令`tma_load_async`，以及一个异步的信号注册`make_ready_on_tma_done`，即，当数据加载到位以后，自动翻转 buffer 状态为“可读”。同样的，对于 MMA Warp 而言，它每次也是先等待对应的 TMA buffer 变为可读状态，之后便 issue MMA 指令，然后注册一个异步的信号`make_free_on_mma_done`，即，当对应的 TMA buffer 中的数据被消费完以后自动翻转其状态为可写。两者结构上唯一的差别是 MMA warp 多了一层 MMA buffer 的轮转 —— `acc = tile % 2`，每换一个 output tile 就换一块 accumulator，即用来消除 WAR 停顿的双 MMA buffer；每个 output tile 的第一次迭代前需要等待对应的 accumulator 已经被 drain 干净，而 num_k 次累加做完之后再用 `make_ready_on_mma_done` 通知 epilogue 可以开始 drain。

epilogue warps 那一侧的逻辑如下：

```text
# ── Epilogue Warps ───────────────────────────────────
for tile in my_output_tiles:
    acc = tile % 2                                     # 和 MMA warp 相同的轮转
    wait_until_ready(mma_buffers[acc])                 # 等这块 accumulator 累加完毕
    for c in range(BN / 64):                           # 每次 drain 64 列
        tcgen05_ld(mma_buffers[acc], c * 64, ld_buffer)  # TMEM -> RMEM，同步
        if c == BN / 64 - 1:
            make_free(mma_buffers[acc])                # 最后一段读完，MMA 就能开始下一个 tile
        tma_store_wait(1)                              # 等待上上轮的 store buffer 空出，1 代表只有 1 个 store in-flight
        stage(ld_buffer, store_buffers[c % 2])         # RMEM -> SMEM，同步
        tma_store_async(store_buffers[c % 2], C[tile, c])   # SMEM -> 内存，异步
```
在 epilogue 的部分，我们先等待 MMA buffer 变为“可读”，然后在一个循环里面每次读取 TMEM 结果的 64 列，在最后的 64 列读取完毕以后，便可释放完成 draining 的 MMA buffer，即上面的`make_free`。每 64 列读取完以后，先等待有 store buffer 空出来了，然后往里面进行写入，最后 issue 一个 TMA store。

#### 性能数字

下面我们看一下这个设计在方阵上的性能是多少，我们一共有 6 种不同的实现，包括两种不同的 BK 选配：64/128，以及三种不同的 GROUP_SIZE_M （简称 GSM，代表 CTA swizzle M 方向上的深度）选配：8/12/16。不同的 GSM 选配仅仅需要改一个常数参数，而 BK=64/128 的区别也仅仅在于把 TMA buffer 的数量砍半，逻辑部分完全一致。
以下性能数字的测量方法使用 triton.do_bench 获得 median runtime，warmup 和 repetition time 都设置为 1 秒；每个尺寸跑三轮独立的测量，每轮内部再做三次打乱顺序的采样，最后取中位数。

![BN=256 性能对比](https://raw.githubusercontent.com/tongzhou8086/mmc/main/blog/figures/perf-bn256.png)


### 设计二：单 buffer BN512
BN256 的设计其实已经非常流畅了：TMA buffer 有 6 个，应该能够流畅地将数据加载到 SMEM 中，这样 MMA 总是有操作数可以计算 —— 这一部分针对的是 RAW 停顿；此外 MMA buffer 也有两个，所以如果一个 output tile 的 epilogue 能在下下个 output tile 开始之前完成，WAR 停顿也就被藏了起来。两者合起来，理论上我们就可以持续不停地 issue MMA，这已经是一个非常流畅的设计了。

不过，根据实际性能结果，我们发现，在比较大的方阵上，BN256 的性能停滞在了 1300T 左右。一个可能的原因是，尽管 MMA 的 issue 的确没什么停顿，但是每次只用了一半的 TMEM 做 accumulation 达到的算术强度还是不太够，也就是说，同样的数据量加载进来，它产生的计算量有限。

于是在设计二中我们换一种思路：把整个 TMEM 当作一块 MMA buffer 用，BN 配为 512。算一下就能看出这笔交易的两面 —— 每个 k tile 的数据量变为 1.5 倍（32KB → 48KB），而计算量变为两倍，也就是说同样的数据能喂出更多的计算，等价于减少了 RAW 停顿；代价则是只剩一块 accumulator，epilogue 在 drain 它的时候后续的 MMA 发不出去，于是在两个 output tile 的交接处换回了 WAR 停顿。由于 WAR 停顿发生在外层循环、RAW 停顿发生在内层循环，k 迭代次数越多，这笔交易就越划算。

我们再计算一下 TMA buffer 的大小和数量，BM 依然设为 128，BN 现在是 512，BK 先取 64，套用前面的公式：

* A tile: BM × BK × 2 = 128 × 64 × 2 = **16KB**
* B tile: BK × BN × 2 = 64 × 512 × 2 = 64KB，2-CTA MMA 下每个 CTA 只加载一半，即 **32KB**

所以一个 TMA buffer（一个 A tile 加一个 B tile）是 **48KB**，于是我们给 TMA buffer 配 **4 个**，共 192KB；store buffer 依然配 **2 个**，每个是 128 × 64 × 2 = 16KB，共 32KB。两者相加还是 224KB。

调度逻辑几乎可以照搬设计一，所以这里不再重复贴一遍代码，只说三处差别：

* **TMA warp**：代码完全一样，只是每次搬的数据量从 32KB 变成 48KB（B 的那一半从 16KB 变成 32KB），buffer 个数从 6 个变成 4 个（BK=128 时为 2 个）。
* **MMA warp**：`acc = tile % 2` 这一层轮转消失了 —— 只有一块 accumulator，`wait_until_free` 等的永远是同一块。于是下一个 output tile 的第一条 MMA 必须等当前 tile 彻底 drain 完才能发出，这正是前面说的、用算术强度换回来的 WAR 停顿。
* **Epilogue warps**：结构不变，仍然是按 64 列一段依次 drain，只是 BN 翻倍之后每个 output tile 从 4 段变成 8 段。

#### 性能数字

BN512 的上述设计也可以有六种选配，BK=64/128，GSM=8/12/16，如果 BK=64 和 128 分别使用 4 个和 2 个 TMA buffer。性能测量方法与上面相同。

![BN=512 性能对比](https://raw.githubusercontent.com/tongzhou8086/mmc/main/blog/figures/perf-bn512.png)


几点观察：

* 相比 BN256，BN512 在稍大一些的方阵上确实能达到高得多的性能 —— 稳定飚在了 1450T 左右。一个可能的解释是，方阵越大 K 维度也越大，Epilogue 所占的时间比例便相应缩小 —— BN512 让计算部分更快，代价是 Epilogue 带来的 WAR 停顿更难藏，所以正适合 K 比较大的情况。
* GSM 在这里的作用比 BN256 明显得多：BK=128 在 20480 上从 GSM=8 的 1423 涨到 GSM=16 的 1468；BK=64 在 17408 上从 1316 涨到 1412。
* 最好的一档（BK=128 + GSM=16）在 18 个尺寸中有 10 个跑赢了 cuBLAS。

### 设计三：双 buffer BN512


## 结语
我们这个优化系列的第一篇就写到这里，这是这个系列的第一篇，我们介绍了主旨思想，即从流水线编排的视角来看待矩阵乘法的设计与实现，以及呈现了一个流水线的理论框架。我们也给出了两种流水线编排设计，已经在 18 个尺寸中有 10 个跑赢了 cuBLAS。在后续的系列中，我们还会继续优化之旅，stay tuned！


