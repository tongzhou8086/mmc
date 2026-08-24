# Blackwell 矩阵乘法优化分享（一）

这个系列的文章中，以我们在 B200 上优化矩阵乘法的经历为基础，讲一个以“流水线编排”为核心的故事，一种从“算法设计”的视角来看待矩阵乘法 kernel 的编写。除流水线编排之外，我们只使用两种必备的常规优化 —— 2-CTA MMA 与 CTA swizzling —— 便能在 4096 及以上的方阵上接近乃至超越 cuBLAS。事实上，文中介绍的第四版设计在 18 个尺寸的方阵下，有 12 个都跑赢了 cuBLAS，steady-state 性能最高可达 1450T 以上。

希望读完之后你能感受到：在补齐必要的 GPU 架构背景之后，高性能矩阵乘法算子的设计，实质上可以被建模成一个妙趣横生的算法设计问题。

## 内容提要

这个系列一共会分为四部分，我们会首先介绍一下矩阵乘法、算术强度的概念以及必要的 GPU 架构背景和 2-CTA MMA、CTA swizzling 这两种常规优化，熟悉这部分内容的读者可以自行跳过，这些都属于背景。第二部分中，我们会先进行流水线设计的理论分析篇，从“数据流动”的角度来看待流水线的编排，我们总结出了 4 种 Buffer 和 5 种操作，每一种操作都可以认为是数据从一种 Buffer 流出，最后流进了另一种 Buffer。相当于是操作数放在源 Buffer 中，操作结果放在目的 Buffer 中。以“4 种 Buffer、5 种操作”为根基，我们又自然地推导出了 Buffer 的读写互斥性，即任何一个 Buffer 都无法同时被一个操作读和被另一个操作写，由此又进一步推导出了这 5 种操作之间任意两种要能够同时进行（即重叠起来）的条件。这便构成了流水线同步机制的根本原理。

之后我们以单 buffer 为例，阐述流水线的时间线，从而看出两类会导致流水线上 MMA issue 停顿的原因，而它们恰好就是两种经典的数据依赖：源 Buffer 不可读造成的 **RAW 停顿**，即一个输出块（output tile）内部由于需要等待 TMA load 数据的就绪；以及目的 Buffer 不可写造成的 **WAR 停顿**，即多个 output tile 之间由于需要等待 MMA buffer 中的数据完成 draining 才能再次写入。针对这两类停顿，我们分别给出了解决方案。
后面第三和第四部分是实现篇，它们实现篇会直接继承理论分析的结果，将它们转化成对应的 CUDA Kernel（文中是以伪代码的形式呈现）。作为例子，我们会呈现四种不同的设计方案，围绕着如何一步步减少上述 RAW、WAR 这两类停顿而展开。

## 理论篇

### 分块矩阵乘法与算术强度
矩阵乘法的计算模式，天然适合于“分块”（Tiling）这样一种优化方式，即每次加载一小块输入到片上，也只计算一小块输出。这样的好处是提高数据局部性，充分使用每一小块的数据进行计算，减少对于全局内存的冗余访问。这里我们假定读者已对数据局部性、分块等基础背景具有相当的了解，便不再赘述其[基本原理](https://zhuanlan.zhihu.com/p/292539074)，直接探讨分块的大小如何影响流水线的编排。

![分块矩阵乘法：A 的一个行条与 B 的一个列条，产生 C 的一个 tile](https://raw.githubusercontent.com/tongzhou8086/mmc/main/blog/figures/tiled-gemm.png)

如上图所示，分块计算的矩阵乘法有三个维度，它们常常被称为 BM、BN、BK，即，每次加载 BMxBK 大小的 A（称为 A tile），以及 BKxBN 大小的 B（称为 B tile），以此计算 BMxBN 大小的 C 的部分结果（Partial accumulation）。BM、BN、BK 我们统称为分块大小（tile sizes），与此相关的另一个核心概念叫做算术强度（arithmetic intensity），它用来衡量每单位的 memory traffic ，譬如每字节，能够产生的计算量是多少。算术强度是一个软件本身的特征，不同的软件编写方式，会产生不同的算术强度。假如 A tile 和 B tile 都能够完全地存放在片上存储中，即 SMEM（Shared Memory，共享内存）或寄存器，那算术强度的计算方式即为

$$ A.I. = \frac{(2\times BM \times BN \times BK)}{2\times BM \times BK + 2\times BK \times BN}$$

通过简单的数学推导，我们可以看出，BM 和 BN 越大，算术强度就越大。所以在实际的矩阵乘法算子的设计与实现中，我们会尽可能把 BM 和 BN 配得更大一点，只要能放得下。不过片上存储空间毕竟有限，你不可能使得 BM 和 BN 无限大。对于 Blackwell 而言，能够使用的最大的 SMEM 的大小是 227 KB，而寄存器的总共的容量是 256KB，TMEM（Tensor Memory）的总容量也是 256KB，这些都限制了 BM、BN、BK 的配置大小。在实际应用中，一个常见的配置方案是 BM=128，BN=256，BK=64 或 128。

### Blackwell 的异步设计 —— 软件成为调度者

现代 GPU 的标配便是在 SM 中配置一个独立的 Tensor Core 单元，用来完成矩阵乘法计算 —— 相当于在通用硬件里面放置了一小块专用硬件，来提高计算效率。从 Hopper 架构开始，又出现了一种新的独立硬件单元，专门负责数据搬运，叫做 TMA。相比传统的 SIMT 式的数据搬运指令，即所有的线程都需要参与，用 TMA 完成数据搬运只需要一个线程发出指令，然后由专用硬件在背后异步完成。
Blackwell 架构将这种单线程发出指令、专用硬件背后完成异步计算的方式进一步推到了极致：无论是 Tensor Core 单元还是 TMA 单元，都只需要一个线程进行指令发射，然后背后的硬件异步完成运算。从软件的角度，这意味着矩阵乘法计算以及数据搬运不再采用传统的 SIMT 模型。GPU 也并非完全摒弃了 SIMT 计算能力，SIMT 算力依然存在。只不过现在做了分工 —— TMA 负责内存与 GPU 片上之间的数据搬运，异步运行；Tensor Core 负责矩阵乘法计算，也是异步运行；而 SIMT 算力则负责其他的，譬如将计算结果在 SMEM 中进行重组、或者特定的 epilogue 计算，同步运行。

这种高度异步化的架构设计的结果就是，软件更多成为了一个调度者的角色，调度 TMA 指令和 MMA 指令什么时候、以何种顺序发射，以及何时进行同步的 epilogue 等等。除了指令的交织与调度，还有一种资源，便是片上的存储资源，即 SMEM 和寄存器如何进行分配？划分出多少用于辅助 TMA、MMA 或者是 epilogue？本文正是从流水线指令与资源调度这样一个视角来展开全文。鉴于篇幅限制，这里我们仅借做一个概述，更多的关于 Blackwell 的硬件特性请参考[Blackwell 架构背景篇]，硬件架构特性实际上会成为流水线设计的 constraints。

## 流水线资源调度
在介绍流水线的资源调度之前，我们先看一下 GEMM kernel 代码的整体框架，以便对数据的生产与消费流程有一个宏观的认知。首先大小为 M x N 的输出矩阵被以 BMxBN 的块大小划分成 M/BM x N/BN 块，每一块就是一个独立的计算任务（output tile）。计算任何的 BM x BN 一小块都需要在 K 维度进行迭代，即每次处理 BK，分 K/BK 步进行。与此同时，由于同一个 CTA 会处理多个 BM x BN 的块（即 persistent kernel，一个 CTA 会常驻在一个 SM 上），于是还会存在一个外层循环，来对不同的块进行迭代。两者循环嵌套起来，便可以得到如下的代码框架：

```text
num_k = ceil(K / BK)                         # 每个 output tile 需要多少次 k 迭代

# my_output_tiles：分配给当前 CTA 的那一批 output tile，分配方式见上图
for tile in my_output_tiles:                 # ── 外层循环：遍历 output tile ──

    acc = zeros(BM, BN)                      # 这个 output tile 的累加器

    for k in range(num_k):                   # ── 内层循环：遍历 k tile ──
        a_tile = load(A, tile.m, k)          #   BM x BK
        b_tile = load(B, k, tile.n)          #   BK x BN
        acc += a_tile @ b_tile               #   一次 BK 的部分累加

    store(C, tile.m, tile.n, acc)            # K 维度累加完毕，写回这块 BM x BN 的结果，又称为 epilogue
```

在实际实现中，我们会开启的两种优化 2-CTA MMA 以及 CTA swizzle 会对如上的代码框架进行轻微调整，但是框架的本质并不会做任何变化。它本质上传达了这样一种流程：数据从内存被按块加载到 GPU 片上，之后会进入 Tensor Core 单元进行 MMA 操作。K 层循环结束，即代表一个输出块的结果计算完毕，这时便将结果写入内存。按照这样的流程依次计算所有分配给当前 CTA 的 output tiles。

### 5 种操作和 5 种容器
我们上述的伪代码是比较宏观的和架构无关的，但在具体的流水线设计中，我们则会引入以下五种 Blackwell 架构特定的操作以及五种逻辑容器，每一种操作会从一种容器里读入数据，然后结果会写入另一种容器。下图是一个图示，箭头代表操作而框框代表容器。每种容器里也注明了它对应的物理存储介质，譬如是 SMEM、TMEM 还是寄存器（RMEM）等。

![5 种操作及其源和目的 buffer](https://raw.githubusercontent.com/tongzhou8086/mmc/main/data-flow-models/figures/operations-chain.png)

其中，TMA load、MMA 以及最后的 TMA store 操作可以直接对应到上述伪代码中的`load`，`a_tile @ b_tile`和`store(C, tile.m, tile.n, acc)`部分。而剩下的两种操作 tcgen05.ld 和 stage，则是 Blackwell 特定的、上述架构无关的伪代码中没有体现出来的。它们存在的原因是因为，根据 Blackwell 架构设计，MMA 的结果必须存储在 TMEM 中，而 TMEM 中的结果必须先通过 tcgen05.ld 系列指令读取到寄存器中以后才能进行后续操作，譬如 epilogue 或写回内存。于是，这里我们便多了一个 tcgen05.ld 操作。与此同时，由于 tcgen05.ld 是按列进行数据读取的，如果将它们的结果直接写入内存，就会导致 uncoalesced memory write。于是我们会先把寄存器中的数据，在 store buffer / SMEM 中做一个临时的缓冲与重组，当一行凑满连续的 128 个字节了，再进行内存写入，直接 issue 一个 TMA store 指令，便能达到 coalesced memory write 的效果。这便是 stage 操作的来历。

> 值得说明的是，我们不保证所有情况下使用 TMA store buffer 进行缓冲后再写入内存都是最高效的，另一种不同的设计完全可以为了节省 SMEM 空间而直接进行 uncoalesced memory write，本文这里探讨的是一种比较通用的设计框架，不保证在任何情况下都是最高性能，但是是比较通用的。

### 容器的状态翻转与操作之间的同步法则
上图中我们也可以看出，既然同一个容器会被一个操作读，也会被另一个操作写，那为了保证操作的正确性，我们需要保证操作的原子性，即，同一时间，任何一个容器无法同时既被读也被写。于是我们给所有的容器都分配两种互斥的状态：可读和可写。一个操作的完成，即产生一个事件，可以翻转容器的状态。假设一个容器的初始状态为“可写”，然后写操作开始，容器里被灌入新的数据，当写操作结束时，容器的状态即翻转为“可读”，代表数据已经就绪。类似地，假如一个容器的状态为“可读”，然后读操作开始，容器里的数据逐渐被消费。当读操作结束时，容器的状态即翻转为“可写”，代表数据已被消费完，内容可以被覆盖了。

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

WAR 停顿来源于 write-after-read 数据依赖，这种依赖并非真实的依赖，在编译原理以及计算机体系结构中的标准解决方案就是让 write 写入一个新的 buffer，在计算机体系结构中，这个被称为 register renaming。于是这里我们通过增加一个 MMA buffer，便能够使得 MMA 操作和 tcgen05.ld 操作重叠起来 —— 各自操作不同的 MMA buffer，如下图所示：

![两份 TMA buffer 加两份 MMA buffer 下的流水线时序](https://raw.githubusercontent.com/tongzhou8086/mmc/main/blog/figures/two-accumulator-timeline.png)

这下 MMA 那一行从头到尾连成了一片，再没有空档 —— 而 MMA 能持续 issue，正是整个流水线编排追求的目标。

## 实现篇

### 各类容器大小的计算方式
流水线设计的第一步便是计算上述的各类容器的大小，TMA buffer 和 MMA buffer 的大小取决于 BM、BN、BK 的配置 《请帮我完成这些 Tile Sizes 以及 Buffer Sizes 的大小的计算写作，以及解释一下 BK 为什么要设成 64 的倍数，我们写完以后就可以把下面“第一种设计：BN256”一章中重复的内容便可以删掉》，tcgen05.ld buffer 我们会默认配置为 128x64，即能装下 TMEM 数据的 64 列，有时会了加快 TMEM 结果的 draining，我们会扩大 tcgen05.ld buffer，即使用尽可能多的寄存器空间用作 tcgen05.ld buffer。Store buffer 的大小我们也默认为 128x64，与 BK 设为 64 的倍数的原因类似，这里凑满 64 列，便能保证 Coalesced Memory Write。

《注明：这里相当于是在介绍所有的流水线设计之前，我们先介绍各类容器大小的计算方式，相当于是一个共有的头文件，之后的所有设计都可以用同样的方式来计算，以及 BK 的配置等等。我的这条注明不需要任何的内容的补全，就仅仅只是解释一下这里的写作的上下文。》

### 第一种设计：BN256
现在我们介绍第一种流水线编排设计。尽管相对基础，它已经使用多个 TMA buffer 来减少 RAW 停顿，以及使用两个 MMA buffer 来减少 WAR 停顿。任何一种流水线设计，我们都需要首先确定 BM、BN、BK 如何配置，由此便能计算各种 buffer 的大小，然后便能计算每一种 buffer 可以配置几个。我们先看 BM 和 BN 的配置。
由于 MMA buffer 的物理存储介质是 TMEM（128行 x 512列），BM 实际上由于硬件限制被固定为 128，与此同时，为了配置两个 MMA buffer 来减少上述的 WAR 停顿，我们将 BN 配置为 512/2 = 256。我们再看 BK 的配置，由于 BK 是 A tile 的 inner dimension，所以考虑到一些内存访问的连续性，我们会将 BK 配置为 64 的倍数，因为在 BF16 数据类型的情况下，BK 等于 64 的倍数，便能得到 128 个连续字节的内存访问模式，不会浪费任何内存带宽。这样我们便能得出逻辑上 A tile 和 B tile 的大小（字节数）分别为：

* A tile：128 * 64 * 2
* B tile：64 * 256 * 2

与此同时，由于 2 CTA MMA 的开启，对于 B tile，一个 cluster 中的两个 CTA 分别只需要加载 B tile 的一半即可，另一半 B tile 数据可以直接从隔壁 SM 读取，这样便使得这个 Tile Sizes 的配置下，一个 A Tile 和 B tile 占用的空间分别是 16KB，一共便是 32KB，也就是一个 TMA buffer 的大小。而一个 store buffer 的大小是 16KB。这里，我们选择配置 6 个 TMA buffer，加上 2 个 store buffer，这样占用的 SMEM 空间正好是  `32*6 + 16*2 = 224KB`，刚好在 227KB 的容量范围内。

tcgen05.ld buffer 的话，我们总是只会配置一个，大小为 128 行 x 64 列，即每次从 MMA buffer 中读取 64 列数据。这里我们没有配置多个 tcgen05.ld buffer，而是让 epilogue warps 串行地进行 tcgen05.ld 和 stage 操作。

确定了参数配置，我们再来看流水线的调度逻辑，这里我们使用上面这套原语来表达：

```text
# ── 参数 ──────────────────────────────────────────────
NS        = 6              # TMA buffer 个数
NUM_ACC   = 2              # MMA buffer 个数（BN=256，TMEM 刚好放得下两个）
NUM_STORE = 2              # store buffer 个数
STORE_N   = 64             # 每次 TMA store 的列数
num_k     = K / BK         # 每个 output tile 需要的 k 迭代次数

# ── Buffer 配置 ───────────────────────────────────────
tma_buffers   = [TMA_Buffer(32KB)        for _ in range(NS)]
mma_buffers   = [MMA_Buffer(128KB)       for _ in range(NUM_ACC)]
ld_buffer     =  TCGEN05_LD_Buffer(32KB)
store_buffers = [Store_Buffer(16KB)      for _ in range(NUM_STORE)]

# ── TMA Warp ─────────────────────────────────────────
gk = 0
for tile in my_output_tiles:                 # 持久化 kernel：每个 CTA 处理若干 output tile
    for k in range(num_k):
        s = (gk++) % NS                      # 在 NS 个 buffer 上轮转
        wait_until_free(tma_buffers[s])
        tma_load_async(A, tile.m, k * BK, BM,     BK,       tma_buffers[s], 0)
        tma_load_async(B, k * BK, tile.n, BK,     BN / 2,   tma_buffers[s], 16KB)
        make_ready_on_tma_done(tma_buffers[s])

# ── MMA Warp ─────────────────────────────────────────
gk = 0
for tile in my_output_tiles:
    acc = tile % NUM_ACC                     # 每换一个 output tile 就换一个 MMA buffer
    for k in range(num_k):
        s = (gk++) % NS
        wait_until_ready(tma_buffers[s])
        if k == 0:
            wait_until_free(mma_buffers[acc])
        issue_mma_chain_async(mma_buffers[acc], tma_buffers[s], accumulate = (k > 0))
        make_free_on_mma_done(tma_buffers[s])    # MMA 消费完这片数据，TMA buffer 就能重新装填
    make_ready_on_mma_done(mma_buffers[acc])     # num_k 次累加全部完成，可以 drain 了

# ── Epilogue Warps ───────────────────────────────────
gs = 0
for tile in my_output_tiles:
    acc = tile % NUM_ACC
    wait_until_ready(mma_buffers[acc])
    for c in range(BN / STORE_N):            # 一次处理 64 列
        tcgen05_ld_x32_async(mma_buffers[acc], c * STORE_N, ld_buffer, 0)
        tcgen05_wait_ld()
        if c == BN / STORE_N - 1:
            make_free(mma_buffers[acc])  # 最后一段读进寄存器即可释放 MMA buffer

        b = (gs++) % NUM_STORE
        wait_until_free(store_buffers[b])
        stage(ld_buffer, store_buffers[b])   # RMEM -> SMEM，同步
        make_ready(store_buffers[b])
        tma_store_async(store_buffers[b], C[tile, c])
        make_free_on_tma_done(store_buffers[b])  # TMA store 写完，这块 SMEM 才能复用
```

这已经是一个高度重叠的流水线：MMA 操作在对 TMA buffer0 的数据进行 MMA 操作时，与此同时，TMA load 操作也在进行，譬如或许在对 TMA buffer1 进行数据写入；与此同时，tcgen05.ld 和 stage 操作也在进行（这两者统称为 draining），譬如对另一个 MMA buffer 中数据进行 draining；与此同时，TMA store 操作也在进行，譬如对已经 draining 结束的数据进行内存写入。

我们前面提到过的 5 种操作，除了 tcgen05.ld 和 stage 是串行的以外，其他所有操作全部并行了起来，全部处于同时运行的状态。这种串行是设计使然：tcgen05.ld buffer 只有一个，所以下一段数据必须等这一段 stage 完、寄存器空出来以后才能从 TMEM 里读出来。

从伪代码里也能读出这个设计最关键的一处安排：MMA warp 每换一个 output tile 就切换 MMA buffer，而 epilogue 在把最后一段读进寄存器之后就立刻 `make_free(mma_buffers[acc])`。两者合起来的效果是，一个 output tile 的 draining 只要在**下下个** output tile 开始前完成，MMA 就不会卡住。这个时间窗口和 K 的大小有关：K 越大窗口越大，K 越小则越可能造成 WAR 停顿。流水线设计的最终目标就是**最小停顿地吃满 MMA**。

值得注意的是，同样的这一套逻辑也完全能够适配 BK=128 的情况，BK=128 与 BK=64 唯一的区别就是 TMA Buffer 从 6 个变为了 3 个，而逻辑部分完全不变 —— 这里我们把 BK 作为一个选配参数，有两种选择：64 和 128。

#### 性能数字

下面我们看一下这个设计在方阵上的性能是多少，事实上我们考虑两种不同的 BK 选配：64/128，以及三种不同的 GROUP_SIZE_M （简称 GSM，代表 CTA swizzle 的深度）选配:8/12/16。不同的 GSM 选配仅仅需要改一个常数参数，而 BK=64/128 的区别也仅仅在于把 TMA buffer 的数量砍半，逻辑部分完全一致。
以下性能数字的测量方法使用 triton.do_bench 获得 median runtime，warmup 和 repetition time 都设置为 1 秒；每个尺寸跑三轮独立的测量，每轮内部再做三次打乱顺序的采样，最后取中位数。

![BN=256 性能对比](https://raw.githubusercontent.com/tongzhou8086/mmc/main/blog/figures/perf-bn256.png)


### 结语
我们这个优化系列的第一篇就写到这里，这是这个系列的第一篇，我们介绍了主旨思想，即从流水线编排的视角来看待矩阵乘法的设计与实现，以及呈现了一个流水线的理论框架。我们也给出了第一种流水线编排设计，尽管性能数字尚未跑赢 CUBLAS，但是它已经为我们后续的设计奠定了一个坚实的基础。Stay tuned for the next article！


