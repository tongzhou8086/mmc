# B200 矩阵乘法优化分享

本文以我们在 B200 上优化矩阵乘法的经历为基础，讲一个以“流水线编排”为核心的故事，一种从“算法设计”的视角来看待矩阵乘法 kernel 的编写。除流水线编排之外，我们只使用两种必备的常规优化 —— 2-CTA MMA 与 CTA swizzling —— 便能在 4096 及以上的方阵上接近乃至超越 cuBLAS。事实上，文中介绍的第三版设计在 18 个尺寸的方阵下，有 10 个都跑赢了 cuBLAS，steady-state 性能最高可达 1450T 以上。

希望读完之后你能感受到：在补齐必要的 GPU 架构背景之后，高性能矩阵乘法算子的设计，实质上可以被建模成一个 coherent 的算法设计问题 —— 这也正是它有趣的地方。

## 背景：Tiled GEMM
矩阵乘法的计算模式，天然适合于“分块”（Tiling）这样一种优化方式，即每次加载一小块输入到片上，也只计算一小块输出。这样的好处是提高数据局部性，充分使用每一小块的数据进行计算，减少对于全局内存的冗余访问。这里我们假定读者已对数据局部性、分块等基础背景具有相当的了解，便不再赘述其基本原理，直接探讨分块的大小如何影响流水线的编排。

![分块矩阵乘法：A 的一个行条与 B 的一个列条，产生 C 的一个 tile](https://raw.githubusercontent.com/tongzhou8086/mmc/main/blog/figures/tiled-gemm.png)

如上图所示，分块计算的矩阵乘法有三个维度，它们常常被称为 BM、BN、BK，即，每次加载 BMxBK 大小的 A（称为 A tile），以及 BKxBN 大小的 B（称为 B tile），以此计算 BMxBN 大小的 C 的部分结果（Partial accumulation）。BM、BN、BK 我们统称为分块大小（tile sizes），与此相关的另一个核心概念叫做算术强度（arithmetic intensity），它用来衡量每单位的 memory traffic ，譬如每字节，能够产生的计算量是多少。算术强度是一个软件本身的特征，不同的软件编写方式，会产生不同的算术强度。假如 A tile 和 B tile 都能够完全地存放在片上存储中，即 SMEM（Shared Memory，共享内存）或寄存器，那算术强度的计算方式即为

$$ A.I. = \frac{(2\times BM \times BN \times BK)}{2\times BM \times BK + 2\times BK \times BN}$$

通过简单的数学推导，我们可以看出，BM 和 BN 越大，算术强度就越大。所以在实际的矩阵乘法算子的设计与实现中，我们会尽可能把 BM 和 BN 配得更大一点。但是这里的 trade off 在于片上存储空间是有限的，
譬如对于 Blackwell 而言，能够使用的最大的 SMEM 的大小是 227 KB，而寄存器的总共的容量是 256KB，TMEM（Tensor Memory）的总容量也是 256K。所以实际应用中，BM/BN/BK 的尺寸要远小于 M、N 和 K，Blackwell 上一个常见的配置方案是 BM=128，BN=256，BK=64。

## 背景：Blackwell 的 TMA 、MMA 和 TMEM
流水线的设计，本质上就是编写一个软件，使得这个软件能够高效的对于其背后的硬件进行调度。而需要被调度的硬件单元大概有这么三种：TMA（Tensor Memory Accelerator）、MMA（Matrix Multiply Accumulate）以及 CUDA core 或者 Integer core。TMA 是自 Hopper 架构以后引入的一种独立的硬件单元，用来异步的在内存和 SMEM 之间传输数据，既可以将内存数据加载到 SMEM 中，也可以将 SMEM 中的数据写入到内存。由于是独立的硬件单元，TMA 的运作便不再占用 CUDA Cores 或 integer Cores的算力，而可以独立异步地运行。与此同时，它还硬件支持数据的 swizzling。所以在本文所探讨的所有的流水线的编排方案之中，都会默认使用 TMA 来加载数据以及写入数据。

Blackwell 的 MMA 单元是新一代的 TensorCore Engine，和 TMA 单元类似，MMA 单元也是可以独立异步的运作。从软件的角度，只需要单个 Warp 的一个线程发送 MMA 指令，MMA 单元便可以在背后异步地进行 MMA 运算。其实也正是因为 TMA 和 MMA 单元都是异步的，才会使得流水线的设计大放异彩。

TMEM 也是 Blackwell 引入的一种新的硬件单元，但它是一种存储介质，以一种类似于矩阵的方式组织，有 128 行、512 列，每一个 cell 可以存放一个 float 类型，4 个字节，总容量为 256K。TMEM 用来存放 MMA 的中间结果或最终结果，而且根据 Blackwell 的设计，MMA 的结果只能存放在 TMEM 中。

这些硬件单元都各有自己的设计上的限制，或者是通用性的最佳性能 practice，这也直接影响了我们的流水线设计。具体的限制以及影响，我们在后面具体的流水线设计中再展开。

## 数据流模型
不论流水线怎么编排，数据流动的总体步骤都是一样的，不同方案的区别在于同步方案、数据读写的粒度、tile sizes 的配置、各种 buffer 的数量等等。为了能把这些区别讲清楚，我们先建立一个数据流模型。

流水线的资源调度涉及如下四种 buffer:

* TMA buffer（位于 SMEM）: 存放 TMA 从内存（HBM，High-Bandwidth Memory，也就是常说的 GMEM / Global Memory）中加载的数据
* MMA buffer（位于 TMEM）: 存放 MMA 的中间结果以及最终结果
* tcgen05.ld buffer（位于寄存器，即 RMEM，Register Memory）: 存放从 TMEM 中读取的结果
* Store buffer（位于 SMEM）: 存放要写入内存的数据

这四种 buffer 是数据流模型里的逻辑概念，它们各自的物理载体如下图所示。可以看到 TMA buffer 和 Store buffer 同属 SMEM 介质、要互相抢容量，MMA buffer 则独占 TMEM 空间，而 tcgen05.ld buffer 只是寄存器文件中的一部分 —— 寄存器的其余部分还要用于地址计算、循环变量、各 warp 的私有状态等等。

![四种 buffer 的物理载体](https://raw.githubusercontent.com/tongzhou8086/mmc/main/data-flow-models/figures/sm-storage-map.png)

事实上内存也是一种 buffer，但由于从流水线调度的视角，内存操作并不涉及任何的资源调度策略，所以这里略去不表。

### Buffer 的两种状态：可读或可写
上述的任何一种 buffer 都具有读写互斥性，即同一个 buffer 不能同时被读写，如果生产者的写入和消费者的读取同时进行，则会导致读取错误的数据。于是一个 buffer 总会有两种状态，要么处于“可读”状态，即数据已经就绪，要么处于“可写”状态，即数据已被消费完、可被覆盖。

这种互斥性建立了我们后续要探讨的同步机制的根基。

### 针对 Buffer 的 5 种操作
任何的流水线设计，都会涉及到下述 5 种操作，每一种操作会从一个源 Buffer 读取数据，并将操作后的结果写入目的 Buffer —— 从数据流的角度，可以视为数据从源 Buffer 流入了目的 Buffer。

这 5 种操作分别是：

* TMA load: 从内存读取数据，写入 TMA buffer
* MMA: 从 TMA buffer 读取数据，结果写入 MMA buffer
* tcgen05.ld: 从 MMA buffer 读取数据，写入 tcgen05.ld buffer
* stage: 从 tcgen05.ld 读取数据，写入 stage buffer
* TMA store: 从 stage buffer 读取数据，写入内存

把这 5 种操作逐条列出来，就是下面这张图 —— 每一行是一种操作，箭头两端分别是它读取的源 buffer 和写入的目的 buffer。箭头的起点表示操作开始，**终点则表示操作完成**，也就是说数据要到箭头的终点才真正落在目的 buffer 里。注意同一个 buffer 会出现两次：它既是上一种操作的目的地，也是下一种操作的源头，把相邻的两行接起来，就是数据流过的完整链条：

![5 种操作及其源和目的 buffer](https://raw.githubusercontent.com/tongzhou8086/mmc/main/data-flow-models/figures/operations-chain.png)

第一个操作 TMA load 是从内存中读取用来做矩阵乘法的输入数据，这个好理解；第二个操作，MMA，即是对输入数据进行矩阵乘法操作，这个也好理解；第三个操作是什么呢？事实上，这里的背景是 MMA 的结果，必须保存在 TMEM 中，如果所有的 MMA 都计算完毕，你必须先从 TMEM 中将计算完的结果读取出来、读取到寄存器中才能进行后续的操作，譬如写回内存等等。将数据从 TMEM 中读取到寄存器中使用的指令系列叫做 tcgen05.ld，于是我们把这个操作称为 tcgen05.ld，这是第三个操作；接下来的操作称为 stage，这里又需要一些背景，即按照 Blackwell 架构的设计，通过 tcgen05.ld 读取到寄存器中的数据是按列分布到各线程中的，也就是说，一个线程会拥有同一行上连续的数据。这样的 layout 方式使得，如果你直接将寄存器结果写入内存就会导致 uncoalesced memory access。于是在我们所有的设计中，都会将 tcgen05.ld 的结果先写入一个 SMEM 缓冲区，在缓冲区进行重组，每行能够凑满 128 个连续字节了再进行 coalesced memory write，这也便是最后的两步。

### 一个操作能开始的两个条件
划重点来了！！上述的任何一个操作要能够开始，但必须同时满足以下两个条件：
* 源 Buffer 可读
* 目的 Buffer 可写

这是整个流水线调度设计正确性保障的根本原理。

(对于 TMA load 和 store 操作而言，内存可以视为总是可读或者总是可写)

### 单个 output tile 的时序图
对于单个的 output tile，如果我们假设它的 k 层循环迭代只有一次，即 K = BK，那上述 5 种操作的时序图便会长下面这个样子：

![一个 output tile 的五步操作](https://raw.githubusercontent.com/tongzhou8086/mmc/main/data-flow-models/figures/pipeline-timeline.png)

一个箭头的起点表示操作的开始，终点则表示操作完成，所以箭头的终点也会代表对应 buffer 的状态。譬如，TMA load 箭头的终点则表示对应的 TMA buffer 状态变为“可读”，即 TMA load 操作已完成；与此同时，一种操作的结束也代表其源 buffer 的状态变为“可写”。譬如 MMA 箭头的终点代表一次 BK tile 的 MMA 操作完成，假设 K=BK，这时便会有两个Buffer 的状态都会改变：目的 Buffer 状态变为“可读”，以及源 Buffer 状态变为“可写” —— 数据既然已被消费完毕，那源 Buffer 当然就可以重新写入新的数据喽。

另外，这张图上看不到的部分还包括，它只画出了一个操作能开始的条件之一，即源 Buffer 可读，另一个条件，即目的 Buffer 可写，图上是看不出来的。举个例子，假如一次 MMA 操作进行之前，哪怕对应的 TMA load 操作已经完成，若是将要被写入的 MMA buffer 目前状态并不可写，那 MMA 操作也无法被 issue。我们用下面这张图来表示这种情况，注意 MMA 开始前的那个小的 gap —— 数据在这段时间里一直待在 TMA buffer 里，所以图上把这个 buffer 在 gap 的两端各画了一次：

![同一个 tile，出现了 stall](https://raw.githubusercontent.com/tongzhou8086/mmc/main/data-flow-models/figures/pipeline-timeline-stall.png)

### 减少 MMA issue 的 stall
流水线调度设计的根本主旨是减少 MMA issue 的 stall。

这里实际上有两种不同的 MMA issue stall。一种是一个 output tile 内部多次 K 迭代之间的，这种 stall 我们可以使用多个 TMA buffer 来减少 —— 也就是说，在一次 MMA 操作进行的时候，TMA load 同时也在往另外一个 buffer 里面写入数据，这样等当前的 MMA 操作完成之后，它可以立即从另外一个 buffer 里面继续取数据进行 MMA 操作，而无需等待同一个 buffer。

另一种 MMA issue stall 是连续的多个 output tile 之间的 stall。在连续的多个 output tile 之间，如果要进行 MMA 操作的话，不光是要 TMA 加载的数据到位，同样也还需要 MMA buffer 能够被写入。如果上一个 output tile 的 MMA 结果正在从 MMA buffer 中被读取出来、正在 draining 的过程中，那下一轮的就无法写入，不然就会覆盖数据。针对这样的 stall，我们也有两种解决方案：一种就是使用多个 MMA buffer，譬如两个；另外一种方案就是加快 draining 的过程，通过把数据先暂存到寄存器中，提前释放 MMA buffer。



## Warp Specialization
具体到软件编写的层面，上述的 5 种操作会被映射给不同的 warps，各个操作配置多少 Warps 也是流水线设计的一部分。对于本文所探讨的设计方案而言，均采用如下配置：

* 配一个 Warp 进行 TMA load，又称 TMA Warp
* 配一个 Warp 进行 MMA，又称 MMA Warp
* 配 4 个或者 8 个 warps 进行 tcgen05.ld 和 stage，这些 warps 也被称为 epilogue warps
* TMA Store 操作也顺便由上面的 Epilogue Warps 完成

这样的配置既有原理层面的约束，也是实际经验的结果。先说原理层面，之所以给 TMA load 和 MMA 操作都只配一个 Warp，是因为发射 TMA load 或者 Store 指令以及 MMA 指令，都只需要一个 Warp 的一个线程发出指令就行，所以分别配一个 warp 就行了 —— 发射 TMA load 或者 MMA 指令之前，也还会有一些整数计算，算一下 address offset 之类的，但是这些也都是 scalar 计算，所以理论上其实对于 TMA load 和 MMA 操作，配一个线程其实就够了，只是因为 GPU 的调度单元最小是一个 Warp，所以我们给它配了一个 Warp。

但是后面的 tcgen05.ld 和 stage 操作就不太一样了，这两操作要对数据进行批量操作，需要使用 SIMT 模型进行编程，于是我们要配多个 warps 来提高并行度 —— 4 个 warps 是 GPU 编程的一个标配，有时候我们也会配 8 个 warps 来提高能使用的寄存器数量。另外最后的 TMA Store 操作本质上也只需要配一个线程，我们可以给它单独配一个 Warp。但是经验上我们发现，就让 Epilogue Warps 顺便完成 TMA Store，也挺高效的。因为 TMA Store 是一个比较简单的操作，只需要发射一条 TMA store 指令，加上一点简单的地址计算。
 

## 流水线设计的参数配置与原语
流水线设计的表达是一个程序，它表达的是数据在前文所说的这些 buffer 之间流动的时候的同步规则以及时序设计，与此同时，它也涉及到一些基础的参数配置，这些参数就是流水线这个程序的常量。任何的一种调度方式，至少需要配置如下参数：

* Tile sizes: BM/BN/BK
* 上述的 4 种 Buffer 每种分别配几个？

如前文所述，我们会对同一个类型的 Buffer 配置多个，来提高流水线的重叠程度，从而以减少 stall。在硬件资源（SMEM、TMEM、RMEM 的容量）给定的情况下，每种类型的 Buffer 具体配几个就存在一个设计问题。此外，Tile Sizes 的配置中，BM 的配置是被硬件写死了，Blackwell 架构每次 MMA 支持的尺寸，M 维度必须是 128，所以在本文的所有探讨中，BM 都会被配成 128，没有别的选择。如果输入矩阵的 M 尺寸小于 128 怎么办？那 BM 还是会被配成 128，只是对超出来的部分进行 out of bound 处理。而单次 MMA 操作可以支持的 N 的尺寸可以是 64 或 128 或 256。再者，由于 TMEM 的大小为 128 行乘 512 列，这实质上限制了我们对 Tile Sizes 的配置：BM 最大为 128，BN 最大为 512。所以后文会讨论两种不同的配置，BN 分别配为 256 和 512，它们达成的流水线状态会相当不一样。BK 也是需要配置的一个 Tile size 参数，由于它是 A Tile 的 inner dimension，于是便涉及内存的连续访问问题。对 GPU 内存进行读写，你最好能以 128 个连续字节为单位进行操作，这样就不会浪费内存带宽。于是这里我们会把 BK 配成 64 的整数倍。对于 BF16 数据类型，64 个元素就是 128 字节。

另外，由于 2 CTA MMA 这个优化本身会影响到 tile sizes 以及各类 Buffer 的配置数量，所以我们也提前说明，本文所探讨的所有设计都默认了 2 CTA MMA 开启，于是每个 CTA 只需从内存中读取它实际所需的 BN 的一半，由此也可以看出 2 CTA MMA巨大的功效：除了提高算术强度以外，它还能将每个 CTA 从内存中读取的 B Tile 大小砍半，从而腾出更多的 SMEM 空间来做成 TMA buffer 或者 Store buffer。


所以总结起来就是，对于所有的设计，BM 必须配成 128，BN的配置，我们会讨论两种方案：256 和 512。BK 为 64 的整数倍，再加上 2 CTA MMA 的开启，A Tile 和 B Tile 分别占用的字节数的计算公式如下：

* A tile: BM x BK x 2
* B tile: BK x BN x 2 / 2

B tile 之所以后面会除以 2，是因为我们默认了 2 CTA MMA 的开启。


除了参数配置，流水线设计定义了一套基本操作。不同的调度方案虽然在 tile sizes、buffer 数量、
同步时机上各不相同，但都可以用同一套原语来表达，即一套流水线调度的基本对象，及其对应的
操作与信号；完整定义详见 [docs/pipeline-primitives.md](../docs/pipeline-primitives.md)。
后文的伪代码都基于这套原语书写。

## 第一种设计：BN256
谈流水线设计，我们先确定 BN 的大小，因为 BM 是硬件设计死的，只能是 128，而 BK 不影响算术强度，只是影响数据操作的 granularity，是一个可选参数。而 BN 的大小则是决定了 accumulator 的大小，也影响算术强度。在第一个设计中，我们采用双 MMA buffer，这样在一个 MMA buffer 到了 draining 阶段的时候，与此同时，下一个 output tile 的 MMA 依然能够继续进行，只需要将数据保存在另一个 MMA buffer 中即可，这样便能实现 draining 和 MMA 操作的重叠。由于 TMEM 的大小是 128 行 x 512 列，所以 BN 设为 256，就可以放下两个 MMA buffer。

确定了 BN 的大小，我们再看一下其他 buffer 的配置，这里暂且将 BK 设为 64，于是在使用 2 CTA MMA 的情况下，一个 A Tile 和 B Tile 占用的空间分别是 16KB，一共便是 32KB，也就是一个 TMA buffer 的大小。在这个配置中，我们选择配置 6 个 TMA buffer，加上两个 store buffer，这样占用的 SMEM 空间正好是 192KB+32KB = 224KB，刚好在 227KB 的容量范围内。

tcgen05.ld buffer 的话，我们总是只会配置一个。一方面是因为 tcgen05.ld 是一个比较快的操作，即从 TMEM 中读取数据到 RMEM 中；；另外为了在 epilogue 期间尽早地释放 MMA buffer，我们也会使用大量使用寄存器来暂存 TMEM 数据，所以也配置不了多个。

确定了参数配置，我们再来看各种操作时序逻辑，这里我们使用上面这套原语来表达：

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

从伪代码里也能读出这个设计最关键的一处安排：MMA warp 每换一个 output tile 就切换 MMA buffer，而 epilogue 在把最后一段读进寄存器之后就立刻 `make_free(mma_buffers[acc])`。两者合起来的效果是，一个 output tile 的 draining 只要在**下下个** output tile 开始前完成，MMA 就不会卡住。这个时间窗口和 K 的大小有关：K 越大窗口越大，K 越小则越可能造成 MMA issue 的卡顿。流水线设计的最终目标就是**最小卡顿地吃满 MMA**。

值得注意的是，同样的这一套逻辑也完全能够适配 BK=128 的情况，BK=128 与 BK=64 唯一的区别就是 TMA Buffer 从 6 个变为了 3 个，而逻辑部分完全不变。

### 性能数字

下面我们看一下这个设计在方阵上的性能是多少，事实上我们考虑两种不同的 BK 选配：64/128，以及三种不同的 GROUP_SIZE_M （简称 GSM，代表 CTA swizzle 的深度）选配:8/12/16。不同的 GSM 选配仅仅需要改一个常数参数，而 BK=64/128 的区别也仅仅在于把 TMA buffer 的数量砍半，逻辑部分完全一致。
以下性能数字的测量方法使用 triton.do_bench 获得 median runtime，warmup 和 repetition time 都设置为 1 秒；每个尺寸跑三轮独立的测量，每轮内部再做三次打乱顺序的采样，最后取中位数。

![BN=256 性能对比](https://raw.githubusercontent.com/tongzhou8086/mmc/main/blog/figures/perf-bn256.png)


## 第二种设计：BN512 
BN256 的设计其实已经非常流畅了，TMA buffer 有 6个，应该能够流畅地将数据加载到 SMEM 中，这样 MMA 总是有操作数可以计算；此外，MMA buffer 也有两个，所以如果一个 output tile 的 epilogue 能在下下个 output tile 开始之前完成，理论上我们就可以持续不停的无卡顿地 issue MMA，这已经是一个非常流畅的设计了。

不过，根据实际性能结果，我们发现，哪怕 MMA 的确能够持续不断地在 issue，但是每次只用了一半的 TMEM 做 accumulation 算术强度还是不太够，也就是说，同样的数据量进来，它产生的计算量有限，于是，哪怕 MMA 的 issue 不卡顿，但是实际产生的计算量还是无法吃满 MMA Engine。

于是在第二种设计中，我们换一种新的思路，即，把整个 TMEM 作为一个 MMA buffer 使用，其 BN 是 512，简单的计算我们可以发现，BN=512 下，每个 K tile 的数据量会变为 1.5 倍 —— 从 32KB 变成 48KB，而计算量则会变为两倍，这样会导致同样的数据量进来，能够产生更大的计算量，更有可能能够吃满 MMA engine。但是与此同时，只用了一个逻辑上的 MMA buffer 的话，就会导致在 epilogue draining 期间，后续的 MMA 无法进行 issue，需要等待这个 MMA buffer 被腾空以后才能够 issue 后续的 MMA，这会导致在两个 output tile 中间交接的时候会造成一些卡顿。简而言之，两者的区别总结如下：

**BN256 可以无卡顿地持续 issue MMA，但是产生的 MMA 计算量可能无法吃满硬件算力；BN512 通过加大 accumulation buffer 能够将 MMA 的算力吃得更满，但是与此同时，在两个 output tile 交接的时候会有些卡顿。**

我们先计算一下 A tile 和 B tile 现在分别的大小，BM 依然设为 128，BN 配成 512，BK 先取 64，套用前面的公式：

* A tile: BM × BK × 2 = 128 × 64 × 2 = **16KB**
* B tile: BK × BN × 2 / 2 = 64 × 512 × 2 / 2 = **32KB**

所以一个 TMA buffer（一个 A tile 加一个 B tile）是 **48KB**，比 BN256 时的 32KB 大了 50% —— 大出来的部分全在 B tile 上，因为 BN 翻倍了。

SMEM 的容量决定了能配几个。我们给 TMA buffer 配 **4 个**，共 192KB；store buffer 依然配 **2 个**，每个是 128 × 64 × 2 = 16KB，共 32KB。两者相加还是 224KB。


代码的同步逻辑如下，主要区别在于此时没有两个 MMA buffer 的互相切换了：

```text
# ── 参数 ──────────────────────────────────────────────
NS        = 4              # TMA buffer 个数（BK=128 时为 2）
NUM_ACC   = 1              # BN=512 占满整个 TMEM，只有一个 MMA buffer
NUM_STORE = 2
STORE_N   = 64             # 每次从 TMEM 读出、stage、store 的列数
num_k     = K / BK

# ── Buffer 配置 ───────────────────────────────────────
tma_buffers   = [TMA_Buffer(48KB)   for _ in range(NS)]
mma_buffer    =  MMA_Buffer(256KB)                        # 整块 TMEM，128 x 512
ld_buffer     =  TCGEN05_LD_Buffer(32KB)
store_buffers = [Store_Buffer(16KB) for _ in range(NUM_STORE)]

# ── TMA Warp ─────────────────────────────────────────
gk = 0
for tile in my_output_tiles:
    for k in range(num_k):
        s = (gk++) % NS
        wait_until_free(tma_buffers[s])
        tma_load_async(A, tile.m, k * BK, BM,   BK,     tma_buffers[s], 0)
        tma_load_async(B, k * BK, tile.n, BK,   BN / 2, tma_buffers[s], 16KB)
        make_ready_on_tma_done(tma_buffers[s])

# ── MMA Warp ─────────────────────────────────────────
gk = 0
for tile in my_output_tiles:
    for k in range(num_k):
        s = (gk++) % NS
        wait_until_ready(tma_buffers[s])
        if k == 0:
            wait_until_free(mma_buffer)   # 只有一个 accumulator：必须等上个 tile 彻底 drain 完
        issue_mma_chain_async(mma_buffer, tma_buffers[s], accumulate = (k > 0))
        make_free_on_mma_done(tma_buffers[s])
    make_ready_on_mma_done(mma_buffer)

# ── Epilogue Warps ───────────────────────────────────
gs = 0
for tile in my_output_tiles:
    wait_until_ready(mma_buffer)
    for c in range(BN / STORE_N):            # 一次处理 64 列，一共 8 段
        tcgen05_ld_x32_async(mma_buffer, c * STORE_N, ld_buffer, 0)
        tcgen05_wait_ld()
        if c == BN / STORE_N - 1:
            make_free(mma_buffer)        # 最后一段读完即可释放，让下一个 tile 的 MMA 尽早开始

        b = (gs++) % NUM_STORE
        wait_until_free(store_buffers[b])
        stage(ld_buffer, store_buffers[b])
        make_ready(store_buffers[b])
        tma_store_async(store_buffers[b], C[tile, c])
        make_free_on_tma_done(store_buffers[b])
```

和 BN256 版本对比，结构上其实只差了一处：`wait_until_free` 前面没有了 `acc = tile % NUM_ACC` 这一层轮转。只有一个 accumulator，所以下一个 output tile 的第一次 MMA 必须等到当前 tile 完全 drain 完才能发出，这就是前面说的「卡一下」。epilogue 本身则和 BN256 完全一样，仍然是 8 个 64 列的 chunk 依次读出、stage、store。

### 性能数字

BN512 的上述设计也可以有六种选配，BK=64/128，GSM=8/12/16，如果 BK=64 和 128 分别使用 4 个和 2 个 TMA buffer。性能测量方法与上面相同。

![BN=512 性能对比](https://raw.githubusercontent.com/tongzhou8086/mmc/main/blog/figures/perf-bn512.png)


几点观察：

* 相比 BN256，BN512 在稍大一些的方阵上确实能达到更高的性能。一个可能的解释是，方阵越大 K 维度也越大，Epilogue 所占的时间比例便相应缩小 —— BN512 让计算部分更快，代价是 Epilogue 会卡顿，所以正适合 K 比较大的情况。
* GSM 在这里的作用比 BN256 明显得多：BK=128 在 20480 上从 GSM=8 的 1423 涨到 GSM=16 的 1468；BK=64 在 17408 上从 1316 涨到 1412。
* 最好的一档（BK=128 + GSM=16）在 18 个尺寸中有 10 个跑赢了 cuBLAS。

## 第三种设计：BN512 加强版
在上述的 BN512 基础设计中，epilogue 的部分还是每次从 TMEM 中 load 64 列数据，直到最后一列都 load 完成以后才释放完整的 MMA buffer，这会导致 epilogue 占据 MMA buffer 比较长地时间。在下面的新版设计中，我们会做两个方面的改进。首先，我们加大 tcgen05.ld buffer 的容量，使得它一次能存放下 128 行 x 256 列的数据 —— 共计 128KB tcgen05.ld 的结果，于是 tcgen05.ld buffer 的容量需要扩充到 128KB，即整个 SM 上一半的寄存器容量。我们先把这 128KB 的数据都加载到寄存器中以后，随即释放 MMA buffer 左边的一半，之后再对这 128KB 的数据进行 stage 和 TMA store 操作，以及后续的 load 另一半的 128KB 结果。这样设计的益处在于，MMA buffer 的其中一半（256 列）可以被提前释放，而非要等到整个 MMA buffer 数据都搬完以后才能释放。更早地释放可以使得左边一半的 MMA 可以先开始 issue，只要对应的 k tile 的数据已经到位，这样便能和 epilogue 的后续操作重叠起来，减少 MMA issue 的卡顿。

### 性能数字

这一组我们同样扫了 GSM 8/12/16 与两种 BK，共 6 种选配。

![BN=512 加强版性能对比](https://raw.githubusercontent.com/tongzhou8086/mmc/main/blog/figures/perf-bn512-splitacc.png)

可以看出，在大尺寸上，这种新设计和前面的 BN512 基础版设计性能差不多，大差不差，但是对于相当小一些的方阵的性能提升还是非常显著的，譬如 5120、6144 等。

## 工具
在 Meshy 我们开发了以下两个与此话题相关的工具：
* [MMComposer](https://mmcomposer.streamlit.app/)：一个 web app，上面 host 了各种我们开发的流水线编排设计，kernel 代码清晰易懂，对应的 host code 也有，可以一键下载，直接在 B200 上运行
* [mmc](github.com/tongzhou8086/mmc)：一个高性能的 GEMM 算子库，目前支持 BF16 和 MXFP8；里面集成了各种 kernel 的实现，并且会 auto tune 自动选择最优的 Kernel







