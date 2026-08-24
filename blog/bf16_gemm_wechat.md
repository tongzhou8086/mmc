# Blackwell 矩阵乘法优化分享（一） —— 矩阵乘法与 Blackwell 架构背景

这个系列的文章中，以我们在 B200 上优化矩阵乘法的经历为基础，讲一个以“流水线编排”为核心的故事，一种从“算法设计”的视角来看待矩阵乘法 kernel 的编写。除流水线编排之外，我们只使用两种必备的常规优化 —— 2-CTA MMA 与 CTA swizzling —— 便能在 4096 及以上的方阵上接近乃至超越 cuBLAS。事实上，文中介绍的第四版设计在 18 个尺寸的方阵下，有 12 个都跑赢了 cuBLAS，steady-state 性能最高可达 1450T 以上。

希望读完之后你能感受到：在补齐必要的 GPU 架构背景之后，高性能矩阵乘法算子的设计，实质上可以被建模成一个妙趣横生的算法设计问题。

## 内容提要

这个系列一共会分为四部分，我们会首先介绍一下矩阵乘法、算术强度的概念以及必要的 GPU 架构背景和 2-CTA MMA、CTA swizzling 这两种常规优化，熟悉这部分内容的读者可以自行跳过，这些都属于背景。第二部分中，我们会先进行流水线设计的理论分析篇，从“数据流动”的角度来看待流水线的编排，我们总结出了 4 种 Buffer 和 5 种操作，每一种操作都可以认为是数据从一种 Buffer 流出，最后流进了另一种 Buffer。相当于是操作数放在源 Buffer 中，操作结果放在目的 Buffer 中。以“4 种 Buffer、5 种操作”为根基，我们又自然地推导出了 Buffer 的读写互斥性，即任何一个 Buffer 都无法同时被一个操作读和被另一个操作写，由此又进一步推导出了这 5 种操作之间任意两种要能够同时进行（即重叠起来）的条件。这便构成了流水线同步机制的根本原理。

之后我们以单 buffer 为例，阐述流水线的时间线，从而看出两类会导致流水线上 MMA issue 停顿的原因，而它们恰好就是两种经典的数据依赖：源 Buffer 不可读造成的 **RAW 停顿**，即一个输出块（output tile）内部由于需要等待 TMA load 数据的就绪；以及目的 Buffer 不可写造成的 **WAR 停顿**，即多个 output tile 之间由于需要等待 MMA buffer 中的数据完成 draining 才能再次写入。针对这两类停顿，我们分别给出了解决方案。
后面第三和第四部分是实现篇，它们实现篇会直接继承理论分析的结果，将它们转化成对应的 CUDA Kernel（文中是以伪代码的形式呈现）。作为例子，我们会呈现四种不同的设计方案，围绕着如何一步步减少上述 RAW、WAR 这两类停顿而展开。

## 背景篇

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

### 数据流经的 4 种片上容器（buffer）
我们上述的伪代码是比较宏观的和架构无关的，但在具体的流水线设计中，我们则会引入以下五种 Blackwell 架构特定的操作以及五种逻辑容器，每一种操作会从一种容器里读入数据，然后结果会写入另一种容器。

![5 种操作及其源和目的 buffer](https://raw.githubusercontent.com/tongzhou8086/mmc/main/data-flow-models/figures/operations-chain.png)

其中，TMA load、MMA 以及最后的 TMA store 操作可以直接对应到上述伪代码中的`load`，`a_tile @ b_tile`和`store(C, tile.m, tile.n, acc)`部分。而剩下的两种操作 tcgen05.ld 和 stage，则是 Blackwell 特定的、上述架构无关的伪代码中没有体现出来的。它们存在的原因是因为，根据 Blackwell 架构设计，MMA 的结果必须存储在 TMEM 中，而 TMEM 中的结果必须先通过 tcgen05.ld 系列指令读取到寄存器中以后才能进行后续操作，譬如 epilogue 或写回内存。于是，这里我们便多了一个 tcgen05.ld 操作。与此同时，由于 tcgen05.ld 是按列进行数据读取的，如果将它们的结果直接写入内存，就会导致 uncoalesced memory write。于是我们会先把寄存器中的数据，在 store buffer / SMEM 中做一个临时的缓冲与重组，当一行凑满连续的 128 个字节了，再进行内存写入，直接 issue 一个 TMA store 指令，便能达到 coalesced memory write 的效果。这便是 stage 操作的来历。

> 值得说明的是，我们不保证所有情况下使用 TMA store buffer 进行缓冲后再写入内存都是最高效的，另一种不同的设计完全可以为了节省 SMEM 空间而直接进行 uncoalesced memory write，本文这里探讨的是一种比较通用的设计框架，不保证在任何情况下都是最高性能，但是是比较通用的。



四种 Blackwell 架构相关的片上容器，我们先看这四种容器分别是什么，然后再和以上的伪代码对应起来。

* TMA buffer: 存放 TMA 从内存（GPU global memory）中加载的数据，对应上述代码的 `load` 部分
* MMA buffer: 存放 MMA 的中间结果以及最终结果
* tcgen05.ld buffer: 存放从 TMEM 中读取的结果，即 MMA 的结果
* Store buffer: 存放要写入内存的数据，即 MMA 的结果做过一些重排或者特定 epilogue 后的状态

由于硬件限制，上述 4 种片上容器的物理存储介质分别是，TMA buffer 和 Store buffer 都使用 SMEM（TMA 单元通过 SMEM 与内存交换数据），MMA buffer 使用 TMEM（MMA 的结果必须存放在 TMEM 中），而 tcgen05.ld buffer 使用寄存器（TMEM 中的结果必须先读取到寄存器中才能进行后续操作）。从一致性的角度来讲，TMA buffer 和 Store buffer 更完整的名称应该分别叫做 TMA load buffer 以及 TMA store buffer，本文我们将它们统一简称为 TMA buffer 以及 Store buffer。

值得一提的是，For completeness，内存也是一种 buffer，数据会最初来自于内存，最后又流入内存。但由于从流水线资源调度的视角，内存并不参与，所以这里略去不表。

![四种 buffer 的物理载体](https://raw.githubusercontent.com/tongzhou8086/mmc/main/data-flow-models/figures/sm-storage-map.png)

### 针对 Buffer 的五种操作（operation）
从上述我们提供的代码宏观框架中，我们可以抽象出 5 种操作，每种操作分别有一个源容器和目的容器，即，该操作从源容器中读取数据，而操作的结果被写入目的容器。这五种操作分别是：

* TMA load: 从内存读取数据，写入 TMA buffer
* MMA: 从 TMA buffer 读取数据，结果写入 MMA buffer
* tcgen05.ld: 从 MMA buffer 读取数据，写入 tcgen05.ld buffer
* stage: 从 tcgen05.ld 读取数据，写入 store buffer
* TMA store: 从 store buffer 读取数据，写入内存

前面两种操作 TMA load 和 MMA 的意义一目了然。第三步之所以需要 tcgen05.ld 操作，是因为根据 Blackwell 的设计 TMEM 中的计算结果必须要先搬运到寄存器中以后才能进行后续操作，譬如写回内存。这个搬运操作的指令系列叫做 [tcgen05.ld](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=tcgen05%2520ld#tcgen05-matrix-fragments-shape-3232b)，于是我们才把这个操作称为 tcgen05.ld。此外，stage 操作的意义在于，如果将 tcgen05.ld 到寄存器中的结果（不同的线程按列读取）直接写入内存，会导致 uncoalesced memory access，即同一个 Warp 中的不同线程会按列写入。于是我们会先将寄存器中的结果先写入 SMEM，做一个临时的缓冲与重组。当一行凑满连续的 128 个字节了，再进行内存写入，直接 issue 一个 TMA store 指令，便能达到 coalesced memory write 的效果。

