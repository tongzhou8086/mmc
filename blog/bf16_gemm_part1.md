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

再一个问题是，软件算法的算术强度一定要提到最高吗？会不会到什么时候就已经足够高了？再提高也没有效果？事实上，每一个 GPU 硬件都会有一个自己的 [Roofline](https://en.wikipedia.org/wiki/Roofline_model) 模型，以及一个对应的平衡点（balance point），当运行在该硬件上的软件的算术强度在平衡点之下时，软件的性能便被称为 Memory Bound，即受限于内存/L2带宽；而当软件的算术强度高于平衡点时，整体性能便被称为 Compute Bound，即受限于计算能力，对于现代 GPU 而言，MMA engine（或称 Tensor Core Engine）的算力则代表了硬件的计算能力。事实上，对于不同的硬件，要想达到它们的平衡点的难度也不一样，这取决于硬件具体的内存带宽、L2 缓存带宽以及 MMA 算力的比例关系。其次，内存和 L2 缓存会各自有一个自己的平衡点，而对实际运行一个矩阵乘法算子而言，真正实际有效的平衡点则是二者的混合，这也是精准的理论分析没那么容易的原因。不过在后续的性能实测里面，我们会看到，从 BN256 到 BN512 带来的理论上的算术强度的提升，的确显著体现在了实际的性能提升上。



### Blackwell 的异步设计 —— 软件成为调度者
流水线的调度设计，本质上就是编写一个软件，使得这个软件能够高效的对于其背后的硬件进行调度。而需要被调度的硬件计算/数据搬运单元大概有这么三种：TMA（Tensor Memory Accelerator）、MMA（Matrix Multiply Accumulate）以及 CUDA core 或者 Integer core。TMA 是自 Hopper 架构以后引入的一种独立的硬件单元，用来异步的在内存和 SMEM 之间传输数据，既可以将内存数据加载到 SMEM 中，也可以将 SMEM 中的数据写入到内存。由于是独立的硬件单元，TMA 的运作便不再占用 CUDA Cores 或 integer Cores的算力，而可以独立异步地运行。与此同时，它还硬件支持数据的 swizzling。所以在本文所探讨的所有的流水线的编排方案之中，都会默认使用 TMA 来加载数据以及写入数据。相比传统的 SIMT 式的数据搬运方式，即所有线程都需要参与，使用 TMA 只需要一个 warp 的一个线程发出 [TMA 指令](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#using-the-tensor-memory-accelerator-tma) 即可，也称为 bulk load —— 批量加载。

Blackwell 的 MMA 单元则是新一代的 Tensor Core Engine，和 TMA 单元类似，他们都处于一个 SM 内部，都是可以独立异步运作的硬件单元。从软件的角度，也只需要一个 warp 的一个线程发送 [MMA 指令](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/mma/tcgen05_programming.html)，MMA 单元便可以在背后异步地进行 MMA 运算。其实，也正是因为 TMA 和 MMA 单元都是异步的，才会使得流水线的设计大放异彩 —— 软件的功能更趋近于一个“调度者”的角色，而很多的操作都是专门的硬件在背后异步地完成。

[TMEM](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=Tensor%2520memory#tensor-memory) 也是 Blackwell 引入的一种新的硬件单元，但它是一种存储介质，以一种类似于矩阵的方式组织，有 128 行、512 列，每一个 cell 可以存放一个 float 类型，4 个字节，总容量为 256K。根据 Blackwell 的设计，MMA 的结果只能存放在 TMEM 中，而操作数可以放在 SMEM 或 TMEM 中。不过，在我们后续的讨论中，我们都会默认操作数就放在 SMEM 中，即 MMA 单元从 SMEM 中读取数据，在 TMEM 中存放累加的结果。

上述的硬件单元的设计限制，就成了我们流水线设计的硬性限制，影响了各种 trade off。这里我们仅仅论述 Blackwell 新引入的硬件特性，而像传统的 GPU 架构内容，我们假设读者已经熟悉，不再赘述。

### 2-CTA MMA

Blackwell 的 tcgen05 MMA 指令支持一种叫做 `cta_group::2` 的模式：相邻的两个 CTA 组成一个 cluster，共同完成同一条 MMA 指令。这带来的变化如下图所示。

左边是不开启的情形：两个 CTA 各自独立地计算一块 BMxBN 的输出，各自都需要完整的 B tile —— 于是同一份 B 数据被从内存里读了两遍。右边是开启之后：两个 CTA 合起来计算一块 2BMxBN 的输出，A 天然地被切成两半（每个 CTA 各持有自己的那 BM 行），而 B 也被切成两半（每个 CTA 只持有 BN/2 列），MMA 单元则跨 cluster 去读取另一个 CTA 的那一半。同样大小的输出，B 只被读了一遍。

![2-CTA MMA 开启前后：B tile 从读两遍变成只读一遍](https://raw.githubusercontent.com/tongzhou8086/mmc/main/blog/figures/two-cta-mma.png)

这一条优化带来两个后果，后文都会反复用到：

* **算术强度提高**。回到前面那个算术强度的公式，分母里 B tile 的那一项从 `BK x BN x 2` 变成了 `BK x BN x 2 / 2`，同样的计算量只需要一半的 B 流量。
* **省出了 SMEM**。B 的驻留空间少了一半，腾出来的容量可以用来多配几个 TMA buffer，或者把 tile 配得更大 —— 这正是后文流水线设计里可以支配的资源。


### CTA Swizzle（相当于 L2 缓存分块）

一个 GEMM kernel 会把输出矩阵切成许多 output tile，再按某种顺序分配给各个 CTA（后文的「程序的宏观框架」一节会给出完整的框架）。这个顺序不影响计算结果，但它决定了**同一时刻正在运行的那一批 CTA 会碰到哪些数据** —— 而这直接决定了 L2 的命中率。

最直观的分配方式是 row-major，即按行依次往下发。问题在于，同时在跑的这一批 CTA 会落在同一行上，它们共享同一条 A 的行条，却各自需要一条不同的 B 的列条。换一种走法：先沿着一列往下走 GROUP_SIZE_M 个 tile，再换到下一列，走满一批之后再重新开始 —— 同一批 CTA 就落在一个比较方的矩形里，需要的 A 行条虽然变多了，B 列条却少得多。

![row-major 与 grouped 两种分配顺序](https://raw.githubusercontent.com/tongzhou8086/mmc/main/blog/figures/cta-swizzle.png)

图中以 8x8 的 tile 网格、一批 8 个 CTA 为例：row-major 走法下，这 8 个 tile 需要 1 条 A 加 8 条 B，共 9 条；而 grouped 走法（GROUP_SIZE_M=4）下只需要 4 条 A 加 2 条 B，共 6 条。同样的计算量，需要同时驻留在 L2 里的数据少了三分之一。数据能命中 L2，TMA load 就回来得更快，MMA 也就更不容易因为等数据而停顿。

GROUP_SIZE_M（后文简称 GSM）就是这里唯一的旋钮。它也不是越大越好：分组越深，一批 CTA 需要的 A 行条就越多，工作集重新变大，而且在矩阵边缘还会出现凑不满一组的情况。后文的性能测试里我们会实测 GSM = 8 / 12 / 16 三档。

### 程序的宏观框架

在介绍流水线编排模型之前，我们先放出代码的宏观框架，以便为读者建立一个宏观的认知：每一个 CTA 在计算什么？首先在逻辑上我们先把 MxN 大小的输出矩阵按照 BMxBN 的块大小划分，即，划分成 ceil(M/BM) 行和 ceil(N/BN) 列，每一块的大小是 BMxBN（当 M、N 除不尽时，边缘的那一行/一列块会算不满，需要额外的边界处理，这里不展开）。这样的一个块，我们也将它称为 output tile，计算它对应的输入数据则被称为 input tile 或者 A tile 和 B tile。为方便后文引用，记 output tile 的总数为 `num_tiles = ceil(M/BM) * ceil(N/BN)`。

一种简单的矩阵乘法实现方式是让一个 CTA 计算一个 output tile，这样需要 launch 的 CTA 的总数就正好是 num_tiles，即每一个 CTA 完成一个 output tile 的 accumulation 之后，将它写入内存，这个 CTA 便退出了，空出来的 SM 再由硬件调度下一个尚未开始的 CTA 上来执行。这样的问题在于，将 accumulation 结果写回内存的过程，没有和新一轮的计算重叠起来。所以实际上在高性能的矩阵乘法实现中，都会让一个 CTA 计算多个 output tile，这样在前一个 output tile 完成了 accumulation，在写入内存的同时，下一个 output tile 的计算便可以同时运行。举个例子，譬如 GPU 上有多少个 SM，我们就可以 launch 多少个 CTA（记作 num_CTA），再把 num_tiles 个 output tile 依次分给它们。这样的分配方式又被称为 persistent kernel，即这些 CTA 常驻在 SM 上，是 persistent 的，连着算许多个 output tile。

> 在这种分配下，每个 CTA 拿到的 output tile 数量并不一定相等：num_tiles 未必能被 num_CTA 整除，前 num_tiles mod num_CTA 个 CTA 会多分到一个。外层循环的最大迭代次数、也即整个 kernel 的时长，是由多算一个的那些 CTA 决定的，即 ceil(num_tiles / num_CTA)。当 num_tiles 不是 num_CTA 的整数倍时，最后一轮里就有一部分 SM 是闲着的，这个损失通常被称为 [wave quantization](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#wave-quant)。

注意，在这样 persistent kernel 的 CTA 分配方式下，每个 CTA 到底被分配到哪些 output tiles 依然存在一个分配问题，即上面所说的 CTA swizzle 和一个 CTA 算多个 output tile 是一个正交的关系：CTA swizzle 决定的是 output tile 的**遍历顺序**，persistent kernel 决定的是把这个顺序上的第 i 个 tile 交给**哪一个 CTA**（最直接的做法就是交给第 i mod num_CTA 个 CTA）。如果假设一共有 8 个 SM，然后我们使用 8 个常驻 CTA，总共合起来计算 8x8 = 64 个 output tiles，每个 CTA 正好分到 8 个。那结合了 persistent kernel 以及 CTA swizzle 二者之后，每个 CTA 分配到的 output tiles 则如下图所示，左边是 row-major 顺序，右边是 GSM=4 的 grouped 顺序。

![persistent grid 与 CTA swizzle 结合后的 output tile 分配](https://raw.githubusercontent.com/tongzhou8086/mmc/main/blog/figures/persistent-swizzle.png)

这个 persistent kernel 的部分，我们在以下伪代码中称为外层循环，即每一次迭代计算一个不同的 output tile。与此同时，还会有一个内层循环，因为每计算一个 output tile 便需要分成 ceil(K/BK) 步完成计算，即每次加载 BMxBK 的 A 数据，和 BKxBN 的 B 数据，我们也把这个内层循环称为 k tile 迭代。

这两层循环用伪代码描述如下：

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

> 一点说明：上面为了叙述简洁，都是以「一个 CTA 算一个 output tile」来描述的。若开启前面讲的 2-CTA MMA，则应把这里的「一个 CTA」理解成「一个 2-CTA cluster」—— 分配的单位是 cluster，实际 launch 的 CTA 数是上述数字的两倍，一个 cluster 负责的输出块高度也变成 2BM。

外层循环的每一轮产出一整块 output tile，内层循环的每一轮只推进 BK 这一步 —— 后文所有的流水线设计，本质上都是在给这两层循环里的 load、MMA、epilogue 安排先后顺序和重叠方式，而循环结构本身是不变的。在下一部分中，我们会介绍流水线的每一个阶段，以及各自操作的容器，和因此产生的依赖关系。
