# B200 矩阵乘法优化分享

本文以我们在 B200 上优化矩阵乘法的经历为基础，讲一个以“流水线编排”为核心的故事，一种从“算法设计”的视角来看待矩阵乘法 kernel 的编写。除流水线编排之外，我们只使用两种必备的常规优化 —— 2-CTA MMA 与 CTA swizzling —— 便能在 4096 及以上的方阵上接近乃至超越 cuBLAS。事实上，文中介绍的第三版设计在 18 个尺寸的方阵下，有 10 个都跑赢了 cuBLAS，steady-state 性能最高可达 1450T 以上。

希望读完之后你能感受到：在补齐必要的 GPU 架构背景之后，高性能矩阵乘法算子的设计，实质上可以被建模成一个妙趣横生的算法设计问题。

## 内容提要

本文会首先介绍一下矩阵乘法、算术强度的概念以及必要的 GPU 架构背景和 2-CTA MMA、CTA swizzling 这两种常规优化，熟悉这部分内容的读者可以自行跳过。这些都属于背景，背景之后，我们会先进行流水线设计的理论分析篇，从“数据流动”的角度来看待流水线的编排，我们总结出了 4 种 Buffer 和 5 种操作，每一种操作都可以认为是数据从一种 Buffer 流出，最后流进了另一种 Buffer。相当于是操作数放在源 Buffer 中，操作结果放在目的 Buffer 中。以“4 种 Buffer、5 种操作”为根基，我们又自然地推导出了 Buffer 的读写互斥性，即任何一个 Buffer 都无法同时被一个操作读和被另一个操作写，由此又进一步推导出了这 5 种操作之间任意两种要能够同时进行（即重叠起来）的条件。这便构成了流水线同步机制的根本原理。

之后我们以单 buffer 为例，阐述流水线的时间线，从而看出两类会导致流水线上 MMA issue 卡顿的原因：输入 Buffer 不可读造成的的卡顿，即一个输出块（output tile）内部由于需要等待 TMA load 数据的就绪；以及输出 Buffer 不可写造成的卡顿，即多个 output tile 之间由于需要等待 MMA buffer 中的数据完成 draining 才能再次写入。针对两种类型的卡顿，我们分别给出了解决方案。
最后第三章是实现篇，实现篇会直接继承理论分析的结果，将它们转化成对应的 CUDA Kernel（文中是以伪代码的形式呈现）。作为例子，我们会呈现三种不同的设计方案，三种设计方案其实还是围绕着减少上述两种卡顿而展开。



## 背景篇

### Tiled GEMM
矩阵乘法的计算模式，天然适合于“分块”（Tiling）这样一种优化方式，即每次加载一小块输入到片上，也只计算一小块输出。这样的好处是提高数据局部性，充分使用每一小块的数据进行计算，减少对于全局内存的冗余访问。这里我们假定读者已对数据局部性、分块等基础背景具有相当的了解，便不再赘述其基本原理，直接探讨分块的大小如何影响流水线的编排。

![分块矩阵乘法：A 的一个行条与 B 的一个列条，产生 C 的一个 tile](https://raw.githubusercontent.com/tongzhou8086/mmc/main/blog/figures/tiled-gemm.png)

如上图所示，分块计算的矩阵乘法有三个维度，它们常常被称为 BM、BN、BK，即，每次加载 BMxBK 大小的 A（称为 A tile），以及 BKxBN 大小的 B（称为 B tile），以此计算 BMxBN 大小的 C 的部分结果（Partial accumulation）。BM、BN、BK 我们统称为分块大小（tile sizes），与此相关的另一个核心概念叫做算术强度（arithmetic intensity），它用来衡量每单位的 memory traffic ，譬如每字节，能够产生的计算量是多少。算术强度是一个软件本身的特征，不同的软件编写方式，会产生不同的算术强度。假如 A tile 和 B tile 都能够完全地存放在片上存储中，即 SMEM（Shared Memory，共享内存）或寄存器，那算术强度的计算方式即为

$$ A.I. = \frac{(2\times BM \times BN \times BK)}{2\times BM \times BK + 2\times BK \times BN}$$

通过简单的数学推导，我们可以看出，BM 和 BN 越大，算术强度就越大。所以在实际的矩阵乘法算子的设计与实现中，我们会尽可能把 BM 和 BN 配得更大一点，只要能放得下。不过片上存储空间毕竟有限，你不可能使得 BM 和 BN 无限大。对于 Blackwell 而言，能够使用的最大的 SMEM 的大小是 227 KB，而寄存器的总共的容量是 256KB，TMEM（Tensor Memory）的总容量也是 256KB，这些都限制了 BM、BN、BK 的配置大小。在实际应用中，一个常见的配置方案是 BM=128，BN=256，BK=64 或 128。

再一个问题是，软件算法的算术强度一定要提到最高吗？会不会到什么时候就已经足够高了？再提高也没有效果？事实上，每一个 GPU 硬件都会有一个自己的 [Roofline](https://en.wikipedia.org/wiki/Roofline_model) 模型，以及一个对应的平衡点（balance point），当运行在该硬件上的软件的算术强度在平衡点之下时，软件的性能便被称为 Memory Bound，即受限于内存/L2带宽；而当软件的算术强度高于平衡点时，整体性能便被称为 Compute Bound，即受限于计算能力，对于现代 GPU 而言，MMA engine（或称 Tensor Core Engine）的算力则代表了硬件的计算能力。事实上，对于不同的硬件，要想达到它们的平衡点的难度也不一样，这取决于硬件具体的内存带宽、L2 缓存带宽以及 MMA 算力的比例关系。其次，内存和 L2 缓存会各自有一个自己的平衡点，而对实际运行一个矩阵乘法算子而言，真正实际有效的平衡点则是二者的混合，这也是精准的理论分析没那么容易的原因。不过在后续的性能实测里面，我们会看到，从 BN256 到 BN512 带来的理论上的算术强度的提升，的确显著体现在了实际的性能提升上。



### Blackwell 的 TMA 、MMA 和 TMEM
流水线的设计，本质上就是编写一个软件，使得这个软件能够高效的对于其背后的硬件进行调度。而需要被调度的硬件单元大概有这么三种：TMA（Tensor Memory Accelerator）、MMA（Matrix Multiply Accumulate）以及 CUDA core 或者 Integer core。TMA 是自 Hopper 架构以后引入的一种独立的硬件单元，用来异步的在内存和 SMEM 之间传输数据，既可以将内存数据加载到 SMEM 中，也可以将 SMEM 中的数据写入到内存。由于是独立的硬件单元，TMA 的运作便不再占用 CUDA Cores 或 integer Cores的算力，而可以独立异步地运行。与此同时，它还硬件支持数据的 swizzling。所以在本文所探讨的所有的流水线的编排方案之中，都会默认使用 TMA 来加载数据以及写入数据。相比传统的 SIMT 式的数据搬运方式，即所有线程都需要参与，使用 TMA 只需要一个 warp 的一个线程发出 TMA 指令即可，也称为 bulk load —— 批量加载。

Blackwell 的 MMA 单元则是新一代的 Tensor Core Engine，和 TMA 单元类似，他们都处于一个 SM 内部，都是可以独立异步运作的硬件单元。从软件的角度，也只需要一个 warp 的一个线程发送 MMA 指令，MMA 单元便可以在背后异步地进行 MMA 运算。其实，也正是因为 TMA 和 MMA 单元都是异步的，才会使得流水线的设计大放异彩 —— 软件的功能更趋近于一个“调度者”的角色，而很多的操作都是专门的硬件在背后异步地完成。

TMEM 也是 Blackwell 引入的一种新的硬件单元，但它是一种存储介质，以一种类似于矩阵的方式组织，有 128 行、512 列，每一个 cell 可以存放一个 float 类型，4 个字节，总容量为 256K。根据 Blackwell 的设计，MMA 的结果只能存放在 TMEM 中，而操作数可以放在 SMEM 或 TMEM 中。不过，在我们后续的讨论中，我们都会默认操作数就放在 SMEM 中，即 MMA 单元从 SMEM 中读取数据，在 TMEM 中存放累加的结果。

上述的硬件单元的设计限制，就成了我们流水线设计的硬性限制，影响了各种 trade off。这里我们仅仅论述 Blackwell 新引入的硬件特性，而像传统的 GPU 架构内容，我们假设读者已经熟悉，不再赘述。

### 2-CTA MMA

Blackwell 的 tcgen05 MMA 指令支持一种叫做 `cta_group::2` 的模式：相邻的两个 CTA 组成一个 cluster，共同完成同一条 MMA 指令。这带来的变化如下图所示。

左边是不开启的情形：两个 CTA 各自独立地计算一块 BMxBN 的输出，各自都需要完整的 B tile —— 于是同一份 B 数据被从内存里读了两遍。右边是开启之后：两个 CTA 合起来计算一块 2BMxBN 的输出，A 天然地被切成两半（每个 CTA 各持有自己的那 BM 行），而 B 也被切成两半（每个 CTA 只持有 BN/2 列），MMA 单元则跨 cluster 去读取另一个 CTA 的那一半。同样大小的输出，B 只被读了一遍。

<img width="751" height="396" alt="图片" src="https://github.com/user-attachments/assets/e80a51c8-aac1-45b0-991c-74ec8d791fce" />

这一条优化带来两个后果，后文都会反复用到：

* **算术强度提高**。回到前面那个算术强度的公式，分母里 B tile 的那一项从 `BK x BN x 2` 变成了 `BK x BN x 2 / 2`，同样的计算量只需要一半的 B 流量。
* **省出了 SMEM**。B 的驻留空间少了一半，腾出来的容量可以用来多配几个 TMA buffer，或者把 tile 配得更大 —— 这正是后文流水线设计里可以支配的资源。


### CTA Swizzle

一个 GEMM kernel 会把输出矩阵切成许多 output tile，再按某种顺序分配给各个 CTA（后文的「程序的宏观框架」一节会给出完整的框架）。这个顺序不影响计算结果，但它决定了**同一时刻正在运行的那一批 CTA 会碰到哪些数据** —— 而这直接决定了 L2 的命中率。

最直观的分配方式是 row-major，即按行依次往下发。问题在于，同时在跑的这一批 CTA 会落在同一行上，它们共享同一条 A 的行条，却各自需要一条不同的 B 的列条。换一种走法：先沿着一列往下走 GROUP_SIZE_M 个 tile，再换到下一列，走满一批之后再重新开始 —— 同一批 CTA 就落在一个比较方的矩形里，需要的 A 行条虽然变多了，B 列条却少得多。

![row-major 与 grouped 两种分配顺序](https://raw.githubusercontent.com/tongzhou8086/mmc/main/blog/figures/cta-swizzle.png)

图中以 8x8 的 tile 网格、一批 8 个 CTA 为例：row-major 走法下，这 8 个 tile 需要 1 条 A 加 8 条 B，共 9 条；而 grouped 走法（GROUP_SIZE_M=4）下只需要 4 条 A 加 2 条 B，共 6 条。同样的计算量，需要同时驻留在 L2 里的数据少了三分之一。数据能命中 L2，TMA load 就回来得更快，MMA 也就更不容易因为等数据而停顿。

GROUP_SIZE_M（后文简称 GSM）就是这里唯一的旋钮。它也不是越大越好：分组越深，一批 CTA 需要的 A 行条就越多，工作集重新变大，而且在矩阵边缘还会出现凑不满一组的情况。后文的性能测试里我们会实测 GSM = 8 / 12 / 16 三档。

### 程序的宏观框架

在介绍流水线编排模型之前，我们先放出代码的宏观框架，以便为读者建立一个宏观的认知：我们在编什么程。以 CUDA 编程语言为例，众所周知，在 CUDA 编程中，我们需要指明每个线程的行为，同时一个 CTA 中的线程又能通过 SMEM 协作、交换数据等等。于 GEMM kernel 的表达而言，更加自然的方式是以 CTA 为视角来描述；具体在 CUDA 的层面，则需要再映射为每一个线程的操作。我们的程序框架如下：

GEMM 程序不是最终会计算出一个 MxN 的矩阵的输出结果嘛，在逻辑上我们先把这个输出矩阵按照 BMxBN 的块大小划分，即，划分成 M/BM 行和 N/BN 列，每一块的大小是 BMxBN。这样的一个块，我们也将它称为 output tile，计算它对应的输入数据则被称为 input tile 或者 A tile 和 B tile。

每个 CTA 会计算不止一个 output tile，事实上，它们会使用一个外层循环，在循环里依次计算分配给它们的每个 output，具体 output tiles 是如何分配给 CTA 的，就存在一个分配问题，也就和 CTA swizzle 相关，下图给出了一种可能的分配方式的图示。

![8 个 CTA 与 output tile 的一种分配方式](https://raw.githubusercontent.com/tongzhou8086/mmc/main/blog/figures/cta-assignment.png)

上图中一共 8 个 CTA，分配的顺序是先沿着一列往下走完 4 个 tile，再走下一列，走满 8 个之后又从 CTA0 重新开始 —— 所以这个分配模式每两列重复一次。不同的分配顺序会决定相邻的 CTA 之间共享哪些 A tile 和 B tile，也就直接影响 L2 的复用效率，这正是后文 CTA swizzle 要调的东西。

既然存在一个外层循环用来计算不同的 output tiles，那自然也有个内层循环了。事实上，计算一个 output tile 也不是一步到位的，而是会每次加载一部分的 A tile 和 B tile，即每次加载 BMxBK 的 A 数据，和 BKxBN 的 B 数据 —— 即，一个 BK tile，然后分成 K/BK 步完成计算，这便是内层循环的迭代，我们也称它为 k tile 迭代。

这两层循环用伪代码描述如下：

```text
num_k = K / BK                               # 每个 output tile 需要多少次 k 迭代

# my_output_tiles：分配给当前 CTA 的那一批 output tile，分配方式见上图
for tile in my_output_tiles:                 # ── 外层循环：遍历 output tile ──

    acc = zeros(BM, BN)                      # 这个 output tile 的累加器

    for k in range(num_k):                   # ── 内层循环：遍历 k tile ──
        a_tile = load(A, tile.m, k)          #   BM x BK
        b_tile = load(B, k, tile.n)          #   BK x BN
        acc += a_tile @ b_tile               #   一次 BK 的部分累加

    store(C, tile.m, tile.n, acc)            # K 维度累加完毕，写回这块 BM x BN 的结果，又称为 epilogue
```

外层循环的每一轮产出一整块 output tile，内层循环的每一轮只推进 BK 这一步 —— 后文所有的流水线设计，本质上都是在给这两层循环里的 load、MMA、epilogue 安排先后顺序和重叠方式，而循环结构本身是不变的。要探讨具体的操作之间的编排方式，譬如谁和谁重叠、如何保证结果正确，则有必要先引入一套理论模型，便于我们分析操作的性质和需要满足的时序关系。

## 理论篇

不论流水线怎么编排，数据流动的总体步骤都是一样的，不同方案的区别在于同步方案、数据读写的粒度、tile sizes 的配置、各种 buffer 的数量等等。为了能把这些区别讲清楚，我们先建立一个数据流模型。

### 四种不同的 buffer 资源

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

把这 5 种操作串起来看，就是下面这张图 —— 每个箭头是一种操作，箭头两端则是它读取的源 buffer 和写入的目的 buffer：

![5 种操作及其源和目的 buffer](https://raw.githubusercontent.com/tongzhou8086/mmc/main/data-flow-models/figures/operations-chain.png)

第一个操作 TMA load 是从内存中读取用来做矩阵乘法的输入数据，这个好理解；第二个操作，MMA，即是对输入数据进行矩阵乘法操作，这个也好理解；第三个操作是什么呢？事实上，这里的背景是 MMA 的结果必须保存在 TMEM 中，当一个 output tile 所有的 MMA 都计算完毕，你必须先从 TMEM 中将计算完的结果读取到寄存器中才能进行后续的操作，譬如写回内存等等。将数据从 TMEM 中读取到寄存器中使用的指令系列叫做 [tcgen05.ld](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=tcgen05%2520ld#tcgen05-matrix-fragments-shape-3232b)，于是我们把这个操作称为 tcgen05.ld，这是第三个操作；接下来的操作称为 stage，这里又需要一些背景，即按照 Blackwell 架构的设计，通过 tcgen05.ld 读取到寄存器中的数据是[按列分布](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=tcgen05%2520ld#tcgen05-matrix-fragments-shape-3232b)到各线程中的，也就是说，一个线程会拥有同一行上连续的数据。这样的 layout 方式使得，如果你直接将寄存器结果写入内存就会导致一个 warp 中不同线程按列写入，即 uncoalesced memory access。于是在我们所有的设计中，都会将 tcgen05.ld 的结果先写入一个 SMEM 缓冲区，在缓冲区进行重组，每行能够凑满 128 个连续字节了再进行 coalesced memory write，这也便是最后的两步。

### 操作之间同步的法则
划重点来了！！上述的任何一个操作要能够开始，但必须同时满足以下两个条件：
* 源 Buffer 可读
* 目的 Buffer 可写

这是整个流水线调度设计正确性保障的根本原理。

> 对于 TMA load 和 store 操作而言，内存可以视为总是可读或者总是可写

延伸一下，我们可以推导出，任意两个操作要能够同时运行的条件：

* 其中一个操作的源 buffer 不是另一个操作的目的 buffer

### 单 buffer 的流水线时序图
根据上面的同步法则，我们可以推导出在所有 buffer 只配置一份的情况的流水线的时序图，即：

* TMA load 不能与 MMA 同时运行，但是可以和 tcgen05.ld 及其后续操作同时运行
* MMA 不能与 TMA load 或者 tcgen05.ld 同时运行，但是可以和 stage 以及 TMA store 同时运行
* 以此类推等等

为了简化图形的显示，我们假设每个 output tile 只需要两轮内层循环，即 K = 2*BK，即可得到如下的流水线时序图。

![单 buffer 配置下的流水线时序（K = 2·BK）](https://raw.githubusercontent.com/tongzhou8086/mmc/main/blog/figures/single-buffer-timeline.png)

可以看出，图中 MMA 的 issue 有两种类型的停顿：
* 同一个 output tile，不同的 k tile 之间存在停顿（或者空挡），需要等待 TMA load 的结束
* 不同的 output tile 交接时，也存在空挡，需要等待 tcgen05.ld 的结束

这两种停顿我们也可以称为内层循环的停顿和外层循环的停顿。

### 解决内层停顿

使用多个 TMA buffer 便可让 TMA load 和 MMA 操作并行起来，而无需互相等待同一个 buffer。下图演示使用两个 TMA buffer 的情况，实际Blackwell 上实现中，我们一般会使用更多的 TMA buffer。

![两份 TMA buffer 下的流水线时序](https://raw.githubusercontent.com/tongzhou8086/mmc/main/blog/figures/two-tma-buffer-timeline.png)

### 解决外层停顿

类似的，我们通过使用两个 MMA buffer，便能够使得 MMA 操作和 tcgen05.ld 操作重叠起来 —— 各自操作不同的 MMA buffer，如下图所示：

![两份 TMA buffer 加两份 MMA buffer 下的流水线时序](https://raw.githubusercontent.com/tongzhou8086/mmc/main/blog/figures/two-accumulator-timeline.png)

这下 MMA 那一行从头到尾连成了一片，再没有空档 —— 而 MMA 能持续 issue，正是整个流水线编排追求的目标。后文第一种设计里的双 MMA buffer，做的就是这件事。

### 到底需要几个 buffer？

上图中我们显示了，通过增加 TMA buffer 的数量，我们可以达到重叠 TMA load 和 MMA 操作的效果；同理，通过增加 MMA buffer 的数量，可以达到重叠 MMA 和 tcgen05.ld 的效果，那问题来了，到底需要几个 buffer 才能够实现完全的无等待？这实际上取决于生产操作和消费操作之间耗时的比例。

这里的思维模型是，当一次 TMA load 操作结束，

以上图所示的两个 buffer 为例，是否可以完全无等待，则是看，当一次 TMA load 操作结束，此时是否可以直接轮转到另一个 buffer 往里面开始加载数据？如果总是可以，那 TMA load 永远都不会有等待，即，使用另外一个 buffer 的 MMA 操作已经结束。鉴于一个 buffer 的 MMA 操作和另一个 buffer 的 TMA load 操作总是可以同时开始，于是可以推导出，如果 MMA 操作每一次耗时都小于等于 TMA load，则 TMA load 总是不会有等待。

假设每一次 TMA load 和每一次 MMA 的耗时完全一样，那么两个 TMA buffer 就完全够用了 —— 这两个 buffer 可以不断地轮回，每次一个 buffer 中数据到位时，对这个 buffer 的消费，即 MMA 操作，和下一个 buffer 的生产，即 TMA load 同时开始，之后两者同时结束 —— MMA 操作和 TMA load 操作则轮换一下各自的源和目的 buffer，正好完全卡点。

如果 MMA 操作需要的时间更长呢？那就会出现 TMA load 操作提前结束了，此时另一个 buffer 的数据也到位了，但是下一轮的 TMA load 则需要等待，因为 MMA 对上一个 buffer 的数据读取还没结束。

猜想：

如果 MMA 的耗时和 TMA load 的耗时的比例 <= N / 2，则使用 N 个 TMA buffer 可以保证 TMA load 操作持续轮转运行，不会等待。

通过画图，我感觉上述猜想对于 N=2、3、4似 乎成立，但是是否真的成立，有待进一步证明。

## 实现篇

### Warp Specialization
具体到软件编写的层面，上述的 5 种操作会被映射给不同的 warps，各个操作配置多少 Warps 也是流水线设计的一部分。对于本文所探讨的设计方案而言，均采用如下配置：

* 配一个 Warp 进行 TMA load，又称 TMA Warp
* 配一个 Warp 进行 MMA，又称 MMA Warp
* 配 4 个或者 8 个 warps 进行 tcgen05.ld 和 stage，这些 warps 也被称为 epilogue warps
* TMA Store 操作也顺便由上面的 Epilogue Warps 完成

这样的配置既有原理层面的约束，也是实际经验的结果。先说原理层面，之所以给 TMA load 和 MMA 操作都只配一个 Warp，是因为发射 TMA load 或者 Store 指令以及 MMA 指令，都只需要一个 Warp 的一个线程发出指令就行，所以分别配一个 warp 就行了 —— 发射 TMA load 或者 MMA 指令之前，也还会有一些整数计算，算一下 address offset 之类的，但是这些也都是 scalar 计算，所以理论上其实对于 TMA load 和 MMA 操作，配一个线程其实就够了，只是因为 GPU 的调度单元最小是一个 Warp，所以我们给它配了一个 Warp。

但是后面的 tcgen05.ld 和 stage 操作就不太一样了，这两操作要对数据进行批量操作，需要使用 SIMT 模型进行编程，于是我们要配多个 warps 来提高并行度 —— 4 个 warps 是 GPU 编程的一个标配，有时候我们也会配 8 个 warps 来提高能使用的寄存器数量。另外最后的 TMA Store 操作本质上也只需要配一个线程，我们可以给它单独配一个 Warp。但是经验上我们发现，就让 Epilogue Warps 顺便完成 TMA Store，也挺高效的。因为 TMA Store 是一个比较简单的操作，只需要发射一条 TMA store 指令，加上一点简单的地址计算。
 

### 流水线设计的参数配置与原语
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

### 第一种设计：BN256
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

#### 性能数字

下面我们看一下这个设计在方阵上的性能是多少，事实上我们考虑两种不同的 BK 选配：64/128，以及三种不同的 GROUP_SIZE_M （简称 GSM，代表 CTA swizzle 的深度）选配:8/12/16。不同的 GSM 选配仅仅需要改一个常数参数，而 BK=64/128 的区别也仅仅在于把 TMA buffer 的数量砍半，逻辑部分完全一致。
以下性能数字的测量方法使用 triton.do_bench 获得 median runtime，warmup 和 repetition time 都设置为 1 秒；每个尺寸跑三轮独立的测量，每轮内部再做三次打乱顺序的采样，最后取中位数。

![BN=256 性能对比](https://raw.githubusercontent.com/tongzhou8086/mmc/main/blog/figures/perf-bn256.png)


### 第二种设计：BN512 
BN256 的设计其实已经非常流畅了，TMA buffer 有 6个，应该能够流畅地将数据加载到 SMEM 中，这样 MMA 总是有操作数可以计算；此外，MMA buffer 也有两个，所以如果一个 output tile 的 epilogue 能在下下个 output tile 开始之前完成，理论上我们就可以持续不停的无卡顿地 issue MMA，这已经是一个非常流畅的设计了。

不过，根据实际性能结果，我们发现，在比较大的方阵上，BN256 的性能停滞在了 1300T 左右。一个可能的原因是，尽管 MMA 的 issue 的确没什么卡顿，但是每次只用了一半的 TMEM 做 accumulation 达到的算术强度还是不太够，也就是说，同样的数据量进来，它产生的计算量有限，于是，哪怕 MMA 的 issue 不卡顿，实际产生的计算量还是无法吃满 MMA Engine。

于是在第二种设计中，我们换一种新的思路，即，把整个 TMEM 作为一个 MMA buffer 使用，其 BN 是 512，简单的计算我们可以发现，BN=512 下，每个 K tile 的数据量会变为 1.5 倍 —— 从 32KB 变成 48KB，而计算量则会变为两倍，这样会导致同样的数据量进来，能够产生更大的计算量，更有可能能够吃满 MMA engine。但是与此同时，只用了一个逻辑上的 MMA buffer 的话，就会导致在 epilogue draining 期间，后续的 MMA 无法进行 issue，需要等待这个 MMA buffer 被腾空以后才能够 issue 后续的 MMA，这会导致在两个 output tile 中间交接的时候会造成一些卡顿。简而言之，两者的区别总结如下：

**BN256 可以无卡顿地持续 issue MMA，但是产生的计算量可能无法吃满硬件算力；BN512 通过加大 accumulation buffer 能够将 tensor core 的算力吃得更满，但是与此同时，在两个 output tile 交接的时候 issue MMA 会有些卡顿。**

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

#### 性能数字

BN512 的上述设计也可以有六种选配，BK=64/128，GSM=8/12/16，如果 BK=64 和 128 分别使用 4 个和 2 个 TMA buffer。性能测量方法与上面相同。

![BN=512 性能对比](https://raw.githubusercontent.com/tongzhou8086/mmc/main/blog/figures/perf-bn512.png)


几点观察：

* 相比 BN256，BN512 在稍大一些的方阵上确实能达到更高的性能 —— 稳定飚在了 1450T 左右。一个可能的解释是，方阵越大 K 维度也越大，Epilogue 所占的时间比例便相应缩小 —— BN512 让计算部分更快，代价是 Epilogue 会卡顿，所以正适合 K 比较大的情况。
* GSM 在这里的作用比 BN256 明显得多：BK=128 在 20480 上从 GSM=8 的 1423 涨到 GSM=16 的 1468；BK=64 在 17408 上从 1316 涨到 1412。
* 最好的一档（BK=128 + GSM=16）在 18 个尺寸中有 10 个跑赢了 cuBLAS。

### 第三种设计：BN512 加强版
在上述的 BN512 基础设计中，epilogue 的部分还是每次从 TMEM 中 load 64 列数据，直到最后一列都 load 完成以后才释放完整的 MMA buffer，这会导致 epilogue 占据 MMA buffer 比较长地时间。在下面的新版设计中，我们会做两个方面的改进。首先，我们加大 tcgen05.ld buffer 的容量，使得它一次能存放下 128 行 x 256 列的数据 —— 共计 128KB tcgen05.ld 的结果，于是 tcgen05.ld buffer 的容量需要扩充到 128KB，即整个 SM 上一半的寄存器容量。我们先把这 128KB 的数据都加载到寄存器中以后，随即释放 MMA buffer 左边的一半，之后再对这 128KB 的数据进行 stage 和 TMA store 操作，以及后续的 load 另一半的 128KB 结果。这样设计的益处在于，MMA buffer 的其中一半（256 列）可以被提前释放，而非要等到整个 MMA buffer 数据都搬完以后才能释放。更早地释放可以使得左边一半的 MMA 可以先开始 issue，只要对应的 k tile 的数据已经到位，这样便能和 epilogue 的后续操作重叠起来，减少 MMA issue 的卡顿。

#### 性能数字

这一组我们同样扫了 GSM 8/12/16 与两种 BK，共 6 种选配。

![BN=512 加强版性能对比](https://raw.githubusercontent.com/tongzhou8086/mmc/main/blog/figures/perf-bn512-splitacc.png)

可以看出，在大尺寸上，这种新设计和前面的 BN512 基础版设计性能差不多，大差不差，但是对于相当小一些的方阵的性能提升还是非常显著的，譬如 5120、6144 等。

### 工具
在 Meshy 我们开发了以下两个与此话题相关的工具：
* [MMComposer](https://mmcomposer.streamlit.app/)：一个 web app，上面 host 了各种我们开发的流水线编排设计，kernel 代码清晰易懂，对应的 host code 也有，可以一键下载，直接在 B200 上运行
* [mmc](github.com/tongzhou8086/mmc)：一个高性能的 GEMM 算子库，目前支持 BF16 和 MXFP8；里面集成了各种 kernel 的实现，并且会 auto tune 自动选择最优的 Kernel







