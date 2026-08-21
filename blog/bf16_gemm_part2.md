# B200 矩阵乘法优化分享 Part 2

不论流水线怎么编排，数据流动的总体步骤都是一样的，不同方案的区别在于同步方案、数据读写的次序和粒度、tile sizes 的配置、各种容器（buffer）的数量和大小等等。为了能把这些区别讲清楚，我们先建立一个数据流模型。

### 数据在内存和四种 Buffer 之间流动

本文所探讨的所有流水线调度方式，数据都会在以下四类容器之间流动：

* TMA buffer（位于 SMEM）: 存放 TMA 从内存（HBM，High-Bandwidth Memory，也就是常说的 GMEM / Global Memory）中加载的数据
* MMA buffer（位于 TMEM）: 存放 MMA 的中间结果以及最终结果
* tcgen05.ld buffer（位于寄存器，即 RMEM，Register Memory）: 存放从 TMEM 中读取的结果
* Store buffer（位于 SMEM）: 存放要写入内存的数据

事实上内存也是一种 buffer，数据会最初来自于内存，最后又流入内存。但由于从流水线资源调度的视角，内存并不参与。
上述任何一种类型的 buffer 都会在 GPU 片上有自己的物理存储介质 —— 如下图所示。TMA buffer 和 Store buffer 同属 SMEM 介质、要互相抢容量，MMA buffer 则独占 TMEM 空间，而 tcgen05.ld buffer 是寄存器文件中的一部分 —— 寄存器的其余部分还要用于地址计算、循环变量、各 warp 的私有状态等等。

![四种 buffer 的物理载体](https://raw.githubusercontent.com/tongzhou8086/mmc/main/data-flow-models/figures/sm-storage-map.png)


### 针对 Buffer 的五种操作
GEMM kernel 的运行都会涉及到下述五种对容器的操作，每一种操作会从一个源 Buffer 读取数据，并将操作后的结果写入目的 Buffer —— 从数据流的角度，可以视为数据从源 Buffer 流入了目的 Buffer。

这五种操作分别是：

* TMA load: 从内存读取数据，写入 TMA buffer
* MMA: 从 TMA buffer 读取数据，结果写入 MMA buffer
* tcgen05.ld: 从 MMA buffer 读取数据，写入 tcgen05.ld buffer
* stage: 从 tcgen05.ld 读取数据，写入 stage buffer
* TMA store: 从 stage buffer 读取数据，写入内存

把这五种操作串起来看，就是下面这张图 —— 每个箭头是一种操作，箭头两端则是它读取的源 buffer 和写入的目的 buffer：

![5 种操作及其源和目的 buffer](https://raw.githubusercontent.com/tongzhou8086/mmc/main/data-flow-models/figures/operations-chain.png)

第一个操作 TMA load 是从内存中读取用来做矩阵乘法的输入数据，这个好理解；第二个操作，MMA，即是对输入数据进行矩阵乘法操作，这个也好理解；第三个操作是什么呢？事实上，这里的背景是 MMA 的结果必须保存在 TMEM 中，当一个 output tile 所有的 MMA 都计算完毕，你必须先从 TMEM 中将计算完的结果读取到寄存器中才能进行后续的操作，譬如写回内存等等。将数据从 TMEM 中读取到寄存器中使用的指令系列叫做 [tcgen05.ld](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=tcgen05%2520ld#tcgen05-matrix-fragments-shape-3232b)，于是我们把这个操作称为 tcgen05.ld，这是第三个操作；接下来的操作称为 stage，这里又需要一些背景，即按照 Blackwell 架构的设计，通过 tcgen05.ld 读取到寄存器中的数据是[按列分布](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=tcgen05%2520ld#tcgen05-matrix-fragments-shape-3232b)到各线程中的，也就是说，一个线程会拥有同一行上连续的数据。这样的 layout 方式使得，如果直接将寄存器结果写入内存就会导致一个 warp 中不同线程按列写入，产生 [uncoalesced memory access](https://developer.nvidia.com/blog/unlock-gpu-performance-global-memory-access-in-cuda/)。于是在我们所有的设计中，都会将 tcgen05.ld 的结果先写入一个 SMEM 缓冲区，在缓冲区进行重组，每行能够凑满 128 个连续字节了再进行 coalesced memory write，这也便是最后的两步。

### Buffer 状态翻转与操作之间的同步法则
上述的四种容器中的每一种都会被一种操作读，也会被另一种操作写，为了保证操作的正确性，任何一个容器都无法同时被读以及被写，即同一个容器要么处于在读的状态，要么处于在写的状态，否则就会导致读入错误的数据。于是我们给所有的容器都分配两种互斥的状态：可读和可写。一个操作的完成，即产生一个事件，可以翻转容器的状态。假设一个容器的初始状态为“可写”，然后写操作开始，容器里被灌入新的数据，当写操作结束时，容器的状态即翻转为“可读”，代表数据已经就绪。类似的，假如一个容器的状态为“可读”，然后读操作开始，容器里的数据逐渐被消费。当读操作结束时，容器的状态即翻转为“可写”，代表数据已被消费完，内容可以被覆盖了。

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

既然我们流水线调度的最终目标便是减少 MMA issue 的停顿，让我们把视线专注在 MMA 的 issue 那一行，会发现它有两种类型的停顿（图中显示为空白），而它们恰好对应经典的两种数据依赖：

* **RAW 停顿**：同一个 output tile，不同的 k tile 之间存在停顿，需要等待 TMA load 的结束。MMA 要读的那份数据得先由 TMA load 写进去，这是一个 true dependence（read after write）。
* **WAR 停顿**：不同的 output tile 交接时，也存在空挡，需要等待 tcgen05.ld 的结束。MMA 要写的那块 accumulator 得先被 tcgen05.ld 读走，这是一个 anti dependence（write after read）。

这两种依赖，其实正是前面同步法则的两半：「源 buffer 可读」说的是 true dependence 已经满足，「目的 buffer 可写」说的是 anti dependence 已经满足。

### 解决 RAW 停顿

RAW 停顿来源于 read-after-write 数据依赖，减少停顿的方法则是预取（prefetch）数据，即 read 的时候同时开始下一轮的 write，这样可以减少下次 read 的时候的等待。这样的数据预取需要我们使用多个 TMA buffer，即当一个 MMA 开始时，与此同时，TMA load 可以同时开始往另一个 buffer 里面写入。
下图演示使用两个 TMA buffer 的情况，实际具体用几个 TMA buffer 是最优的取决于 MMA 和 TMA load 的时长比例。

![两份 TMA buffer 下的流水线时序](https://raw.githubusercontent.com/tongzhou8086/mmc/main/blog/figures/two-tma-buffer-timeline.png)

### 解决 WAR 停顿

WAR 停顿来源于 write-after-read 数据依赖，这种依赖并非真实的依赖，在编译原理以及计算机体系结构中的标准解决方案就是让 write 写入一个新的 buffer，在计算机体系结构中，这个被称为 register renaming。于是这里我们通过增加一个 MMA buffer，便能够使得 MMA 操作和 tcgen05.ld 操作重叠起来 —— 各自操作不同的 MMA buffer，如下图所示：

![两份 TMA buffer 加两份 MMA buffer 下的流水线时序](https://raw.githubusercontent.com/tongzhou8086/mmc/main/blog/figures/two-accumulator-timeline.png)

这下 MMA 那一行从头到尾连成了一片，再没有空档 —— 而 MMA 能持续 issue，正是整个流水线编排追求的目标。后文第一种设计里的双 MMA buffer，做的就是这件事。

### Warp Specialization
具体到软件编写的层面，上述的 5 种操作会被映射给不同的 warps，各个操作配置多少 Warps 也是流水线设计的一部分。对于本文所探讨的设计方案而言，均采用如下配置：

* 配一个 Warp 进行 TMA load，又称 TMA Warp
* 配一个 Warp 进行 MMA，又称 MMA Warp
* 配 4 个或者 8 个 warps 进行 tcgen05.ld 和 stage，这些 warps 也被称为 epilogue warps
* TMA Store 操作也顺便由上面的 Epilogue Warps 完成

这样的配置既有原理层面的约束，也是实际经验的结果。先说原理层面，之所以给 TMA load 和 MMA 操作都只配一个 Warp，是因为发射 TMA load 或者 Store 指令以及 MMA 指令，都只需要一个 Warp 的一个线程发出指令就行，所以分别配一个 warp 就行了 —— 发射 TMA load 或者 MMA 指令之前，也还会有一些整数计算，算一下 address offset 之类的，但是这些也都是 scalar 计算，所以理论上其实对于 TMA load 和 MMA 操作，配一个线程其实就够了，只是因为 GPU 的调度单元最小是一个 Warp，所以我们给它配了一个 Warp。

但是后面的 tcgen05.ld 和 stage 操作就不太一样了，这两操作要对数据进行批量操作，需要使用 SIMT 模型进行编程，于是我们要配多个 warps 来提高并行度 —— 4 个 warps 是 GPU 编程的一个标配，有时候我们也会配 8 个 warps 来提高能使用的寄存器数量。另外最后的 TMA Store 操作本质上也只需要配一个线程，我们可以给它单独配一个 Warp。但是经验上我们发现，就让 Epilogue Warps 顺便完成 TMA Store，也挺高效的。因为 TMA Store 是一个比较简单的操作，只需要发射一条 TMA store 指令，加上一点简单的地址计算。
 

### 流水线调度的参数配置与原语
流水线调度表达的是一个程序，它表达的是数据在前文所说的这些 buffer 之间流动的时候的时序设计以及同步关系。它涉及到一些基础的参数配置，这些参数就是流水线调度程序的常量。任何的一种调度方式，至少需要配置如下参数：

* Tile sizes: BM/BN/BK 设为多少
* 上述的 4 种 Buffer 每种分别配几个，大小几何

如前文所述，我们会对同一个类型的 Buffer 配置多个，来提高流水线的重叠程度，从而减少上面这两类停顿。在硬件资源（SMEM、TMEM、RMEM 的容量）给定的情况下，每种类型的 Buffer 具体配多大、配几个就存在一个设计问题。Tile Sizes 的配置中，BM 的配置固定为 128，和 TMEM 的 layout 一致（128 行），没有别的选择。对于 M 无法被 128 整除的情况，则需要对做 out of bound 处理。而单次 MMA 操作可以支持的 N 的尺寸可以是 64 或 128 或 256。再者，由于 TMEM 的大小为 128 行乘 512 列，这实质上也限制了我们对 BN 的配置，最大为 512。所以后文会讨论两种不同的配置，BN 分别配为 256 和 512，它们达成的流水线状态会相当不一样。BK 也是需要配置的一个 Tile size 参数，由于它是 A Tile 的 inner dimension，于是便涉及内存的连续访问问题。对 GPU 内存进行读写，你最好能以 128 个连续字节为单位进行操作，这样就不会浪费内存带宽。于是这里我们会把 BK 配成 64 的整数倍 —— 对于 BF16 数据类型，64 个元素就是 128 字节。

另外，由于 2 CTA MMA 这个优化本身会影响到 tile sizes 以及各类 Buffer 的配置数量，所以我们也提前说明，本文所探讨的所有设计都默认了 2 CTA MMA 开启，于是每个 CTA 只需从内存中读取它实际所需的 BN 的一半，由此也可以看出 2 CTA MMA巨大的功效：除了提高算术强度以外，它还能将每个 CTA 从内存中读取的 B Tile 大小砍半，从而腾出更多的 SMEM 空间来做成 TMA buffer 或者 Store buffer。


所以总结起来就是，对于所有的设计，BM 必须配成 128，BN的配置，我们会讨论两种方案：256 和 512。BK 为 64 的整数倍，再加上 2 CTA MMA 的开启，A Tile 和 B Tile 分别占用的字节数的计算公式如下：

* A tile: BM x BK x 2
* B tile: BK x BN x 2 / 2

B tile 之所以后面会除以 2，是因为我们默认了 2 CTA MMA 的开启。


除了参数配置，流水线设计定义了一套基本操作。不同的调度方案虽然在 tile sizes、buffer 数量、
数据的加载、MMA 的 issue、epilogue 的时机上各不相同，但都可以用同一套原语来表达，即一套流水线调度的基本对象，及其对应的
操作与信号；完整定义详见 [docs/pipeline-primitives.md](../docs/pipeline-primitives.md)。
后文的伪代码都基于这套原语书写。

流水线的理论部分介绍完毕，在下一部分中，我们会开始看具体的调度设计。
