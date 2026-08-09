# B200 矩阵乘法优化分享

一般矩阵乘法优化教程会这么写：先写一个简单的 kernel，然后逐步优化，乃至可以视为“叠加”各种优化，然后经过十几次优化之后，终于达到了类似 CuBlas 的性能。

本文以我们自己在 B200 优化矩阵乘法的经历为基础，给大家讲述一个不同的故事，一个以“流水线编排”为核心的故事、一种近乎于“算法设计”的视角来看到矩阵乘法 kernel 的编写，而非传统的叠加式优化。而至于流水线编排之外的优化，我们仅仅会使用两种必备的常规优化 —— 2-CTA MMA（Blackwell 自带核心硬件功能）以及 CTA swizzling（提高 L2 缓存利用率，类似于 L2 Cache Tiling），便能达到接近乃至超越 CuBlas 的性能。

以方阵为例，本文所探讨的几种流水线编排方案，在 4096 到 16384 大小的方阵上所达到的最高性能，均在 CuBLAS 的 95% 以上，在 8192、9216、10240 等尺寸上所达到的最高性能高达 CuBLAS 的 105%。而这一切性能背后的秘密，都隐藏在流水线设计的背后。

## 术语
下面是本文会使用到的术语对应的简称和全称：
* HBM / GMEM: High-Bandwidth Memory / Global Memory (GPU main memory)
* SMEM: Shared Memory
* TMEM: Tensor Memory
* RMEM: Register memory
* TMA: Tensor Memory Accelerator
* MMA: Matrix Multiply Accumulate 

## 背景：Tiled GEMM
矩阵乘法的计算模式，天然适合于“分块”（Tiling）这样一种优化方式，即每次加载一小块输入到片上，也只计算一小块输出。这样的好处是提高数据局部性，充分使用每一小块的数据进行计算，减少对于全局内存的访问。这里我们假定读者已对数据局部性、分块等基础背景具有相当的了解，便不再赘述其基本原理，直接探讨分块的大小如何影响流水线的编排。

<img width="300" alt="图片" src="https://github.com/user-attachments/assets/14bd90bb-6c39-409d-9295-4c3ea792290f" />


如上图所示，分块计算的矩阵乘法有三个维度，它们常常被称为 BM、BN、BK，即，每次加载 BMxBK 大小的 A（称为 A tile），以及 BKxBN 大小的 B（称为 B tile），以此计算 BMxBN 大小的 C 的部分结果（Partial accumulation）。这三者我们统称为 tile sizes。与 Tile Sizes 相关的一个核心概念叫做算术强度（Arithmetic intensity），它用来衡量每单位的 Memory Traffic ，譬如每字节，能够产生的计算量是多少。算术强度是一个软件本身的特征，不同的软件编写方式，便会产生不同的算术强度。假如 A tile 和 B tile 都能够完全地存放在片上存储中，即 SMEM 或寄存器，那算术强数的计算方式即为

$$ A.I. = \frac{(2\times BM \times BN \times BK)}{2\times BM \times BK + 2\times BK \times BN}$$

通过简单的数学推导，我们可以看出，BM 和 BN 越大，算术强度就越大。所以在实际的矩阵乘法算子的设计与实现中，我们会尽可能把 BN 和 BN 配得更大一点。但是这里的 trade off 在于片上存储空间是有限的，譬如对于 Blackwell 而言，能够使用的最大的 SMEM 的大小是 227 KB，而寄存器的总共的容量是 256KB，TMEM 的总容量也是 256K。所以实际应用中，M、N 和 K 实际的输入尺寸总是很大的，甚至可以无限大，而 BM/BN/BK 的尺寸要远小于 M、N 和 K。

## 背景：Blackwell 的 TMA 、MMA 和 TMEM
流水线的设计，本质上就是编写一个软件，使得这个软件能够高效的对于其背后的硬件进行调度。而需要被调度的硬件单元大概有这么三种：TMA、MMA 以及 CUDA core 或者 Integer core。TMA 是自 Hopper 架构以后引入的一种独立的硬件单元，用来异步的在内存和 SMEM 之间传输数据，既可以将内存数据加载到 SMEM 中，也可以将 SMEM 中的数据写入到内存。由于是独立的硬件单元，TMA 的运作便不再占用 CUDA Cores 或 integer Cores的算力，而可以独立异步地运行。与此同时，它还硬件支持数据的 swizzling。所以在本文所探讨的所有的流水线的编排方案之中，都会默认使用 TMA 来加载数据以及写入数据。

Blackwell 的 MMA 单元是新一代的 TensorCore Engine，和 TMA 单元类似，MMA 单元也是可以独立异步的运作。从软件的角度，只需要单个 Warp 的一个线程发送 MMA 指令，MMA 单元便可以在背后异步地进行 MMA 运算。其实也正是因为 TMA 和 MMA 单元都是异步的，才会使得流水线的设计大放异彩。

TMEM 也是 Blackwell 引入的一种新的硬件单元，但它是一种存储介质，它以一种类似于矩阵的方式组织，有 128 行、512 列，每一个 cell 可以存放一个 float 类型，4 个字节，总容量为 256K。TMEM 用来存放 MMA 的中间结果或最终结果，而且根据 Blackwell 的设计，MMA 的结果只能存放在 TMEM 中。

这些硬件单元都各有自己的设计上的限制，或者是通用性的最佳性能 practice，这也直接影响了我们的流水线设计。具体的限制以及影响，我们在后面具体的流水线设计中再展开。

## 数据流动的全景图
不论我们流水线编排如何设计，数据流动的总体的步骤是一样的，不同的编排方案的区别在于不同的同步方案、数据的读取、写入的粒度、Tile sizes 的配置、各种 buffer 的数量的配置等等。但数据的流动都遵循同样的步骤，大概分为下面这么几步：

* 从内存到 SMEM：TMA engine 会首先把数据从内存搬运到 SMEM
* 从 SMEM 到 TMEM：MMA engine 从 SMEM 中读取计算需要的输入数据，计算结果保存在 TMEM 中
* 从 TMEM 到寄存器：Accumulation 结束以后，数据便可以从 TMEM 中读取出来，暂存在寄存器中
* 从寄存器到内存：暂存在寄存器中的数据，最后会被写入内存；事实上我们还会先把寄存器中的数据写入到 SMEM 中进行缓冲和重排，之后再交给 TMA 写入内存，以实现 memory write 的 coalescing

可以看出，数据会在不同的硬件单元、存储介质之间流动，而操作与操作之间具有依赖关系。譬如 MMA 要能够进行，必须要等输入数据到位以后。这便涉及到下面这个章节的数据流建模。

## 数据流模型
流水线的资源调度涉及如下四种 buffer:

* TMA buffer: 存放 TMA 从内存中加载的数据
* MMA buffer: 存放 MMA 的中间结果以及最终结果
* tcgen05.ld buffer: 存放从 TMEM 中读取的结果
* Store buffer: 存放要写入内存的数据

用图形化的表示方式如下。

<img width="500" alt="图片" src="https://github.com/user-attachments/assets/5972016a-c6d8-47ee-8db2-2b7fac8f3e19" />


每一种 buffer 都有一个输出端口和输入端口，类似于一个开关，上图中我们用绿色代表打开，红色代表关闭。上图中所画的则是每一个 buffer 的初始状态：输入端口打开，表示数据可以写入；输出端口关闭，表示数据尚未 ready 被 consume。

Buffer 和 Buffer 之间的数据流动通过“操作”完成，我们定义如下三种操作：

* TMA load: 数据从内存流入 TMA buffer
* MMA: 数据从 TMA buffer 流入 MMA buffer
* tcgen05.ld: 数据从 MMA buffer 流入 tcgen05.ld buffer
* stage: 数据从 tcgen05.ld buffer 流入 Store buffer
* TMA store: 数据从 Store buffer 流入内存

可以看到，在上图中我们并没有把内存画出来，但是实际上数据会先从内存流入 TMA buffer，最后也会从 Store buffer 流入到内存。所以实际内存也是数据流模型的一部分。但是由于内存不涉及任何的同步以及调度，所以为了简化被省略了。如果要把内存作为一个 Buffer 画出来的话，那它的输入和输出端口将总是绿的，即，你总是可以往里面写或者从里面读。

除内存以外的其他 buffer 都存在一个 invariant，即，输入和输出端口不可能同时开放（为绿色）、也不可能同时都关闭（为红色）。当我们往 Buffer 里面写数据的时候，需要保持输入打开，但是输出端口需要关闭，因为数据还没有 Ready；类似的，当我们从 buffer 里面读取数据时，需要保持输出端口打开，这时输入端口则需要关闭，不然就会覆盖原有数据。此外，如果两个端口都关闭，则数据进不去也出不来。

现在我们再来看如何对“操作”建模。一个操作本质上就是连接两个 buffer 之间的一条管道，而一个操作要能够开始，需要同时满足：1）源 Buffer 的 Output 端口开启；2）目的 buffer 的 input 端口开启，即，既要源 buffer 可读，也要目的 buffer 可写。以 MMA 操作为例，它的图形化表示如下。

<img width="500" alt="图片" src="https://github.com/user-attachments/assets/1f7b980d-88cb-45de-bed7-0198a390f63a" />


除了 Buffer 和操作，数据流模型的第三个要素是信号（signal）。信号的作用很简单：翻转 Buffer 的开关。上图中的四种类型的 Buffer 都会有两个自己配套的信号，一个代表“打开输入端口、关闭输出端口”，另一个则相反，代表“打开输出开关，关闭输入开关”；从语义上讲，前者代表 “buffer free”，而后者代表 “data ready”。下图代表了一个 TMA buffer 分别接收 data ready 和 buffer free 信号后端口状态的切换。

<img width="500" alt="图片" src="https://github.com/user-attachments/assets/6aae0506-9f98-4b5f-a9c3-cd3b273b3a09" />



## Warp Specialization
具体到软件编写的层面，上述的 5 种操作会被映射给不同的 warps，各个操作配置多少 Warps 也是流水线设计的一部分。对于本文所探讨的设计方案而言，均采用如下配置：

* 配一个 Warp 进行 TMA load，又称 TMA Warp
* 配一个 Warp 进行 MMA，又称 MMA Warp
* 配 4 个或者 8 个 warps 进行 tcgen05.ld 和 stage，这些 warps 也被称为 epilogue warps
* TMA Store 操作也顺便由上面的 Epilogue Warps 完成

这样的配置既有原理层面的约束，也是实际经验的结果。先说原理层面，之所以给 TMA load 和 MMA 操作都只配一个 Warp，是因为发射 TMA load 或者 Store 指令以及 MMA 指令，都只需要一个 Warp 的一个线程发出指令就行，所以分别配一个 warp 就行了 —— 发射 TMA load 或者 MMA 指令之前，也还会有一些整数计算，算一下 address offset 之类的，但是这些也都是 scalar 计算，所以理论上其实对于 TMA load 和 MMA 操作，配一个线程其实就够了，只是因为 GPU 的调度单元最小是一个 Warp，所以我们给它配了一个 Warp。

但是后面的 tcgen05.ld 和 stage 操作就不太一样了，这两操作要对数据进行批量操作，需要使用 SIMT 模型进行编程，于是我们要配多个 warps 来提高并行度 —— 4 个 warps 是 GPU 编程的一个标配，有时候我们也会配 8 个 warps 来提高能使用的寄存器数量。另外最后的 TMA Store 操作本质上也只需要配一个线程，我们可以给它单独配一个 Warp。但是经验上我们发现，就让 Epilogue Warps 顺便完成 TMA Store，也挺高效的。因为 TMA Store 是一个比较简单的操作，只需要发射一条 TMA store 指令，加上一点简单的地址计算。
 

## 流水线设计的参数配置
流水线设计的表达是一个程序，它表达的是数据在前文所说的这些 buffer 之间流动的时候的同步规则以及时序设计，与此同时，它也涉及到一些基础的参数配置，这些参数就是流水线这个程序的常量。任何的一种调度方式，至少需要配置如下参数：

* Tile sizes: BM/BN/BK
* 上述的 4 种 Buffer 每种分别配几个？

在上面的图中，这 4 种 Buffer 一样，我们只画了一个。但是在实际的高性能算子设计中，我们常常会同样的类型的 Buffer 会配多个，来提高并行度，只要硬件资源（SMEM、TMEM、RMEM 的容量）允许的话。所以具体配几个，就存在一个设计问题。此外，Tile Sizes 的配置中，BM 的配置是被硬件写死了，Blackwell 架构每次 MMA 支持的尺寸，M 维度必须是 128，所以在本文的所有探讨中，BM 都会被配成 128，没有别的选择。如果输入矩阵的 M 尺寸小于 128 怎么办？那 BM 还是会被配成 128，只是对超出来的部分进行 out of bound 处理。而单次 MMA 操作可以支持的 N 的尺寸可以是 64 或 128 或 256。再者，由于 TMEM 的大小为 128 行乘 512 列，这实质上限制了我们对 Tile Sizes 的配置：BM 最大为 128，BN 最大为 512。所以后文会讨论两种不同的配置，BN 分别配为 256 和 512，它们达成的流水线状态会相当不一样。BK 也是需要配置的一个 Tile size 参数，由于它是 A Tile 的 inner dimension，于是便涉及内存的连续访问问题。对 GPU 内存进行读写，你最好能以 128 个连续字节为单位进行操作，这样就不会浪费内存带宽。于是这里我们会把 BK 配成 64 的整数倍。对于 BF16 数据类型，64 个元素就是 128 字节。

另外，由于 2 CTA MMA 这个优化本身会影响到 tile sizes 以及各类 Buffer 的配置数量，所以我们也提前说明，本文所探讨的所有设计都默认了 2 CTA MMA 开启，于是每个 CTA 只需从内存中读取它实际所需的 BN 的一半，由此也可以看出 2 CTA MMA巨大的功效：除了提高算术强度以外，它还能将每个 CTA 从内存中读取的 B Tile 大小砍半，从而腾出更多的 SMEM 空间来做成 TMA buffer 或者 Store buffer。


所以总结起来就是，对于所有的设计，BM 必须配成 128，BN的配置，我们会讨论两种方案：256 和 512。BK 为 64 的整数倍，再加上 2 CTA MMA 的开启，A Tile 和 B Tile 分别占用的字节数的计算公式如下：

* A tile: BM x BK x 2
* B tile: BK x BN x 2 / 2

B tile 之所以后面会除以 2，就是因为 2 CTA MMA 的开启。下面我们看本文的第一种流水线设计。对于每种设计，我们首先看参数如何配，参数确定好以后，再看同步如何进行。


## 第一种设计：BN256
我们要讲的第一个设计：BM 为 128、BN 配成 256。如我们刚刚所说，Blackwell MMA engine 支持的 BN 尺寸有三种，64、128 和 256。所以我们默认就把 BN 配成最大的 256 —— BN 越大的话计算效率越高。与此同时，TMEM 的容量是 128 行乘 512 列，刚好可以装下两个 BN256 的 MMA buffer。这里之所以称这个版本为基础版，是因为它对于两个 MMA buffer 的同步模式的设计会比较简单，尽管简单，它已经是一个相当有效的设计，同时也是更加精巧设计的基础。

如前面所说，为了增加并行度，我们会在硬件资源允许的前提下各自 buffer 都配不止一个，这里我们给 TMA buffer 配置 6 个（一个 A tile + B tile 的大小是 32KB），store buffer 配置 2 个（一个 store buffer 的大小也是 32KB），tcgen05.ld buffer 的话我们这里所有讨论都只会配一个，其大小可选配，有多种方案。tcgen05.ld buffer 只配一个的原因在于首先tcgen05.ld 操作是一个比较快的操作，即从 TMEM 中读取数据到 RMEM 中，此外我们的许多设计中会大量使用寄存器来暂存 TMEM 数据，由于容量限制也配置不了多个。

<下面的横杠以内的部分替换成使用伪代码来表达这个算法，而非这种图形化的形式。伪代码比较精简而且准确，适合算法的描述。>
-------------------------------------------
双 MMA buffer 基础版的算法框架如下：

* TMA warp 会依次向每个 TMA buffer 中发射 TMA load 指令，发完一个就移动到下一个。一个 TMA buffer 能够接受 TMA load 指令的前提是其输入端口处于开放状态，即，buffer free。
* MMA warp 会依次对各个 TMA buffer 中的数据发射对应的 MMA 指令，发完一个就移动到下一个。对于一个 TMA buffer，能够发射 MMA 指令的前提是其输出端口处于开发状态，即，数据已经 ready。此外对于每个 output tile 在第一个 MMA 时，MMA warps 还需要先等待其对应的 MMA buffer 的输入端口开放。每次切换到下一个 output tile 的时候，MMA buffer 就会自动在两个 MMA buffer 中切换，这样可以使得当前 MMA buffer 的 epilogue 和后续的 MMA 能够重叠起来。
* 一个 output tile 的所有 MMA 都完成以后，MMA buffer 的输出端口会打开，与此同时其输入端口也会关闭，防止数据被覆盖。这时，epilogue warps 就能开始行动了，它首先把数据从 TMEM 搬运到寄存器（tcgen05.ld buffer）中，然后以 128 行 x 64 列为一个 store 单位，先 stage 到 store buffer 中，然后 issue 一次 TMA store，这样的方式能够达到内存 coalesced 写入的效果。


这已经是一个高度重叠的流水线：MMA 操作在对 TMA buffer0 的数据进行 MMA 操作时，与此同时，TMA load 操作也在进行，譬如或许在对 TMA buffer1 进行数据写入；与此同时，tcgen05.ld 和 stage 操作也在进行（这两者统称为 draining），譬如对另一个 MMA buffer 中数据进行 draining；与此同时，TMA store 操作也在进行，譬如对已经 draining 结束的数据进行内存写入。

我们前面提到过的 5 中操作，除了 tcgen05.ld 和 stage 是串行的以外，其他所有操作全部并行了起来，全部处于同时运行的状态。下面我们用图示的方式把流水线的各个状态画出来，来直观地看出不同操作之间的依赖关系。同时图示中也顺便说明了流水线不同部件之间是重叠起来运行的。不过，值得指出的是，下图中描述的是准确的操作之间的同步关系，那是对操作之间的重叠的时间线并不一定准确，也并非来自实际的 Profiler 数据，而只是在大意上代表这些操作是会重叠起来的。

为了简单起见，我们假设 K=128，这样每个 output tile 只需两次 k 迭代（BK=64）便完成了 accumulation。

### 状态一：TMA load

此时，第一波 A tile 和 B tile 数据正在被加载进 TMA buffer 0。注意此时 TMA buffer 0 的输入端口打开，输出端口保持关闭。

![图片](https://hackmd.io/_uploads/Syq8QzmLMx.png)



### 状态二：TMA load 和 MMA 同时进行
数据加载已完成，TMA buffer 0 的输出端口开放，MMA 操作也已经开始，结果保存在 MMA buffer 0 中。

![图片](https://hackmd.io/_uploads/H11a7zXUMe.png)


### 状态三：第一个 output tile 的第二轮 MMA

上一轮的 MMA 计算完毕，TMA buffer 0 被释放，可以看到它的所有端口重新打开，即，可以接受新的数据进来。与此同时，假定 TMA buffer 1 的数据已就绪，MMA 则继续进行；与此同时，TMA load 也在进行，譬如将数据加载到 TMA buffer 2 中。
![图片](https://hackmd.io/_uploads/HkeS4GQUMg.png)

### 状态四：第一个 output tile 的 draining 开始；与此同时下一个 output tile 的计算也开始

此时，TMA buffer 1 中的数据完成了对应的 MMA，所以第一个 output tile 已经算完了，可以看出这个时候 MMA buffer 0 的输出端口打开，tcgen05.ld 操作可以开始了。这里我们特意把 MMA buffer 画成了四等份 —— 每次 tcgen05.ld 会读取其中的一等份，完成 staging 以后写入内存，然后继续读取下一等份。与此同时，下一个 Output tile 的计算也可以正常开始 —— 每次开始一个新的 output tile 的计算，就会切换到另一个 MMA buffer。这样一个 output tile 的 draining 只要在**下下个** output tile 开始前完成就不会有卡顿。这个时间窗口就和 K 的大小有关，K 越大的话，这个时间窗口就越大，而 K 越小的话，便越可能造成 MMA issue 的卡顿。这里我们流水线设计的最终目标就是**最小卡顿地吃满 MMA**。

![图片](https://hackmd.io/_uploads/SJ22Ez7UGx.png)

------------------------------------------------------------

### 性能数字

下面我们看一下这个设计在方阵上的性能是多少，事实上我们考虑两种不同的选配，BK=64 和 BK=128，上述的讨论中，假定的 BK 等于 64，BK 等于 128 的情况很简单，就是把 TMA buffer 的数量砍半就可以，其他原理不变。以下性能数字的测量方法使用 triton.do_bench 获得 median runtime，warpup 和 repetition time 都设置为 1 秒。



| Shape | bk64 (NS=6) | bk128 (NS=3) | delta | torch.matmul | best CUDA vs torch |
|---:|---:|---:|---:|---:|---:|
| 2048³ | 800 | 799 | -0.1% | 883 | -9.4% |
| 3072³ | 1381 | 1317 | -4.7% | 1447 | -4.6% |
| 4096³ | 1383 | 1329 | -3.9% | 1411 | -2.0% |
| 5120³ | 1401 | 1372 | -2.0% | 1449 | -3.3% |
| 6144³ | 1438 | 1429 | -0.6% | 1495 | -3.8% |
| 7168³ | 1424 | 1413 | -0.8% | 1459 | -2.4% |
| 8192³ | 1415 | 1407 | -0.5% | 1463 | -3.3% |
| 9216³ | 1408 | 1403 | -0.4% | 1434 | -1.8% |
| 10240³ | 1402 | 1406 | +0.3% | 1443 | -2.6% |
| 11264³ | 1385 | 1404 | +1.4% | 1437 | -2.3% |
| 12288³ | 1380 | 1401 | +1.5% | 1483 | -5.5% |
| 13312³ | 1359 | 1410 | +3.8% | 1412 | -0.1% |
| 14336³ | 1357 | 1412 | +4.1% | 1428 | -1.1% |
| 15360³ | 1297 | 1422 | **+9.6%** | 1479 | -3.9% |
| 16384³ | 1273 | 1392 | **+9.3%** | 1493 | -6.8% |

## 第二种设计：BN512 
对于 BN512 的情况，本质上就是在计算的时候，把完整的 TMEM 全部用上 —— 即两个 MMA buffer 同时被使用，以此提高计算效率，算得更高效一些。但是 trade off 在于每一轮算完以后，最后 Epilogue 开始的时候会稍微卡一下，直到 MMA buffer 被释放为止。为了减少 Epilogue 部分的卡顿，这里我们在 Epilogue 的部分稍做一点点修改，此前 BN 256 的设计中，每次会 从 TMEM load 64 列，这里我们改成每次 load 128 列，寄存器也完全是放得下的。然后还是分两次做两个 64 列的 chunk 来 Store，相当于就是外层循环每次 load 128 列，内层循环还是一次处理 64 列的 chunk。

<在下面的伪代码部分开始之前，先计算一下 BN512 情况下的 Tile Sizes，最后的结论应该是，每一个 TMA buffer 的 size 应该是 48KB，然后可以用四个。Store buffer 还是照样配两个。>

<同理，在这个地方也加上伪代码的描述 使用伪代码来描述这个算法，这个算法对应的 kernel 名称应该是bf16-single-ns2-store2-bk64-bn512-load128 以及其 bk128 版本>

### 性能数字

BN512 的上述设计也可以有两种选配，BK=64 和 BK=128，如果 BK 等于 128 的话，则只有两个 TMA buffer。

<性能数字这里我们先就这样放着，后面再加进去,这里不做任何改动。>






