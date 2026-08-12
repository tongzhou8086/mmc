# 流水线调度原语（Pipeline Scheduling Primitives）

本文所探讨的各种流水线调度方案，虽然在 tile sizes、buffer 数量、同步时机上各不相同，
但都可以用同一套原语来表达：四种 buffer 对象，以及它们对应的操作与信号。这份文档就是
这套原语的完整定义。

关于这套原语背后的模型 —— buffer 的输入/输出端口、把操作建模成两个 buffer 之间的管道、
以及信号如何翻转端口 —— 见博客正文的「数据流模型」一节。

```c
// 四种 buffer。每个 buffer 只有一个 state：输入端口和输出端口互斥，
// 打开一个就等于关掉另一个 —— 这正是数据流模型里「一个信号翻转两个端口」的含义。
struct TMA_Buffer {
  int_t size;  // its size
  bool state;  // 0: input open  1: output open
  TMA_Buffer(int_t size) {  state = 0; size = size; // initialize to input open }
}

struct MMA_Buffer {
  int_t size;  // its size
  bool state;  // 0: input open  1: output open
  MMA_Buffer(int_t size) {  state = 0; size = size; // initialize to input open }
}

struct TCGEN05_LD_Buffer {
  int_t size;  // 寄存器容量，决定了一次能从 TMEM 搬多少列
  bool state;
  TCGEN05_LD_Buffer(int_t size) { state = 0; size = size; }
}

struct Store_Buffer {
  int_t size;  // 一块 SMEM staging 区，大小为 BM x STORE_N x 2
  bool state;
  Store_Buffer(int_t size) { state = 0; size = size; }
}

// TMA load operation
* tma_load_async(input, cor_x, cor_y, SIZE_X, SIZE_Y, buffer, offset)
  * 以 cor_x，cor_y 为坐标，从矩阵 matrix 加载 一个大小为（SIZE_X，SIZE_Y）的块进入 SMEM；Swizzling mode 会自动适配
  * Non blocking：发射完指令就返回，数据到没到由 buffer 的端口状态来表达

// MMA operation
* issue_mma_chain_async(mma_buffer, ...)
  * 为对应的 A tile 和 B tile issue MMA 指令
  * 也是 non blocking 的，不需要等待 MMA 结束

// tcgen05.ld operation
* tcgen05_ld_x32_async(mma_buffer, offset, ld_buffer, offset)
  * 从 mma_buffer 的 offset 列处读出一段，写进 ld_buffer 的 offset 处，即 TMEM → RMEM；non blocking
* tcgen05_wait_ld()
  * 阻塞，直到本 warp 发出的 tcgen05.ld 全部完成。注意它只保证本 warp 自己的读取，别的 warp 读完没有它不管

// Stage operation
* stage(ld_buffer, store_buffer)
  * 把寄存器里的 fp32 结果转成输出精度，按 swizzle 布局写进 store buffer，即 RMEM → SMEM
  * 同步操作：返回时数据已经在 SMEM 里了

// TMA write operation
* tma_store_async(store_buffer, output)
  * 把 store buffer 的内容写回 HBM；non blocking

// Signals
// 前四个是「回调」式的：调用时立刻返回，等到对应的异步操作真正完成时，端口才翻转。
* make_ready_on_tma_done(buffer)
  * Non blocking call，类似于注册一个回调函数
  * 当 TMA 操作完成时，打开 Buffer 的 Out 端口 
* make_free_on_tma_done(buffer)
  * 同上，但打开的是 In 端口。用在 store buffer 上：TMA store 把数据写回内存以后，这块 SMEM 才能重新装填
* make_ready_on_mma_done(buffer)
  * 当此前发射的 MMA 全部完成时，打开 Out 端口。用在 accumulator 上：一个 output tile 累加完，结果才可以被 drain
* make_free_on_mma_done(buffer)
  * 当此前发射的 MMA 全部完成时，打开 In 端口。用在 TMA buffer 上：MMA 把这一片数据消费完，它才能被重新装填

// 后两个是「立即」式的，用在同步操作之后 —— 此时操作已经完成，不需要再等回调。
* make_ready(buffer)
  * 立刻打开 Out 端口（同时关掉 In）：数据已经写好了，可以被消费
* make_free(buffer)
  * 立刻打开 In 端口（同时关掉 Out）：数据已经被消费完了，可以重新装填

// Waits
// 两个 wait 对四种 buffer 通用，都是阻塞的，等到对应端口打开为止。
// 注意它们和上面两个信号是一一对应的：make_ready 打开的端口由 wait_until_ready 等待，make_free 同理。
* wait_until_ready(buffer)
  * 阻塞，直到该 buffer 的 Out 端口打开，即里面的数据已经写好了，可以拿去消费
  * 四种 buffer 通用：等 TMA buffer 就是等 TMA load 落地，等 MMA buffer 就是等一个 output tile 的所有 k 迭代累加完
* wait_until_free(buffer)
  * 阻塞，直到该 buffer 的 In 端口打开，即里面的数据已经被消费完了，可以重新装填
  * 同样四种通用：等 TMA buffer 就是等 MMA 消费完，等 store buffer 就是等上一次 TMA store 把这块 SMEM 读完

```
