# Throughput as issue rate times compute per issue

A GEMM kernel's throughput factors into three terms, all of which can be
measured rather than argued about:

```
throughput = A x f x duty
```

| term | meaning | how it is obtained |
|:--|:--|:--|
| `A` | compute produced by one `tcgen05.mma` issue | exact arithmetic |
| `f` | how often that issue happens, inside the k loop | measured |
| `duty` | fraction of time spent in the k loop at all | measured |

The point of this decomposition is that it is complete: the MMA engine is idle
for exactly two reasons, waiting for operands (which lowers `f`) or waiting for
its accumulator to drain (which lowers `duty`). Everything else a kernel does -
buffer counts, barrier placement, swizzle, arithmetic intensity - is upstream,
and matters only through these two numbers. NVIDIA's quoted peak is the same
product with the engine never idle.

## A is a hardware constant here

`tcgen05.mma` caps N at 256, and the 2-CTA MMA fixes M at 256, so

```
A = 2 * 256 * 256 * 16 = 2.097 MFLOP
```

for **every** design in this repo. BN=512 is not one wider instruction; it is
two N=256 MMAs per k-tile:

```
design 1 (BN=256):  make_idesc_bf16_cluster(CTA_GROUP*BM, BN)        BN       = 256
design 2 (BN=512):  make_idesc_bf16_cluster(CTA_GROUP*BM, BN_PANEL)  BN_PANEL = 256
```

So no design can win by producing more compute per issue. With `A` fixed, the
entire design space is `f` and `duty` - which is why pipeline scheduling is the
whole game on this hardware.

## Instrumentation

`-mmarate` variants record two timestamps per output tile in the MMA warp: one
immediately before that tile's first MMA issue, one immediately after its last.

```
compute_i = t_end(i)     - t_begin(i)      the k loop
pause_i   = t_begin(i+1) - t_end(i)        everything between tiles
```

`duty` is then a ratio of cycle counts, so it needs no clock calibration.
Combined with a benchmarked TFLOP/s it yields the k-loop rate, and dividing by
`A` yields `f`. Two clock reads per output tile is negligible overhead, and
both instrumented kernels compile to the same register count as their parents
(87 and 85, no spills). Passing a null buffer disables the instrumentation, so
the same cubin can also be timed.

## Measured

design              shape  measured    duty  k-loop rate            f  issue every
----------------------------------------------------------------------------------
design 1  BN=256     4096     1309   98.3%        1332       8.58 M/s       117 ns
design 2  BN=512     4096     1311   90.6%        1447       9.32 M/s       107 ns
design 1  BN=256     8192     1341   99.1%        1353       8.72 M/s       115 ns
design 2  BN=512     8192     1367   95.3%        1435       9.25 M/s       108 ns
design 1  BN=256    12288     1281   99.4%        1289       8.31 M/s       120 ns
design 2  BN=512    12288     1333   97.0%        1375       8.86 M/s       113 ns
design 1  BN=256    16384     1262   99.5%        1268       8.17 M/s       122 ns
design 2  BN=512    16384     1299   97.7%        1329       8.57 M/s       117 ns
design 1  BN=256    20480     1212   99.6%        1216       7.84 M/s       128 ns
design 2  BN=512    20480     1240   98.1%        1264       8.14 M/s       123 ns

## What it shows

**BN=512 issues MMAs faster at every shape** - 8.6% faster at 4096, still 3.8%
at 20480. This is the arithmetic-intensity advantage arriving where it actually
acts: 170.7 against 128 FLOP/byte means less time waiting on data-ready, hence
more issues per second. It does not show up as more compute per issue, because
`A` cannot vary.

**BN=512 pays it back in duty cycle.** With a single accumulator the drain is
fully exposed: 90.6% duty at 4096 against BN=256's 98.3%.

The two terms move in opposite directions, and the crossover falls out of the
arithmetic:

| shape | f | duty | net |
|--:|--:|--:|--:|
| 4096 | +8.6% | -7.8% | +0.2% |
| 8192 | +6.1% | -3.8% | +1.9% |
| 20480 | +3.8% | -1.5% | +2.3% |

The duty penalty decays as `1/K` while the issue-rate advantage persists, so
BN=512 goes from level with BN=256 at 4096 to ahead at large shapes - which is
exactly the crossover the main sweep shows around 10240.

## The headroom this identifies

A design with BN=512's issue rate and BN=256's duty cycle would reach:

| shape | BN=256 | BN=512 | both | headroom |
|--:|--:|--:|--:|--:|
| 4096 | 1309 | 1311 | **1422** | +8.5% |
| 8192 | 1341 | 1367 | **1422** | +4.0% |
| 12288 | 1281 | 1333 | 1367 | +2.5% |
| 20480 | 1212 | 1240 | 1259 | +1.5% |

That is precisely what design 3's `splitacc` chases: keep BN=512's operand
efficiency, recover the duty cycle by releasing half the accumulator early.

## Open

`f` is reported in absolute terms. Turning it into a fraction of the hardware
ceiling needs `f_max`, which an MMA-only microbenchmark would give directly -
preload one A/B tile into SMEM and issue back to back with no TMA at all - on
the same footing as these numbers, without relying on a published peak.
