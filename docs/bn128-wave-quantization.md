# BN=128: trading arithmetic intensity for wave quantization

**Result: it does not pay off uniformly. Recorded as a negative result, plus the
number it produced — which is what the split-N tail design needs.**

## The idea

The BF16 kernels are persistent: the grid is `sm_count - sm_count % 2` = 148
CTAs on B200, so **74 clusters** are resident, each owning a `256 x BN` output
tile. Runtime is `ceil(T / 74)` tile-times for `T = ceil(M/256) * ceil(N/BN)`
tiles, so the last wave is partial and its idle clusters are pure loss.

Halving BN doubles the tile count and makes that last wave finer-grained. At
4096 it is worth a lot:

| shape | BN=256 | BN=128 |
|---:|---:|---:|
| 3072 | 97.3% | 97.3% |
| 4096 | **86.5%** | **98.8%** |
| 5120 | 90.1% | 98.3% |
| 6144 | 97.3% | 97.3% |

The cost is arithmetic intensity: `AI = 256*BN/(256+BN)` FLOP/byte, so BN=128
gets 85.3 against BN=256's 128.0 — two thirds. The question was whether the
quantization gain outweighs it. It does not.

## Measurement

Node `ip-172-20-60-53`, `--tuning-window 2`, 11 correctness tests passed.
TFLOP/s:

| shape | bf16-double-ns6-store2-bk64 (BN=256) | ns8-bn128 | ns6-bn128 | bn512 |
|---:|---:|---:|---:|---:|
| 3072 | 1315.9 | 1155.6 | 1069.0 | 1317.8 |
| 4096 | 1302.7 | 1232.1 | 1145.9 | 1302.7 |
| 5120 | 1330.1 | 1219.1 | 1134.8 | 1318.1 |
| 6144 | 1360.7 | 1211.3 | 1158.3 | 1377.0 |

BN=128 loses everywhere. At 4096 it carries a 12.3-point quantization advantage
and still lands 5.4% behind.

## The number worth keeping

Dividing wave efficiency out of both sides isolates the per-tile penalty:

| shape | BN=256 ideal | BN=128 ideal | penalty |
|---:|---:|---:|---:|
| 3072 | 1352 | 1188 | 12.2% |
| 4096 | 1506 | 1247 | 17.2% |
| 5120 | 1476 | 1240 | 16.0% |
| 6144 | 1398 | 1245 | 11.0% |

3072 and 6144 are the clean reads — both tile sizes have identical quantization
there, so the ratio is the penalty directly: **11-12%**. Mean over all four is
14%.

Two side findings:

- **Pipeline depth is worth a lot.** NS=8 beats NS=6 by 7.5% at 4096. BN=128's
  halved B slot is what makes NS=8 fit in the same 230400 bytes NS=6 needs at
  BN=256, so the deeper variant is free.
- Nothing here beats the existing candidates at any shape, so the autotune
  candidate set is unchanged. These two specs are recorded, not promoted.

## What it implies for the split-N tail

Splitting **N** (unlike splitting K) yields complete, independent output tiles:
no partial accumulators, no fp32 workspace, no cross-CTA reduction. So a hybrid
that runs the bulk at BN=256 and re-tiles only the tail wave at BN=128 pays the
14% penalty on the tail alone rather than on all the work:

| shape | baseline | projected hybrid | gain | tail share |
|---:|---:|---:|---:|---:|
| 4096 | 4.00 waves | 3.58 | **+10.4%** | 13% |
| 5120 | 6.00 | 5.58 | **+7.0%** | 7% |
| 7168 | 11.00 | 11.16 | **-1.5%** | 5% |
| 11264 | 27.00 | 26.58 | +1.5% | 0% |

The condition for the split to pay is `2R <= 74` for `R = T mod 74` — the tail
wave must be less than half full, or the two half-tiles still need two passes.
7168 sits just past that boundary (R=37) and goes negative, so a hybrid needs a
runtime guard, not an unconditional split.

Prior art in this repo: the MXFP8 `single-ns4-store3-bk128-bn384-*` kernels
already run one N=256 MMA plus one N=128 MMA per K step, and `-splitacc2`
aligns its epilogue to those N groups. The narrower MMA plus matching epilogue
path a tail split needs is therefore already solved on the MXFP8 side.

## Why a two-kernel prototype was not used *for this measurement*

A second launch can only carve rectangles, and a rectangle's tile count is
always a multiple of `ceil(M/256)` — 16 at 4096 — so it cannot isolate the
34-tile tail. Optimizing over every rectangle split at 4096 and 5120 with a
BN=256 bulk returns `N_A = 0`, i.e. it degenerates to uniform BN=128, which
reaches the same 3.5 / 5.5 tile-times as the ideal tail split. Hence uniform
BN=128 was the cheaper way to measure the same thing.

This is a statement about these two shapes with a BN=256 bulk, not about two-
kernel splits in general: against a BN=512 bulk they capture the full ideal at
7168 and 12288. See the last section.

## The general form: BN as a per-wave choice

The narrow version above pairs one bulk width with one tail width. The general
statement is that **BN need not be constant across the grid at all**: use the
widest tile where throughput dominates (the bulk, where arithmetic intensity is
everything) and the finest where balance dominates (the tail, a few percent of
the work, where granularity is everything). Those two pressures want opposite
things and they act on disjoint parts of the execution, so there is no reason
one BN should serve both.

The ladder is cheap at the top and expensive at the bottom. Relative per-tile
efficiency, taking BN=512 as 1.0: BN=256 is ~0.97 (AI 128 vs 171), BN=128 is
~0.835 (the 14% penalty measured above, compounded). So descending one rung is
nearly free and descending two is not.

Hanging the ladder off BN=512 — the design that actually wins at large shapes —
gains at 12 of 18 shapes, mean 2.1%:

| shape | tail tiles | best tail width | gain |
|---:|---:|---:|---:|
| 7168 | 22 | BN=256 | **+8.1%** |
| 4096 | 54 | BN=128 | +5.1% |
| 11264 | 6 | BN=128 | +5.0% |
| 5120 | 52 | BN=128 | +3.4% |
| 14336 | 14 | BN=128 | +3.2% |
| 13312 | 20 | BN=256 | +2.6% |
| 15360 | 24 | BN=256 | +1.9% |
| 3072, 6144, 8192, 9216, 10240 | 66-72 | - | 0% |

Two things this shows that the BN=256/128 pairing does not:

- **7168 is fixed by a BN=256 tail, not a BN=128 one.** It was BN=512's worst
  shape (88.3% wave efficiency); one rung down recovers 8.1% without paying the
  BN=128 penalty. Roughly half the shapes want a 256 tail and half want 128,
  which is precisely the part that has to be dynamic.
- **It applies on top of the winning design.** The BN=256 + BN=128 pairing sits
  under design 1, which loses to cuBLAS at every shape; BN=512 + a tail rung
  sits under the design that wins.

The fully general form lets the last wave itself be mixed-width - some 256 and
some 128 tiles chosen to fill exactly 74 clusters. That turns the whole thing
into a small makespan minimization over tile widths, and the `2R <= 74` guard
stops being a special case: it falls out of the optimization.

### Two launches capture part of this with no new CUDA

A second kernel launch can only carve rectangles, so it cannot express an
arbitrary tail. But against a BN=512 bulk it does better than expected:

| shape | baseline | ideal | two launches | split | fraction of ideal captured |
|---:|---:|---:|---:|---:|---:|
| 7168 | 6.00 | 5.52 | **5.52** | N_A=6656, tail BN=256 | **100%** |
| 12288 | 16.00 | 15.90 | 15.90 | N_A=11776, tail BN=128 | 100% |
| 13312 | 19.00 | 18.52 | 18.55 | N_A=12288, tail BN=256 | 94% |
| 15360 | 25.00 | 24.52 | 24.58 | N_A=13824, tail BN=256 | 87% |
| 11264 | 14.00 | 13.30 | 13.50 | N_A=10240, tail BN=128 | 72% |
| 4096, 5120, 16384, 18432 | - | - | no gain | - | 0% |

It needs no kernel changes: `_bf16_map` takes `global_dim` and `global_strides`
independently, so a column slice of B is `pointer + c0*2`, `global_dim=[w, K]`,
`global_strides=[N*2]` - the row pitch is unchanged. C slices the same way. So
the prototype is two launches of kernels already shipped, and 7168 - the largest
single win available - is fully expressible that way.

## Prototype: two launches, measured

`benchmarks/prototype_adaptive_bn.py` runs the split for real — one launch of
the BN=512 kernel over `C[:, :N_A]`, one of a narrower kernel over the rest.
`_bf16_map` grew optional `pitch` / `col_offset` so a column slice is described
without copying; `Runtime.launch_bf16_slice` launches any registered kernel over
an N range. Nothing is wired into `mmc.matmul` dispatch.

Node `ip-172-20-60-53`, `do_bench` with 1000/1000 ms:

| shape | split | uniform BN=512 | two launches | modelled | measured |
|---:|:---|---:|---:|---:|---:|
| 7168 | 6656 + 512 @ BN=256 | 1298.7 | **1351.3** | +8.1% | **+3.9%** |
| 20480 | 18944 + 1536 @ BN=256 | 1279.1 | **1312.8** | +0.9% | **+2.6%** |
| 13312 | 12288 + 1024 @ BN=256 | 1350.7 | **1372.5** | +2.4% | **+1.6%** |
| 11264 | 10240 + 1024 @ BN=128 | 1328.3 | **1345.1** | +3.6% | +1.2% |
| 15360 | 13824 + 1536 @ BN=256 | 1347.4 | **1356.0** | +1.7% | +0.6% |
| 12288 | 11776 + 512 @ BN=128 | **1378.9** | 1366.1 | +0.6% | -0.9% |

**Correctness: max abs error 0 at every shape** — the two launches reproduce
`torch.matmul` bit-for-bit, so the sliced descriptors are right.

The idea works: gains at 5 of 6 shapes, best +3.9% at 7168, which is the shape
BN=512 was worst at. But the measured gain is roughly **half** the model, and
the model's ranking is not reliable — 20480 beat its prediction (+2.6% vs +0.9%)
while 12288 went slightly negative.

Two costs the model omits, both plausible causes of the shortfall:

- **The tail launch pays its own pipeline ramp.** It fills NS buffers and drains
  them for a tile column that is only 512-1536 wide, and the model prices the
  tail purely as fractional-width work.
- **No overlap between the launches.** Stream ordering means the tail kernel
  cannot start until the bulk kernel has fully drained, so the drain of the last
  full wave is exposed rather than hidden.

At 7168 the model predicted saving 45 us and 22 us was realised, so roughly 23 us
went to those two effects — the right order of magnitude for a ramp plus a hard
barrier at this K.

4096, 5120 and 6144 are absent because the planner finds no split for them: with
a BN=512 bulk their tails are 54, 52 and 66 tiles, all past the `2R <= 74` guard,
so re-tiling still needs two passes. With a BN=256 bulk 5120 becomes feasible
(R=30) but the model gives +0.1%. These are precisely the shapes two launches
cannot help, and only a fused rebalance could.

Caveats: one run per configuration, so the sub-1% entries are inside run-to-run
noise; and the bulk kernel here is `bf16-single-ns4-store2-bk64-bn512` at GSM=8
and BK=64, not the BK=128 / GSM=16 configuration that wins in the main sweep
(~1470 vs the ~1300 baseline here). Wave quantization does not depend on BK or
GSM, so the model is unchanged, but the measured gain should shift: a faster
bulk makes the tail's fixed ramp relatively more expensive, and splitting N
changes `grid_n`, which changes what the GSM swizzle keeps in L2. Re-running
against the tuned bulk is the open item.

Where this leaves the idea: a fused in-kernel tail would avoid both omitted
costs — no second ramp, no barrier — so the gap between +3.9% and +8.1% is
roughly what fusing is worth at 7168. That is the case for building it, and
these numbers are the baseline to beat.
