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

## Why a two-kernel prototype was not used

A second launch can only carve rectangles, and a rectangle's tile count is
always a multiple of `ceil(M/256)` — 16 at 4096 — so it cannot isolate the
34-tile tail. Optimizing over every rectangle split at 4096 and 5120 returns
`N_A = 0`, i.e. it degenerates to uniform BN=128, which reaches the same
3.5 / 5.5 tile-times as the ideal tail split. Hence uniform BN=128 was the
cheaper way to measure the same thing.
