# Staggering the two accumulator panels by one k-tile

**Result: +0.5% at small shapes, -0.8% at large ones, -0.2% overall. Not a win,
but the model predicts both halves.**

## The idea

Design 3 (`BN=512` splitacc) drains its accumulator as two 256-column panels,
each with its own free barrier; `-splitdr` gives each its own data-ready signal
too, so panel 0 can drain while panel 1 is still accumulating.

This variant pushes that separation as far as it goes. Normally each k-tile
issues panel 0's MMA then panel 1's. Here panel 0 runs **one k-tile ahead** at
both ends of the loop:

```
head   b0(k=0), b0(k=1),      then b1(k=0), b1(k=1)
steady b0(k),   b1(k)
tail   b0(n-2), b0(n-1),      then b1(n-2), b1(n-1)
```

`tcgen05.commit` signals on completion of everything issued before it, so
moving panel 0's last MMA two issues earlier moves its data-ready a whole
k-tile earlier, and the epilogue starts draining panel 0 that much sooner. At
the head, panel 1's free-barrier wait no longer delays panel 0's first MMA.

The epilogue already supports this: it waits on `mbar_tmem_data_ready[load]`
per panel, pulls that panel's 256 columns into registers, and arrives at
`mbar_tmem_panel_free[load]` before any staging or stores, with a
`static_assert(LOAD_N == BN_PANEL_EPI)` enforcing one panel per outer
iteration. So the whole chain moves earlier, not just the signal.

## What it can possibly be worth

The stagger moves panel 0's completion earlier by exactly **one k-tile out of
`num_k`**. That is the ceiling, before any cost:

| shape | num_k (BK=64) | ceiling |
|---:|---:|---:|
| 3072 | 48 | 2.08% |
| 4096 | 64 | 1.56% |
| 8192 | 128 | 0.78% |
| 16384 | 256 | 0.39% |
| 20480 | 320 | 0.31% |

## Measurement

Node `ip-172-20-60-53.us-east-2.compute.internal`, 18 square shapes, BK=64 x GSM 8/12/16, `do_bench` 1000/1000 ms,
median of **5** shuffled passes. Base and staggered kernels ran in the same
process, interleaved in the same passes, so each pair subtracts directly. Every
kernel was checked against `torch.matmul` at every shape - this reorders MMA
issues and barrier signals, so correctness is not a formality.

| Shape | BK=64 GSM=8 base | BK=64 GSM=8 lead | BK=64 GSM=12 base | BK=64 GSM=12 lead | BK=64 GSM=16 base | BK=64 GSM=16 lead | torch.matmul |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3072³ | 1278 | 1286 | 1276 | 1284 | 1275 | 1283 | 1307 |
| 4096³ | 1275 | 1279 | 1264 | 1273 | 1283 | 1292 | 1295 |
| 5120³ | 1299 | 1303 | 1311 | 1315 | 1307 | 1318 | 1332 |
| 6144³ | 1344 | 1351 | 1364 | 1364 | 1359 | 1361 | 1361 |
| 7168³ | 1279 | 1278 | 1277 | 1280 | 1290 | 1297 | 1344 |
| 8192³ | 1352 | 1353 | 1357 | 1353 | 1358 | 1360 | 1335 |
| 9216³ | 1332 | 1338 | 1357 | 1356 | 1350 | 1353 | 1310 |
| 10240³ | 1349 | 1346 | 1343 | 1347 | 1353 | 1352 | 1314 |
| 11264³ | 1313 | 1314 | 1314 | 1317 | 1322 | 1322 | 1309 |
| 12288³ | 1332 | 1324 | 1344 | 1341 | 1348 | 1346 | 1369 |
| 13312³ | 1278 | 1283 | 1307 | 1305 | 1319 | 1318 | 1306 |
| 14336³ | 1266 | 1261 | 1297 | 1291 | 1309 | 1299 | 1318 |
| 15360³ | 1256 | 1250 | 1291 | 1281 | 1305 | 1299 | 1380 |
| 16384³ | 1274 | 1274 | 1287 | 1288 | 1306 | 1305 | 1379 |
| 17408³ | 1204 | 1173 | 1265 | 1260 | 1274 | 1274 | 1373 |
| 18432³ | 1179 | 1161 | 1257 | 1257 | 1271 | 1263 | 1297 |
| 19456³ | 1166 | 1155 | 1241 | 1219 | 1268 | 1266 | 1296 |
| 20480³ | 1149 | 1128 | 1238 | 1209 | 1268 | 1261 | 1354 |

## Stagger minus base

| Shape | BK=64 GSM=8 | BK=64 GSM=12 | BK=64 GSM=16 |
|---:|---:|---:|---:|
| 3072³ | +0.6% | +0.6% | +0.7% |
| 4096³ | +0.3% | +0.7% | +0.8% |
| 5120³ | +0.4% | +0.3% | +0.8% |
| 6144³ | +0.5% | +0.0% | +0.2% |
| 7168³ | -0.0% | +0.2% | +0.5% |
| 8192³ | +0.1% | -0.3% | +0.1% |
| 9216³ | +0.4% | -0.0% | +0.2% |
| 10240³ | -0.2% | +0.3% | -0.0% |
| 11264³ | +0.1% | +0.3% | -0.0% |
| 12288³ | -0.6% | -0.2% | -0.1% |
| 13312³ | +0.4% | -0.1% | -0.1% |
| 14336³ | -0.4% | -0.5% | -0.8% |
| 15360³ | -0.5% | -0.8% | -0.5% |
| 16384³ | +0.0% | +0.1% | -0.0% |
| 17408³ | -2.6% | -0.5% | -0.1% |
| 18432³ | -1.5% | +0.0% | -0.7% |
| 19456³ | -0.9% | -1.7% | -0.1% |
| 20480³ | -1.8% | -2.3% | -0.6% |
| **mean** | **-0.3%** | **-0.2%** | **+0.0%** |

## Analysis

**The benefit is real and lands where the ceiling allows it.**

| shapes | measured | mean ceiling |
|:---|---:|---:|
| 3072-6144 | **+0.49%** | 1.48% |
| 7168-14336 | -0.03% | 0.61% |
| 15360-20480 | **-0.81%** | 0.35% |

About a third of the theoretical maximum is realised at the small end, and the
benefit decays with shape exactly as 1/num_k requires. The rest of the ceiling
is presumably lost where the epilogue was not the thing waiting - if the MMA
warp is the bottleneck, an earlier data-ready changes nothing.

**The cost has a fingerprint.** Holding two TMA slots at each end of the k loop
shortens prefetch depth. If that is the mechanism, the damage should scale with
memory pressure - and it does, monotonically in GSM:

| mean delta, 15360-20480 | GSM=8 | GSM=12 | GSM=16 |
|:---|---:|---:|---:|
| | -1.22% | -0.87% | -0.33% |

Better L2 reuse means less TMA pressure, so the same lost prefetch depth costs
less. Measurement noise does not order itself by a swizzle parameter.

Benefit shrinks as 1/num_k, cost grows with memory pressure, and they cross
around 12288-14336.

## Conclusion

Not adopted: net -0.2% across the sweep, and negative exactly where design 3 is
otherwise strongest. Kept as a record because the two competing effects are both
predicted by the pipeline model rather than discovered empirically - the
benefit's ceiling from the k-loop structure, the cost from slot lifetime.

Nothing is registered as an autotune candidate.
