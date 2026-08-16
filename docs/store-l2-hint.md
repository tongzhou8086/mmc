# An L2 evict_first hint on the epilogue TMA store

**Result: no measurable effect. Recorded so the question stays answered.**

## The idea

Borrowed from ThunderKittens' B200 GEMM, which tags its output store
`cache_policy::EVICT_FIRST`. C is written once and never read again by the
kernel, so with the default policy its lines sit in L2 competing with the A/B
panels that neighbouring CTAs are about to reuse - the working set GSM exists
to protect. At 20480³ that is 839 MB of C streaming through the cache whose
hit rate the whole GSM sweep turned on.

The change is confined to one instruction. Our epilogue store becomes

```
cp.async.bulk.tensor.2d.global.shared::cta.bulk_group.L2::cache_hint
    [tmap, {x, y}], [smem], policy;
```

with `policy` from `createpolicy.fractional.L2::evict_first.b64 %0, 1.0`.
Nothing else differs: all six variants compile to the same 135 registers as
their parents, with no spills.

## Measurement

Design 3 (`BN=512` splitacc) at BK 64/128 x GSM 8/12/16, the 18 square shapes,
node `ip-172-20-41-183.us-east-2.compute.internal`. Base and hinted kernels ran **in the same process, interleaved in
the same shuffled passes**, so each pair subtracts directly; `do_bench` with
1000/1000 ms, median of 3 passes. Every kernel was checked against
`torch.matmul` at every shape - a cache hint must not change results.

| Shape | BK=64 GSM=8 base | BK=64 GSM=8 hint | BK=64 GSM=12 base | BK=64 GSM=12 hint | BK=64 GSM=16 base | BK=64 GSM=16 hint | BK=128 GSM=8 base | BK=128 GSM=8 hint | BK=128 GSM=12 base | BK=128 GSM=12 hint | BK=128 GSM=16 base | BK=128 GSM=16 hint | torch.matmul |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3072³ | 1349 | 1341 | 1342 | 1348 | 1341 | 1346 | 1256 | 1255 | 1252 | 1254 | 1254 | 1253 | 1372 |
| 4096³ | 1321 | 1327 | 1319 | 1322 | 1337 | 1335 | 1271 | 1277 | 1264 | 1270 | 1284 | 1287 | 1346 |
| 5120³ | 1349 | 1345 | 1356 | 1360 | 1354 | 1354 | 1305 | 1304 | 1311 | 1315 | 1317 | 1315 | 1386 |
| 6144³ | 1401 | 1403 | 1421 | 1419 | 1413 | 1410 | 1367 | 1363 | 1400 | 1398 | 1383 | 1389 | 1412 |
| 7168³ | 1323 | 1323 | 1326 | 1326 | 1342 | 1342 | 1293 | 1291 | 1298 | 1301 | 1324 | 1328 | 1401 |
| 8192³ | 1384 | 1381 | 1390 | 1398 | 1389 | 1396 | 1377 | 1382 | 1391 | 1392 | 1404 | 1412 | 1373 |
| 9216³ | 1364 | 1361 | 1384 | 1387 | 1373 | 1372 | 1360 | 1369 | 1404 | 1404 | 1402 | 1402 | 1362 |
| 10240³ | 1358 | 1355 | 1362 | 1350 | 1350 | 1360 | 1377 | 1372 | 1387 | 1390 | 1400 | 1396 | 1353 |
| 11264³ | 1305 | 1303 | 1329 | 1318 | 1334 | 1340 | 1332 | 1332 | 1366 | 1368 | 1381 | 1380 | 1348 |
| 12288³ | 1317 | 1318 | 1330 | 1330 | 1337 | 1337 | 1368 | 1367 | 1401 | 1403 | 1410 | 1409 | 1417 |
| 13312³ | 1300 | 1299 | 1311 | 1312 | 1322 | 1322 | 1349 | 1352 | 1382 | 1380 | 1392 | 1392 | 1374 |
| 14336³ | 1296 | 1301 | 1308 | 1308 | 1322 | 1320 | 1364 | 1365 | 1389 | 1389 | 1404 | 1407 | 1362 |
| 15360³ | 1284 | 1289 | 1304 | 1306 | 1323 | 1323 | 1366 | 1364 | 1397 | 1397 | 1407 | 1406 | 1426 |
| 16384³ | 1307 | 1300 | 1320 | 1318 | 1329 | 1332 | 1360 | 1363 | 1371 | 1382 | 1418 | 1419 | 1427 |
| 17408³ | 1273 | 1267 | 1307 | 1307 | 1316 | 1319 | 1358 | 1363 | 1390 | 1391 | 1401 | 1399 | 1421 |
| 18432³ | 1265 | 1262 | 1289 | 1294 | 1299 | 1297 | 1351 | 1359 | 1382 | 1383 | 1384 | 1394 | 1364 |
| 19456³ | 1235 | 1229 | 1273 | 1280 | 1299 | 1303 | 1355 | 1358 | 1382 | 1374 | 1400 | 1392 | 1358 |
| 20480³ | 1237 | 1235 | 1264 | 1271 | 1301 | 1301 | 1359 | 1357 | 1376 | 1375 | 1396 | 1395 | 1423 |

## Hint minus base

| Shape | BK=64 GSM=8 | BK=64 GSM=12 | BK=64 GSM=16 | BK=128 GSM=8 | BK=128 GSM=12 | BK=128 GSM=16 |
|---:|---:|---:|---:|---:|---:|---:|
| 3072³ | -0.6% | +0.5% | +0.4% | -0.1% | +0.2% | -0.1% |
| 4096³ | +0.4% | +0.2% | -0.1% | +0.4% | +0.5% | +0.2% |
| 5120³ | -0.3% | +0.3% | +0.0% | -0.0% | +0.3% | -0.1% |
| 6144³ | +0.1% | -0.2% | -0.2% | -0.3% | -0.2% | +0.5% |
| 7168³ | +0.1% | -0.0% | +0.0% | -0.1% | +0.2% | +0.3% |
| 8192³ | -0.2% | +0.5% | +0.5% | +0.4% | +0.1% | +0.6% |
| 9216³ | -0.3% | +0.2% | -0.1% | +0.6% | -0.0% | -0.0% |
| 10240³ | -0.2% | -0.9% | +0.7% | -0.4% | +0.2% | -0.3% |
| 11264³ | -0.1% | -0.8% | +0.5% | -0.0% | +0.2% | -0.1% |
| 12288³ | +0.0% | +0.0% | +0.0% | -0.1% | +0.1% | -0.1% |
| 13312³ | -0.0% | +0.0% | +0.0% | +0.2% | -0.1% | -0.0% |
| 14336³ | +0.4% | -0.0% | -0.1% | +0.0% | -0.0% | +0.2% |
| 15360³ | +0.3% | +0.2% | +0.0% | -0.1% | -0.0% | -0.1% |
| 16384³ | -0.6% | -0.1% | +0.2% | +0.2% | +0.8% | +0.1% |
| 17408³ | -0.5% | -0.1% | +0.3% | +0.4% | +0.1% | -0.2% |
| 18432³ | -0.3% | +0.3% | -0.2% | +0.6% | +0.1% | +0.7% |
| 19456³ | -0.5% | +0.5% | +0.3% | +0.3% | -0.6% | -0.6% |
| 20480³ | -0.2% | +0.5% | +0.1% | -0.2% | -0.1% | -0.1% |
| **mean** | **-0.1%** | **+0.1%** | **+0.1%** | **+0.1%** | **+0.1%** | **+0.1%** |

## Analysis

**It does nothing.** Across 108 paired measurements: 48 positive, 41 negative,
19 exactly zero; mean +0.05%, standard deviation 0.32%, largest single
deviation 0.9%. That is the signature of measurement noise, not of a mechanism.

The stronger evidence is that the mechanism's own prediction fails. If the hint
worked by protecting the A/B working set, the benefit would grow with problem
size, since C's volume grows as M x N while L2 stays fixed. It does not:

| | mean delta |
|:---|---:|
| six smallest shapes (3072-8192) | +0.12% |
| six largest shapes (15360-20480) | +0.04% |

If anything the ordering is backwards, and both figures are inside the noise.

Why it might not matter here, none of which this experiment distinguishes:

- Bulk TMA stores may already be treated as streaming writes that do not
  aggressively allocate in L2, making the hint redundant.
- The reuse distance GSM creates may be short enough that A/B panels are hit
  again before C's lines could have evicted them.
- C's lines may be evicted promptly under plain LRU anyway, precisely because
  nothing ever touches them again.

**Conclusion:** not worth adopting. The six variants are kept as sources and
cubins so the result is reproducible, but none is registered as an autotune
candidate.

This does leave the post's 「只使用两种必备的常规优化」 claim intact - the one
borrowed optimization that would have needed a footnote turns out not to earn
one on our kernels.
