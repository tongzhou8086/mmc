# Cluster launch control: measurements

B200, 148 SMs (74 cluster slots), one node, one process per run: absolute
TFLOP/s drift a few percent between nodes, so only the paired deltas below are
meaningful. Every kernel is checked against torch.matmul at every shape before
timing. `triton.testing.do_bench` median, warmup and rep 1 s, three shuffled
repetitions per entry.

* `clc1` / `clc2` / `clc3` - design 4 plus cluster launch control, claiming 1 /
  2 / 3 tiles ahead of the consumers.
* `fullgrid` - the *design 4* kernel launched with the CLC grid (one cluster per
  tile). Its static stride then equals the cluster count, so each cluster runs
  exactly one tile. This is the control that separates "the grid got bigger"
  from "clusters actually stole work".
* `2round` - design 4 as it ships, one persistent cluster per SM pair.

## What it says

1. The full-grid control is neutral everywhere (within +-0.6%), so the CLC
   deltas are work stealing, not grid shape.
2. Claiming one tile ahead is worth +1.5% to +3.6% over design 4 at every shape
   from 4096 up, in every one of the six BK x GSM configs.
3. 3072 is flat, as it must be: 72 clusters for 74 slots is a single
   under-filled wave, so there is nothing to steal. That shape is limited by
   occupancy, not by quantization.
4. Claim depth 3 collapses at 5120 (-6% to -7%, reproduced three times). 5120
   is 2.70 waves, so a cluster that runs three tiles ahead of its consumers
   hoards work at the tail while its neighbours go idle. The pathology needs
   tiles-per-cluster to be near the claim depth: 4096 and 6144 are unaffected.
5. Against cuBLAS the gap narrows but does not close on the small shapes:
   -0.7% at 3072, -1.1% at 4096, -1.3% at 5120, -0.9% at 7168, and wins at
   6144 (+1.1%) and 8192 (+2.3%).

## clc1

| Shape | BK=64 GSM=8 | BK=64 GSM=12 | BK=64 GSM=16 | BK=128 GSM=8 | BK=128 GSM=12 | BK=128 GSM=16 | torch.matmul |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3072³ | 1307 | 1303 | 1306 | 1299 | 1290 | 1295 | 1316 |
| 4096³ | 1297 | 1288 | 1298 | 1266 | 1263 | 1274 | 1313 |
| 5120³ | 1318 | 1328 | 1326 | 1306 | 1316 | 1304 | 1346 |
| 6144³ | 1391 | 1404 | 1400 | 1379 | 1409 | 1395 | 1394 |
| 7168³ | 1338 | 1337 | 1346 | 1318 | 1318 | 1330 | 1358 |
| 8192³ | 1391 | 1395 | 1397 | 1386 | 1390 | 1397 | 1365 |

## clc2

| Shape | BK=64 GSM=8 | BK=64 GSM=12 | BK=64 GSM=16 | BK=128 GSM=8 | BK=128 GSM=12 | BK=128 GSM=16 | torch.matmul |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3072³ | 1307 | 1304 | 1305 | 1299 | 1294 | 1294 | 1316 |
| 4096³ | 1289 | 1284 | 1299 | 1266 | 1267 | 1273 | 1313 |
| 5120³ | 1312 | 1325 | 1315 | 1299 | 1309 | 1300 | 1346 |
| 6144³ | 1389 | 1407 | 1393 | 1378 | 1409 | 1401 | 1394 |
| 7168³ | 1333 | 1337 | 1345 | 1320 | 1323 | 1333 | 1358 |
| 8192³ | 1388 | 1393 | 1389 | 1388 | 1393 | 1401 | 1365 |

## clc3

| Shape | BK=64 GSM=8 | BK=64 GSM=12 | BK=64 GSM=16 | BK=128 GSM=8 | BK=128 GSM=12 | BK=128 GSM=16 | torch.matmul |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3072³ | 1303 | 1307 | 1306 | 1294 | 1289 | 1289 | 1316 |
| 4096³ | 1293 | 1287 | 1298 | 1266 | 1267 | 1273 | 1313 |
| 5120³ | 1194 | 1203 | 1209 | 1183 | 1191 | 1174 | 1346 |
| 6144³ | 1379 | 1391 | 1385 | 1380 | 1396 | 1392 | 1394 |
| 7168³ | 1326 | 1331 | 1338 | 1314 | 1318 | 1328 | 1358 |
| 8192³ | 1386 | 1385 | 1391 | 1385 | 1392 | 1398 | 1365 |

## fullgrid

| Shape | BK=64 GSM=8 | BK=64 GSM=12 | BK=64 GSM=16 | BK=128 GSM=8 | BK=128 GSM=12 | BK=128 GSM=16 | torch.matmul |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3072³ | 1306 | 1308 | 1307 | 1298 | 1296 | 1295 | 1316 |
| 4096³ | 1261 | 1259 | 1262 | 1232 | 1228 | 1240 | 1313 |
| 5120³ | 1284 | 1292 | 1290 | 1259 | 1274 | 1263 | 1346 |
| 6144³ | 1356 | 1372 | 1358 | 1345 | 1365 | 1357 | 1394 |
| 7168³ | 1300 | 1301 | 1310 | 1282 | 1283 | 1300 | 1358 |
| 8192³ | 1372 | 1374 | 1375 | 1359 | 1369 | 1374 | 1365 |

## 2round

| Shape | BK=64 GSM=8 | BK=64 GSM=12 | BK=64 GSM=16 | BK=128 GSM=8 | BK=128 GSM=12 | BK=128 GSM=16 | torch.matmul |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3072³ | 1305 | 1308 | 1306 | 1301 | 1294 | 1291 | 1316 |
| 4096³ | 1263 | 1260 | 1262 | 1234 | 1228 | 1241 | 1313 |
| 5120³ | 1284 | 1295 | 1292 | 1261 | 1275 | 1264 | 1346 |
| 6144³ | 1358 | 1370 | 1365 | 1348 | 1370 | 1359 | 1394 |
| 7168³ | 1302 | 1301 | 1310 | 1283 | 1284 | 1298 | 1358 |
| 8192³ | 1370 | 1372 | 1374 | 1359 | 1367 | 1376 | 1365 |

## best of each design, and the wave picture

| Shape | clusters | waves | clc1 | clc2 | clc3 | fullgrid | 2round | cuBLAS | best clc vs 2round |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3072³ | 72 | 0.97 | 1307 | 1307 | 1307 | 1308 | 1308 | 1316 | -0.1% |
| 4096³ | 128 | 1.73 | 1298 | 1299 | 1298 | 1262 | 1263 | 1313 | +2.9% |
| 5120³ | 200 | 2.70 | 1328 | 1325 | 1209 | 1292 | 1295 | 1346 | +2.6% |
| 6144³ | 288 | 3.89 | 1409 | 1409 | 1396 | 1372 | 1370 | 1394 | +2.9% |
| 7168³ | 392 | 5.30 | 1346 | 1345 | 1338 | 1310 | 1310 | 1358 | +2.8% |
| 8192³ | 512 | 6.92 | 1397 | 1401 | 1398 | 1375 | 1376 | 1365 | +1.8% |

## clc minus 2round, paired within each config (%)

| Config | 3072³ | 4096³ | 5120³ | 6144³ | 7168³ | 8192³ |
|---:|---:|---:|---:|---:|---:|---:|
| clc1 BK=64 GSM=8 | +0.1% | +2.7% | +2.7% | +2.4% | +2.8% | +1.6% |
| clc1 BK=64 GSM=12 | -0.3% | +2.2% | +2.6% | +2.5% | +2.8% | +1.7% |
| clc1 BK=64 GSM=16 | -0.0% | +2.8% | +2.6% | +2.6% | +2.8% | +1.6% |
| clc1 BK=128 GSM=8 | -0.1% | +2.6% | +3.6% | +2.3% | +2.8% | +2.0% |
| clc1 BK=128 GSM=12 | -0.3% | +2.8% | +3.2% | +2.8% | +2.7% | +1.7% |
| clc1 BK=128 GSM=16 | +0.3% | +2.6% | +3.2% | +2.6% | +2.5% | +1.5% |
| clc2 BK=64 GSM=8 | +0.1% | +2.1% | +2.2% | +2.2% | +2.4% | +1.4% |
| clc2 BK=64 GSM=12 | -0.3% | +1.9% | +2.3% | +2.7% | +2.8% | +1.5% |
| clc2 BK=64 GSM=16 | -0.1% | +2.9% | +1.8% | +2.0% | +2.7% | +1.1% |
| clc2 BK=128 GSM=8 | -0.1% | +2.6% | +3.1% | +2.2% | +2.9% | +2.1% |
| clc2 BK=128 GSM=12 | +0.0% | +3.2% | +2.6% | +2.9% | +3.0% | +1.9% |
| clc2 BK=128 GSM=16 | +0.2% | +2.5% | +2.9% | +3.1% | +2.7% | +1.8% |
| clc3 BK=64 GSM=8 | -0.2% | +2.4% | -7.0% | +1.6% | +1.9% | +1.2% |
| clc3 BK=64 GSM=12 | -0.1% | +2.2% | -7.1% | +1.5% | +2.3% | +1.0% |
| clc3 BK=64 GSM=16 | +0.0% | +2.8% | -6.4% | +1.4% | +2.2% | +1.2% |
| clc3 BK=128 GSM=8 | -0.5% | +2.6% | -6.2% | +2.3% | +2.5% | +1.9% |
| clc3 BK=128 GSM=12 | -0.3% | +3.1% | -6.6% | +1.9% | +2.7% | +1.8% |
| clc3 BK=128 GSM=16 | -0.2% | +2.5% | -7.1% | +2.4% | +2.3% | +1.6% |
| fullgrid BK=64 GSM=8 | +0.0% | -0.1% | -0.0% | -0.1% | -0.1% | +0.2% |
| fullgrid BK=64 GSM=12 | -0.0% | -0.1% | -0.2% | +0.2% | -0.0% | +0.1% |
| fullgrid BK=64 GSM=16 | +0.1% | -0.0% | -0.2% | -0.6% | +0.0% | +0.0% |
| fullgrid BK=128 GSM=8 | -0.2% | -0.1% | -0.2% | -0.2% | -0.0% | -0.0% |
| fullgrid BK=128 GSM=12 | +0.1% | +0.0% | -0.1% | -0.4% | -0.1% | +0.2% |
| fullgrid BK=128 GSM=16 | +0.3% | -0.1% | -0.1% | -0.2% | +0.2% | -0.2% |
