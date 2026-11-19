# R2 summary (generated)

Placement: {'expected_arms': ['ms8_1gpu'], 'why': 'max_sessions 8 started on one GPU and 8 sessions x 30 s had no failure', 'arms_run': ['ms8_1gpu'], 'consistent': True}

## ms8_1gpu

Serving capacity: below the lowest level; deadline capacity (1 s): below the lowest level

| level | blocks | miss rate | TTFA P50/P95/P99 ms (N) | unit lag P50/P99 ms (N) | deadline miss 1 s | target |
|---|---|---|---|---|---|---|
| 1 | 3/3 | 0.0056 | 935/N/A/N/A (18) | 279/624 (540) | 0.500 | no |
| 2 | 2/3 | 0.0000 | 1737/2043/N/A (22) | 79/564 (720) | 0.875 | no |
| 4 | 2/3 | 0.0014 | 1557/2068/N/A (44) | 119/734 (1440) | 0.922 | no |
| 8 | 2/3 | 0.4562 | 1755/2545/N/A (74) | 258/1169 (2193) | 0.940 | no |

First limiting stage: talker (neither saturated (uncertain))

