# R1 summary (generated)

Runs: chat_ref_block1=20261118_ppv_r1_chat_ref_block1_run01, chat_ref_block2=20261119_ppv_r1_chat_ref_block2_run01, chat_ref_block3=20261119_ppv_r1_chat_ref_block3_run01, duplex_block1=20261118_ppv_r1_duplex_block1_run01, duplex_block2=20261118_ppv_r1_duplex_block2_run01, duplex_block3=20261119_ppv_r1_duplex_block3_run01

| phase | turns | answered | TTFA P50/P95 ms (N) | unit lag P50/P99 ms (N) | first-audio service P50 ms | stalls @80 ms |
|---|---|---|---|---|---|---|
| A pooled | 48 | 37 | 1361/1708 (37) | 72/429 (900) | 353 | 16/41 |
| B pooled | 48 | 38 | 1269/1706 (38) | 75/430 (900) | 367 | 14/43 |

TTFA parts (A pooled, N=37, early starts 0): (a) P50 319 ms, (b) P50 960 ms, (c) P50 350 ms, k counts {'0': 17, '1': 20}

Server stages (B pooled): top stage over speaking units: thinker

Request-response reference (A pooled): TTFA P50 724 ms, e2e P50 728 ms, audio P50 4.12 s, ok 144/144
