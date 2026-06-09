---
name: benchmark
description: Bắt buộc wrap mọi lệnh benchmark bằng /usr/bin/time -v để summarize.py đọc được elapsed time và peak memory.
---

## Ràng buộc bắt buộc

Mọi lần chạy benchmark — dù chỉ 1 dataset — phải dùng `/usr/bin/time -v`, KHÔNG dùng `time` builtin.
Output của `/usr/bin/time -v` ra **stderr**, phải có `2>&1` để capture vào log.

```bash
/usr/bin/time -v ./mpboot-avx -s <file.phy> [flags] > out.log 2>&1
```

## Tại sao bắt buộc

`summarize.py` đọc hai dòng này để tính speedup và memory:
```
Elapsed (wall clock) time (h:mm:ss or m:ss): 0:45.32
Maximum resident set size (kbytes): 145816
```
Thiếu một trong hai → cột `elapsed_s` hoặc `Speedup` ra `None` trong Excel.

## Params benchmark chuẩn (treebase GPU)

```bash
-use_gpu -seed 1 -sprdist 6 -gpu_device $DEV -gpu_worker 200
```
Thêm `-cost $COST` cho non-uniform. Không cần `-numpars` (default = 100, K2 workers = 200).
