---
name: benchmark
description: Khi được giao nhiệm vụ benchmark, dùng /usr/bin/time -v để thu thập elapsed time và peak memory, đảm bảo dữ liệu đầy đủ cho summarize.py.
---

## Mục đích

Mỗi lần chạy benchmark GPU hoặc CPU, bắt buộc wrap lệnh bằng `/usr/bin/time -v` để log file
chứa đủ `Elapsed (wall clock) time` và `Maximum resident set size` — hai trường này được
`summarize.py` dùng để tính speedup và peak memory trong Excel.

## Khi nào dùng

Khi được giao bất kỳ nhiệm vụ benchmark nào, kể cả:
- Chạy thử nhanh một dataset
- Chạy sweep parameters
- Chạy full benchmark nhiều datasets

## Cách dùng

```bash
# Thay vì:
./mpboot-avx -s data.phy -use_gpu ... > out.log 2>&1

# Luôn dùng:
/usr/bin/time -v ./mpboot-avx -s data.phy -use_gpu ... > out.log 2>&1
```

Trong script benchmark (vòng lặp nhiều dataset):

```bash
for phy in $DATASETS; do
    name="$(basename "$phy" .phy)"
    /usr/bin/time -v "$BIN" -s "$phy" -use_gpu \
        -numpars 400 -sprdist 3 -gpu_stop 4 \
        > "$OUT_DIR/${name}.log" 2>&1
done
```

## Output cần có trong log

`summarize.py` đọc các dòng sau từ mỗi `.log`:

```
Elapsed (wall clock) time (h:mm:ss or m:ss): 0:45.32
Maximum resident set size (kbytes): 145816
```

Cả hai chỉ xuất hiện khi dùng `/usr/bin/time -v`. Thiếu hai dòng này:
- `elapsed_s` → None trong Excel
- `Speedup` → không tính được

## Lưu ý

- `/usr/bin/time -v` khác với shell builtin `time` (bash/zsh) — phải chỉ rõ đường dẫn đầy đủ.
- Trên macOS dùng `gtime -v` (GNU time từ `brew install gnu-time`).
- Output của `/usr/bin/time -v` đi vào **stderr**, nên cần `2>&1` để redirect vào log file.
