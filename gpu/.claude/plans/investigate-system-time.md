# Điều tra: System time 80s trong GPU branch (không có --use_gpu)

## Triệu chứng

Command: `./mpboot-avx -s tree1.phy -seed 1 -numpars 100`
- GPU branch: User=20.31s, **System=80.43s**, Elapsed=1:40, Memory=22MB
- Original branch: ~4.38s tổng
- Không có page fault đáng kể → không phải memory pressure

## Phát hiện đến nay

### 1. `PROFILE_SCOPE("newviewParsimony")` trong hot loop

**File**: `/raid/home/loinguyen/workspace/mpboot-gpu/mpboot/sprparsimony.cpp`, dòng 561

```cpp
static void newviewParsimonyIterativeFast(pllInstance *tr, partitionList *pr, int perSiteScores)
{
  PROFILE_SCOPE("newviewParsimony");  // <-- ĐÂY
  ...
}
```

`PROFILE_SCOPE` expand thành:
```cpp
mpbootgpu::ProfilerTimer timer_561("newviewParsimony");
```

`ProfilerTimer` destructor gọi `Profiler::instance().addTime(name_, duration)`.
`addTime` gọi `getThreadData()`, trong đó có:

```cpp
ThreadLocalData& getThreadData() {
    thread_local ThreadLocalData local_data;
    thread_local bool registered = [this](ThreadLocalData* data) {
        std::lock_guard<std::mutex> lock(global_mutex_);  // MUTEX LOCK MỖI LẦN THREAD MỚI
        all_threads_data_.push_back(data);
        return true;
    }(&local_data);
    return local_data;
}
```

**`thread_local bool registered`**: Lambda chỉ chạy một lần per thread (first call).
Sau lần đầu, mutex KHÔNG lock nữa (thread_local initialized). Nên mutex không phải nguyên nhân.

`accumulated[name] += nanos` dùng `std::unordered_map` → hash lookup mỗi lần.

### 2. Tần suất gọi `newviewParsimonyIterativeFast`

Hàm này được gọi từ nhiều nơi trong SPR loop:
- `sprparsimony.cpp:897, 1053, 1606, 1739, 1938, 2210, 3179`

Với 100 cây, N=295 taxa, SPR:
- Mỗi iteration SPR: ~2*(2N-2) = ~1176 lần evaluateParsimony/newview per pass
- Nhiều passes cho đến convergence

Ước tính: 100 cây × ~1000 newview calls × nhiều passes = **hàng triệu lần**
→ `std::chrono::high_resolution_clock::now()` gọi 2 lần mỗi ProfilerTimer (ctor + dtor)
→ `clock_gettime(CLOCK_REALTIME)` → syscall! → **đây là nguồn gốc system time cao**

### 3. Có 2 định nghĩa `newviewParsimonyIterativeFast`

- Dòng 559-1384: Định nghĩa đầu tiên (Sankoff/special + fallthrough Fitch) — **CÓ PROFILE_SCOPE**
- Dòng 1386+: Định nghĩa thứ hai (plain Fitch) — **KHÔNG có PROFILE_SCOPE**

Với DNA data thông thường (pllCostMatrix = NULL):
- Định nghĩa đầu tiên: check `pllCostMatrix` → FALSE → fall through vào code Fitch bên trong
- Cần kiểm tra: code Fitch của định nghĩa đầu có hay không? (bị interrupt ở dòng ~900)

**TODO**: Đọc tiếp từ dòng 559-900 để xem với `pllCostMatrix=NULL` thì hàm đầu làm gì.

### 4. Các PROFILE_SCOPE khác (ít hot hơn)

- `phylotree.cpp:1139`: `PROFILE_SCOPE("computeParsimony")` — ít hot hơn
- `phyloanalysis.cpp:1724,1741,1808,1872`: Chỉ gọi 1 lần per run — OK
- `iqtree.cpp`: Chỉ trong search loop iteration level — không hot như newview

## Nguyên nhân chính (hypothesis)

`std::chrono::high_resolution_clock::now()` trên Linux thường dùng `clock_gettime(CLOCK_REALTIME)`.
Nếu VDSO không available hoặc bị bypass → kernel syscall → mỗi call = system time.

`newviewParsimonyIterativeFast` được gọi hàng triệu lần → 2 × hàng triệu `clock_gettime` syscalls
→ **80 giây system time**.

## Fix

### Fix đơn giản nhất: xóa PROFILE_SCOPE trong hot path

```cpp
// sprparsimony.cpp dòng 561 — XÓA dòng này:
PROFILE_SCOPE("newviewParsimony");
```

### Fix tốt hơn: guard bằng compile-time flag

```cpp
#ifdef ENABLE_PROFILING
  PROFILE_SCOPE("newviewParsimony");
#endif
```

Thêm `-DENABLE_PROFILING` vào CMake chỉ khi cần profile.

### Fix tốt nhất: dùng atomic counter thay chrono

Thay `ProfilerTimer` bằng simple call counter, không dùng chrono trong hot path.

## Files liên quan

- `/raid/home/loinguyen/workspace/mpboot-gpu/mpboot/sprparsimony.cpp` — dòng 559-561
- `/raid/home/loinguyen/workspace/mpboot-gpu/mpboot/gpu/include/profiler.hpp` — định nghĩa macro

## TODO còn lại

1. Đọc dòng 559-900 trong sprparsimony.cpp để confirm với `pllCostMatrix=NULL` thì fall vào đâu
2. Kiểm tra `phylotree.cpp:1139` — `computeParsimony` được gọi bao nhiêu lần
3. Verify bằng cách remove PROFILE_SCOPE và re-benchmark
4. Kiểm tra `alignment.cpp:423` — `PROFILE_SCOPE("readPhylip")` — chỉ gọi 1 lần, OK
