---
name: log-bug
description: Ghi lại bug đã tìm và fix vào changelog để phục vụ viết khóa luận. Dùng sau khi xác nhận root cause và áp dụng fix thành công.
---

## Mục đích

Khi một bug trong task GPU parsimony được tìm ra và fix xong, ghi lại thông tin vào
`gpu/.claude/docs/changelog.md` theo định dạng chuẩn để sau này dùng làm tài liệu viết khóa luận.

## Khi nào dùng

- Sau khi đã xác nhận root cause của bug (không phải hypothesis)
- Sau khi fix đã được verify (qua output, test, hoặc so sánh với CPU reference)
- Mỗi bug ghi một entry riêng, kể cả khi fix nhiều bug trong cùng một task

## Cách ghi

Mở file `gpu/.claude/docs/changelog.md` và **append** entry mới ở cuối, theo template sau:

```markdown
---

## Bug #<N> — <tên ngắn gọn mô tả bug>

**Ngày**: YYYY-MM-DD
**Task**: <tên task hoặc bước pipeline, ví dụ: SPR hill-climbing, stepwise addition, recomputeAllNodes>
**File liên quan**: `<file>:<dòng>` (có thể liệt kê nhiều)

### Triệu chứng

<Mô tả output sai quan sát được — số, print, crash, divergence so với CPU>

### Root cause

<Giải thích cơ chế gây bug — càng chi tiết càng tốt.
Bao gồm: biến nào sai, tại sao sai, điều kiện nào trigger.
Không viết hypothesis — chỉ ghi nguyên nhân đã được xác nhận.>

### Fix

<Code diff hoặc mô tả thay đổi cụ thể. Ví dụ:
- Trước: ...
- Sau: ...
Giải thích tại sao fix này giải quyết được root cause.>

### Bài học / Ghi chú cho khóa luận

<Điều gì về GPU/CUDA/thuật toán parsimony có thể rút ra từ bug này?
Ví dụ: khác biệt giữa CPU xPars lazy evaluation vs GPU eager recompute, warp divergence trap, index off-by-one trong vface mapping.>
```

## Quy tắc

- `<N>` trong `Bug #<N>` là số thứ tự tăng dần theo thứ tự ghi vào file (đếm từ entry đã có).
- Không xoá hay sửa entry cũ — chỉ append.
- Nếu một fix sau này hóa ra sai và phải sửa lại, ghi một entry mới thay vì sửa entry cũ.
- Dùng code block (``` ``` ```) cho mọi đoạn code hoặc output terminal.
- Ngày dùng định dạng ISO 8601: YYYY-MM-DD.