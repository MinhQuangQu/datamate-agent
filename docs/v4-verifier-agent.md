# v4 — Verifier agent khi bất đồng (2026-10-31)

**Kết quả:** train sub300 PA **0.870** (269 / 8 / 23) · public **0.433**.
File nộp: `submissions/public_v4_2026-10-31.jsonl` (482 number, 141 percent, 48 list, 48 string, 36 date, 10 table,
9 boolean, 17 unanswerable, 209 abstain).

## Bối cảnh

- Sau v3, khoảng một nửa số câu public có bất đồng giữa các solver. Trên train, nhóm này được giải bằng vote
  (87%) và judge (44%). Judge chỉ *đọc* hai đáp án rồi chọn, không chạm vào dữ liệu.
- Yêu cầu từ v4 trở đi: cải tiến phải là **năng lực của agent**, không phải luật theo đề public.

## Thay đổi kiến trúc

### 1. Verifier agent (`verify/arbiter.py: verify_agent`)

Bước mới trong arbitration, nằm giữa vote và judge:

```
A == B ───────────────────────────────────────────► agree_ab
khác → +mẫu code/plan → nhóm theo canonical
   bất đồng (≥ 2 nhóm, hoặc 1 ứng viên lẻ không phải rule)
      └─► VERIFIER: phiên CodeAct mới, có các ứng viên A/B/C kèm evidence
            1. chỉ ra chính xác điểm các ứng viên khác nhau (cột, bộ lọc, khoảng thời gian, định nghĩa)
            2. viết code kiểm điểm đó trên dữ liệu
            3. "thiếu dữ liệu" chỉ hợp lệ sau khi đã tìm trong bảng, cột và giá trị
            4. áp định nghĩa trong rulebook theo đúng câu chữ → final_answer
      ứng viên V (trọng số 1.5) vào pool → xếp hạng lại
   nhóm đứng đầu chứa V → source = verify
   không → vote (margin 0.8) → judge (giữ làm phương án cuối)
```

- Verifier dùng lại `solve_code(..., tag="verify", task=VERIFY_TASK, stage_prefix="verify")`. Vì cùng system
  prompt và prefix với code solver, vLLM dùng lại được prefix cache, nên chi phí tăng khoảng 25–35% thời gian LLM.
- Có ngân sách token riêng: `llm.max_tokens.verify_first/verify_step` (2048 / 768), thinking bật cho bước đầu.
- Bật/tắt bằng `arbitration.verify` (tắt trong profile `minimal` và trong `configs/groq.yaml`).
- Verifier lỗi thì quay về vote/judge như cũ (`info["verify_error"]`).

### 2. Trọng số theo khả năng quan sát của solver

- `W_VERIFY = 1.5`: verifier đã kiểm trên dữ liệu.
- `W_PLAN_ABSENT = 0.5`: plan solver chỉ thấy catalog chứ không thấy bảng, nên phiếu "unanswerable" của nó yếu hơn.

### 3. Sandbox giữ dữ liệu shop khi kernel restart (`solvers/code/sandbox.py`)

Trước đây, khi một cell timeout làm kernel khởi động lại thì toolkit `dm` mất dữ liệu shop (`dm` = None). Khoảng
110 lượt code solver chết ở hai shop public vì lỗi này. Nay sandbox nhớ đoạn code rebase và chạy lại sau khi
restart. Nếu restart thất bại thì trả `KernelDied` rõ ràng. Có test mới: `test_timeout_keeps_rebased_business`.

### 4. DMQL cho điều kiện trên giá trị tổng hợp (`llm/prompts.py`)

Thêm luật L vào PLAN_SYSTEM: điều kiện trên giá trị đã gộp theo entity hoặc theo đơn vị thời gian thì dùng `having`
với `op = count` hoặc `filter_list`. DMQL_DOC có thêm hai ví dụ trung tính: số ngày có doanh thu ≥ 5 triệu, và số
khách có ≥ 3 đơn.

### 5. Trace dễ phân tích hơn

`ctx.failed` ghi lại các quỹ đạo code thất bại, và trace có thêm `failed_trajectories` (`pipeline.py`).

## Kết quả

| | v3 | v4 |
|---|---|---|
| sub300 PA | 0.807 | **0.870** |
| đúng / sai / abstain | 256 / 14 / 30 | 269 / 8 / 23 |
| public | 0.425 | **0.433** |

Verifier quyết định 44 câu sub300: 36 đúng, 3 sai, còn lại abstain.

Public tăng ít hơn train. Một phần là do public v4 chạy chung GPU với sub300 nên có câu bị timeout (vd. 6 câu
dev_002). Phần còn lại là do lỗi đồng thuận ở tầng dữ liệu, vốn không đi qua verifier.

## Còn lại (dẫn tới v5)

Đọc trace sub300 cho thấy agent *đã tìm đúng cột* nhưng tool không cho dùng:
- Lọc theo `customer_type` bị toolkit từ chối.
- Nhóm hàng nằm trong bảng danh sách sản phẩm, trong khi master sản phẩm là báo cáo tồn kho.
- Cột có giá trị hằng ("Hàng hóa", "Cá nhân") bị nhận làm category hoặc loại khách.

Các lỗi nhỏ hơn chưa sửa: nợ theo tuổi nợ, giá nhập bình quân, list một phần tử so với string.
