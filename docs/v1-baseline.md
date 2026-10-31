# v1 — Kiến trúc gốc (2026-10-30)

**Kết quả:** train sub300 PA 0.683 (233 đúng / 28 sai / 39 abstain) · public 0.14
(751 câu trả lời, trong đó 41 unanswerable; 249 abstain).
File nộp: `submissions/public_2026-10-30.jsonl`.

## Ý tưởng

*Semantic-layer-first, dual-solver, calibrated abstention.*

- LLM 9B không đủ tin cậy để tự đọc file Excel lộn xộn và tự nhớ rulebook. Vì vậy phần lớn công việc được đẩy
  vào một tầng **tất định**: đọc file, gán nghĩa cho cột, dựng view đã áp luật nghiệp vụ, và compiler truy vấn.
- LLM chỉ làm hai việc: diễn đạt câu hỏi thành truy vấn, và xử lý phần linh hoạt bằng code.
- Hai cách giải khác nhau thì cho hai ứng viên. Nếu hai bên khớp thì tin; nếu không khớp thì lấy thêm mẫu, bỏ phiếu,
  rồi hỏi trọng tài. Cuối cùng một calibrator quyết định trả lời hay `abstain`, vì PA phạt câu sai.

## Luồng xử lý

```
split
 ├─ RuleSpec (1 lần/split)   rules/: parse rulebook → diff với assets/train_rulebook.md
 │                           → chỉ trích lại mục thay đổi (heuristic + 5 mẫu LLM, vote) → RuleSpec
 ├─ BKP (1 lần/shop)         ingest/ → profile/ → semantic/
 │     heuristics.py         gán vai trò cột theo synonym trong assets/ontology.yaml → BSM đề xuất
 │     builder.py            3 mẫu LLM (thinking) + đề xuất heuristic (trọng số 1.5) → vote từng cột
 │     views.py              sem.sales_lines(_valid), sem.orders(_all), sem.products, sem.customers,
 │                           sem.purchase_lots, sem.inventory, sem.receivables … áp luật RuleSpec
 │     catalog.py            metric (revenue, quantity, aov, debt_outstanding…) + gaps (khái niệm vắng)
 │     card.py               Business Card: mô tả view/metric/giá trị mẫu cho prompt
 └─ mỗi câu (pipeline.py)
       nlu/                  chuẩn hoá, Telex, thời gian, link thực thể, gap → ứng viên "rule" (unanswerable)
       Solver A  plan        LLM sinh DMQL (JSON schema) → compiler.py → SQL
                             op: aggregate, count, rank, compare, share, lookup, exists, filter_list, argmax_time
       Solver B  code        CodeAct ≤ 6 cell trong sandbox, toolkit dm.metric/rank/compare/share/debt/stock/sql
                             (toolkit gọi CÙNG compiler)
       arbitration           A == B → agree_ab
                             khác → +2 mẫu code, +1 mẫu plan → vote (margin 0.8) → judge LLM so cặp
       calibrator            logistic 22 đặc trưng → P(đúng) > τ (0.5; unanswerable 0.6) thì trả lời
       formatter             theo type: number/percent/date/list/table/boolean/string
```

## Cấu hình chạy

- Qwen3.5-9B, vLLM 0.32, FP8 weight-only, prefix cache. Toàn bộ rulebook nằm trong prompt (~42k token/câu).
- Thinking bật cho rule, bsm, judge và bước code đầu; tắt cho các bước code sau.
- Khoảng 3 câu/phút trên 1× 3090; public mất khoảng 4 giờ.

## Các sửa cần có để v1 chạy được

| Sửa | File | Tác động |
|---|---|---|
| Vote BSM coi mẫu "bỏ trống" là phiếu "không có vai trò", làm xoá hết vai trò cột và ra 0 view | `semantic/builder.py` | ~40 câu train: 0.14 → 0.71 |
| Gateway coi read-timeout là replica chết, làm cả run bị abstain | `llm/gateway.py` | hết abstain hàng loạt |
| Làm tròn tiền half-up tới đồng (kể cả trung bình tiền) | `rules`, `catalog.MONEY_METRICS`, `compiler.round_money` | áp lên file nộp v1 (43 đáp án) |

## Vì sao public chỉ 0.14

- Hai solver **không độc lập**: dùng chung BSM, view, metric catalog, compiler và RuleSpec. Khi tầng chung hiểu sai
  một khái niệm, cả hai cùng sai giống nhau, và ứng viên đó được gắn `agree_ab`, nhóm calibrator tin nhất
  (98.6% đúng trên train).
- Public có 3 ngành mới (sửa xe, mẹ và bé, điện thoại) và khoảng 35 dòng luật mới. Patch RuleSpec ra rỗng, nên các
  luật mới chỉ đến LLM dưới dạng văn bản.
- Bộ đo train là in-sample, nên không báo trước được điểm public.
