# v2 — Sửa theo lỗi train + calibrator học từ trace (2026-10-30)

**Kết quả:** train sub300 PA 0.750 (242 / 17 / 41) · public 0.17 (747 câu có đáp án + 34 unanswerable, 219 abstain).
File nộp: `submissions/public_v2_2026-10-30.jsonl`.

## Bối cảnh

v1 sai 28/300 câu train. Các nhóm lỗi lớn nhất là trung bình tiền không làm tròn, câu gõ Telex ("soos bill hqua"),
và câu theo ca bị trả `unanswerable` vì không nhận ra cột thời gian.

## Thay đổi

Kiến trúc **không đổi**. Chỉ sửa các thành phần bên trong và bổ sung tài sản học từ dữ liệu.

| Thay đổi | File | Ghi chú |
|---|---|---|
| Làm tròn tiền theo rulebook (`numbers.money_decimals`), half-up | `rules/extract.py`, `semantic/catalog.py`, `solvers/plan/compiler.py` | 9–10 câu AOV trên train |
| Giải mã Telex có dấu *trước khi* đưa câu vào prompt ("soos" → "số", "ddown" → "đơn") | `nlu/typos.retype_telex` | trước đây NLU đã sửa, nhưng LLM vẫn nhận chữ gốc |
| "thời gian tạo / lập" → `order_datetime` | `assets/ontology.yaml` | câu hỏi theo ca (biz_106, biz_109) |
| **Calibrator đã huấn luyện**: logistic, C = 0.3, 583 dòng từ trace sub300 v1 + v2 | `scripts/oof_from_run.py`, `scripts/train_calibrator.py`, `assets/calibrator.json` | luật cũ giữ ở `assets/calibrator.fallback.json` |
| Bộ đo nhanh `train_s300` | `scripts/make_sample.py` | 10 câu/shop, chọn theo hash câu hỏi |
| vLLM `--max-num-batched-tokens 16384` | `scripts/serve_llm.sh` | +9% tok/s; MTP và chunk 32k tệ hơn |

## Kết quả theo nguồn quyết định (sub300 v2)

| Nguồn | Đúng | Sai | Abstain | Độ chính xác khi trả lời |
|---|---|---|---|---|
| agree_ab | 208 | 3 | 0 | 98.6% |
| vote | 26 | 4 | 28 | 87% |
| judge | 8 | 10 | 3 | **44%** |

Calibrator thắng luật fallback rất ít khi đánh giá LOBO (+0.003 đến +0.019).

## Bài học

- Train +0.067 nhưng public chỉ +0.03. Ước lượng trên public: khoảng 476 đúng, 305 sai, tức độ chính xác khi trả
  lời khoảng 61%, so với 93% trên train.
- Phép tính phản chứng: nếu 498 câu `agree_ab` của public vẫn đúng 98.6% thì PA phải ≥ 0.20. Điểm thực tế 0.17
  nghĩa là nhiều câu public bị **cả hai solver cùng sai**. Lỗi vì vậy nằm ở tầng ngữ nghiệp vụ chung, không phải
  ở calibrator hay exemplar. Kết luận này dẫn tới v3.
- Judge (LLM đọc hai đáp án rồi chọn) chỉ đúng 44%, tức gần như tung đồng xu. Điều này dẫn tới verifier của v4.
