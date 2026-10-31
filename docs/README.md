# Lịch sử kiến trúc DataMate Agent

Mỗi file mô tả một phiên bản đã chạy trên public: trước đó có vấn đề gì, phiên bản đó đổi gì trong kiến trúc,
file nào bị sửa, kết quả đo ra sao và còn lại lỗi gì. Mọi số liệu đo trên 1× RTX 3090 với Qwen3.5-9B FP8 (vLLM).
Bộ đo train là `train_s300`: 10 câu × 30 shop train, có đáp án, chọn theo hash câu hỏi.

| Bản | Ngày | Train sub300 PA (đúng/sai/abstain) | Public | Ý tưởng chính |
|---|---|---|---|---|
| [v1](v1-baseline.md) | 2026-10-30 | 0.683 (233/28/39) | 0.14 | Kiến trúc gốc: semantic layer → 2 solver → vote/judge → calibrator |
| [v2](v2-train-fixes.md) | 2026-10-30 | 0.750 (242/17/41) | 0.17 | Sửa các lỗi thấy trên train, calibrator học từ trace |
| [v3](v3-deterministic-layer.md) | 2026-10-31 | 0.807 (256/14/30) | **0.425** | Sửa tầng dữ liệu tất định để đọc được kiểu file mới; kiểm bằng pandas độc lập |
| [v4](v4-verifier-agent.md) | 2026-10-31 | 0.870 (269/8/23) | **0.433** | Verifier agent: khi bất đồng thì chạy code kiểm đúng điểm lệch |
| [v5](v5-entity-attributes.md) | 2026-10-31 | chưa đo | chưa nộp | Lọc theo thuộc tính entity, nhóm hàng từ bảng phụ, bỏ cột hằng |

## Mạch xuyên suốt

1. **v1→v2** sửa theo lỗi của train. Train tăng lên, nhưng public gần như đứng yên (0.14→0.17), vì bộ đo train là
   in-sample: ontology và heuristic được chỉnh trên chính 30 shop đó.
2. **v3** nhận ra lỗi chính của public là *đồng thuận sai*. Hai solver dùng chung một semantic layer; layer đó đọc
   sai file kiểu mới, nên cả hai cùng ra một số sai và được gắn `agree_ab`. Cách kiểm không cần đáp án là so view
   của agent với phép tính pandas viết riêng. Điểm public tăng gấp 2.5 lần.
3. **v4** chuyển trọng tâm từ "sửa dữ liệu" sang "năng lực agent". Khi các solver bất đồng, một agent có tool sẽ
   tự kiểm điểm tranh chấp trên dữ liệu thay vì chỉ đếm phiếu hay để LLM chọn bằng cách đọc.
4. **v5** sửa các khoảng trống trong *tool* mà agent đã dùng đúng nhưng bị từ chối, ví dụ lọc theo loại khách,
   hoặc nhóm hàng nằm ở bảng khác.

Nguyên tắc từ v3 trở đi: không có logic theo ID shop hay câu, không dùng câu public làm dữ liệu. Trace public chỉ
dùng để tìm *lớp lỗi*; mọi sửa đổi được đo trên train sub300 và kiểm hồi quy bằng build không LLM của 30 shop train.

## Thêm phiên bản mới

Tạo `docs/vN-<ten-ngan>.md` với các mục giống những file trên: Bối cảnh, Thay đổi kiến trúc, File thay đổi,
Kết quả, Còn lại. Sau đó cập nhật bảng ở đây và trong `README.md` gốc.
