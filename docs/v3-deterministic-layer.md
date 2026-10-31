# v3 — Sửa tầng dữ liệu tất định cho kiểu file mới (2026-10-31)

**Kết quả:** train sub300 PA 0.807 (256 / 14 / 30) · public **0.425** (gấp 2.5 lần v2).
File nộp: `submissions/public_v3_2026-10-31.jsonl`.

## Bối cảnh

Trên public có nhiều câu mà hai solver cùng trả một số sai (xem v2). Nguyên nhân: file public trình bày theo kiểu
chưa gặp ở train, như dòng tổng xen giữa dữ liệu, cột tiền của cả đơn đặt trên từng dòng, trạng thái và công nợ
viết trong ghi chú, hay tên viết tắt. Tầng ingest/semantic đọc sai các file này, và vì hai solver dùng chung tầng
đó nên chúng "đồng thuận" trên một con số sai.

## Phương pháp mới: kiểm không cần đáp án

Với mỗi shop public:
1. Build BKP **không dùng LLM** (`dma.run --no-llm`, `scripts/build_one.py`).
2. Viết một phép tính pandas **độc lập** trên file gốc, đọc theo README của shop.
3. So doanh thu và số đơn theo tháng giữa view `sem.*` của agent và phép tính pandas. Chỗ nào lệch là lỗi của
   tầng tất định.

Mọi sửa đều là quy tắc tổng quát (không theo ID shop), và đều được kiểm hồi quy bằng cách build lại 30 shop train.

## Thay đổi kiến trúc

Luồng không đổi, nhưng tầng tất định có thêm năng lực mới:

| Năng lực | File | Lỗi được sửa (ví dụ) |
|---|---|---|
| Phát hiện **dòng tổng** trong bảng ("TỔNG THÁNG", "tổng 01/05" nằm ngay trong cột mã đơn) và loại khỏi dữ liệu | `ingest/tables.py`, `ingest/parsers.py` | dòng tổng tháng bị ghép vào đơn cuối làm doanh thu gấp đôi |
| Chọn đúng **cột tiền dòng** và không lấy cột tổng của đơn làm tiền dòng ("Tiền hàng trước giảm") | `semantic/heuristics.py`, `semantic/views.py` | doanh thu gấp đôi |
| Ưu tiên "Giá trị hàng" hơn tổng có phí giao / giảm đơn khi tính doanh thu | `semantic/heuristics.py`, `catalog.py` | sai cột doanh thu ở file chỉ có cấp hoá đơn |
| Nhận thêm cột mã đơn ("Đơn số", "STT: số thứ tự đơn"); không gom dòng theo IMEI; không nhận "Cá nhân" làm tên khách | `assets/ontology.yaml`, `semantic/heuristics.py`, `semantic/bsm.py` | ghi chú bị chép lan sang dòng khác |
| **Trạng thái đọc từ cột Ghi chú** khi không có cột trạng thái (theo §8 của rulebook) | `semantic/views.py` | đơn huỷ hoặc trả được tính vào doanh thu |
| **Công nợ viết bằng chữ** ("còn thiếu 304 nghìn", "đưa trước 236k") | `semantic/views.py`, `textnorm.py` | nợ bằng 0 |
| Dùng sheet "Ghi tắt", nối biệt danh nhân viên với danh sách thợ | `semantic/readme.py`, `semantic/views.py` | "thợ chăm chỉ nhất" trả về tên tắt |
| **Bộ phân giải thời gian** hiểu thêm "th4/2026", "8/2025", "5/25", "th8, th9, th10/2025" | `timeparse/resolver.py`, `nlu/analyze.py` | trước đây 37% câu public không có mốc thời gian tất định |
| Ghép kết quả chạy lại một phần (dev_009, dev_010) vào file nộp | `scripts/merge_submissions.py` | |

## Kết quả kiểm

- 11/14 shop public có doanh thu và số đơn theo tháng khớp phép tính độc lập. Các shop còn lại khớp ở những tháng
  đã kiểm, hoặc mới chỉ so với bản build không LLM.
- Build lại 30 shop train: chỉ biz_105 thay đổi, và thay đổi đó sửa đúng lỗi "lệch 40–80k" (T2/2025 = 274.341.000,
  khớp đáp án).

## Còn lại

- Plan solver chưa diễn đạt được điều kiện trên giá trị tổng hợp, kiểu "số ngày có doanh thu ≥ X".
- Các luật xếp hạng mới ("sộp", "khách trung thành", "giờ cao điểm") vẫn chỉ nằm trong văn bản.
- Khoảng 52% câu public có bất đồng giữa các solver, và cách giải quyết bất đồng (vote, rồi judge 44%) quá yếu.
