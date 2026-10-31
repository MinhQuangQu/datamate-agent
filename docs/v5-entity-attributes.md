# v5 — Thuộc tính entity, nhóm hàng từ bảng phụ (2026-10-31)

**Kết quả:** train sub300 *chưa đo* · public *chưa nộp*.
File nộp: `submissions/public_v5_2026-10-31.jsonl`. Cả 1000 câu đều chạy bằng code v5, không ghép với kết quả cũ, và
mất 3 giờ 23 phút khi chạy một mình trên 1× 3090. File gồm 905 câu trả lời, 13 unanswerable và 82 abstain (v4 có 209
abstain).

## Bối cảnh

Sau v4, lỗi còn lại trên train phần lớn là các khoảng trống trong **tool**. LLM đã chọn đúng cột và đúng ý, nhưng
compiler/toolkit không biểu diễn được truy vấn đó, hoặc semantic layer không nối được dữ liệu cần thiết. Sửa
tool thì mọi solver (plan, code, verifier) đều hưởng lợi, và lớp lỗi này sẽ gặp lại ở shop mới.

## Thay đổi kiến trúc

### 1. Lọc theo mọi thuộc tính của entity (`solvers/plan/compiler.py`, `solvers/code/toolkit.py`)

- Tách `filter_cond(f, md)` ra khỏi `_where`. Bộ lọc trên thuộc tính entity (`customer.customer_type`,
  `product.<attr>`) được biên dịch thành `IN` / `NOT IN (subquery trên bảng dim)`.
- `_entity_attribute` nhận cả tên thuộc tính trần (`customer_type`): nếu đúng một entity có thuộc tính đó trong
  `dim_table` thì tự hiểu là của entity đó.
- `dm.lines()` trong toolkit dùng chung `filter_cond`, nên code solver và verifier lọc giống hệt plan solver.
- Có test mới: `test_bare_attribute_name_resolves_to_its_entity`.

### 2. Business Card liệt kê giá trị loại khách (`semantic/card.py`)

Khi có `sem.customers`, Business Card thêm dòng
`## customer attribute (filter field customer.customer_type; …): <các giá trị>`. Nhờ vậy LLM biết trường lọc và các
giá trị hợp lệ mà không phải đoán.

### 3. Nhóm hàng lấy từ bảng phụ (`semantic/views.py: _side_product_role`)

Khi master sản phẩm không có category (ví dụ là báo cáo tồn kho), nhưng một bảng phụ không thuộc pack lại có
category (ví dụ danh sách sản phẩm), view sẽ lấy category bằng scalar subquery. Khoá nối là `product_id`
(upper/trim) hoặc tên đã fold dấu. Cách này áp cho cả `sem.sales_lines.category` (khi không có cột category
trên dòng) và `sem.products.category`.

### 4. Cột hằng không phải cột phân nhóm (`semantic/heuristics.py: _value_hint`)

Cột có đúng 1 giá trị phân biệt bị phạt (−2.0) khi xét vai trò `category` hoặc `customer_type`. Ví dụ: "Hàng hóa"
hay "Cá nhân" xuất hiện ở mọi dòng.

### 5. Từ đồng nghĩa cho loại khách (`assets/ontology.yaml`)

Thêm: "nhóm khách", "kiểu khách", "phân loại khách", "đối tượng khách".

## Kiểm

- 90 test pass.
- Build lại 30 shop train không LLM và diff với bản build v3. Doanh thu và số đơn theo tháng **không đổi ở shop
  nào**; chỉ cột category / customer_type ở 4 shop thay đổi, đúng như dự kiến.
- Kiểm từng câu trên train sau khi sửa:
  - biz_128: % theo nhóm khách = 17.19 (khớp đáp án); doanh thu nhóm khai vị = 52.465.000 (khớp).
  - biz_120: nhóm dị ứng = 3.36 (khớp).
  - biz_106: trà sữa = 56.845.000, bánh = 3.14 (khớp).
- Public: dev_003 nay có nhóm hàng và loại khách thật. dev_002, dev_004, dev_009 vẫn không có nguồn category
  trong dữ liệu.

## So sánh nguồn quyết định trên public (v4 → v5)

| Nguồn | v4 trả lời / abstain | v5 trả lời / abstain |
|---|---|---|
| agree_ab | 524 / 0 | 559 / 2 |
| verify | 175 / 54 | 253 / 39 |
| vote | 52 / 17 | 78 / 17 |
| judge | 9 / 22 | 20 / 11 |
| single | 30 / 59 | 8 / 3 |
| none (không ứng viên nào) | 0 / 50 | 0 / 10 |
| timeout (không có quyết định) | 6 | 0 |

Phần lớn abstain giảm đi (none + single: từ 109 còn 13) vì public v4 chạy chung GPU với sub300. Khi đó solver bị
timeout hoặc lỗi nên không có hai ứng viên để so. v5 chạy một mình nên hầu hết câu đi đủ đường agree_ab / verify.
agree_ab tăng 35 câu, phù hợp với việc category và loại khách giờ đã có dữ liệu. So với v4: 741 câu giống hệt,
156 câu v4 abstain nay có trả lời, 29 câu ngược lại, và 74 câu cả hai bản đều trả lời nhưng ra kết quả khác nhau.

## Còn lại / gợi ý cho v6

- Đo v5 trên sub300 để tách tác dụng của từng thay đổi.
- Nợ theo tuổi nợ (aging buckets), giá nhập bình quân, đáp án list một phần tử so với string.
- Các luật xếp hạng mới chỉ có trong văn bản rulebook ("sộp", "khách trung thành", "giờ cao điểm"). Hướng tổng quát
  là cho RuleSpec biểu diễn được "X nhất = xếp theo tiêu chí Y với điều kiện Z", thay vì cài từng luật.
- Calibrator vẫn được huấn luyện trên các nguồn quyết định của v2. Nên huấn luyện lại với đặc trưng `source = verify`.
