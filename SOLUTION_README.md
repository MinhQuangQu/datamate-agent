# DataMate Agent — Solution README (Zalo AI Challenge 2026 · DataMate)

> Điền mọi chỗ `<…>` trước khi build image cuối. File này nằm ở `/app/README.md` trong image và gửi kèm form nộp.

## 1. Đội

- Tên đội: `<team>` · Thành viên: `<…>` · Liên hệ: `<email>`

## 2. Tóm tắt giải pháp

Agent offline trả lời câu hỏi tiếng Việt trên dữ liệu bảng (CSV/XLSX/SQLite):
1. Nạp mọi file vào DuckDB, suy kiểu theo cột (số/ngày kiểu Việt Nam), profile dữ liệu.
2. Dựng tự động semantic model cho từng doanh nghiệp (vai trò cột, quan hệ, khái niệm vắng mặt) và biên dịch rulebook của split thành cấu hình có cấu trúc → các view/metric đã áp luật nghiệp vụ.
3. Hai bộ giải độc lập: (A) LLM sinh kế hoạch truy vấn có kiểu (DMQL) → compiler SQL tất định; (B) LLM viết và chạy Python trong sandbox với bộ công cụ đã mã hóa luật.
4. Chuẩn hóa ứng viên theo luật chấm, bỏ phiếu/trọng tài khi bất đồng, mô hình hiệu chỉnh ước lượng P(đúng); chỉ trả lời khi P vượt ngưỡng (tối ưu Penalized Accuracy), ngược lại `abstain`.

## 3. Model sử dụng khi inference (Rule 2)

| Model | Repo / nguồn | Revision (commit) | Ngày công bố trọng số | License | Số tham số (đo từ checkpoint ship) | Định dạng | Ghi chú |
|---|---|---|---|---|---|---|---|
| `<Qwen3.5-9B>` | `<https://huggingface.co/Qwen/Qwen3.5-9B>` | `<sha>` | `<2026-03-02>` | `<Apache-2.0>` | `<9,6xx,xxx,xxx>` (gồm vision encoder + MTP head) | `<BF16>` | `<đã merge LoRA? có/không>` |
| Calibrator logistic | tự huấn luyện | – | – | – | `<~20>` | JSON | không phải mạng nơ-ron |
| **Tổng** | | | | | **`<…>` ≤ 15B** | | |

Không dùng model embedding, reranker hay classifier nơ-ron nào khác. Số tham số do `python -m dma.utils.count_params /opt/models/main` tính từ header safetensors (file `/app/params.json`).

## 4. Yêu cầu chạy

| Mục | Giá trị |
|---|---|
| GPU | 2× NVIDIA RTX 5090 32 GB (chạy được với 1 GPU, chậm hơn) |
| CUDA runtime trong image | `<12.8 / 12.9 / 13.0>` — yêu cầu NVIDIA driver `<≥ 570 / ≥ 580>` |
| RAM | ≥ 64 GB khuyến nghị (dùng thực tế ~`<…>` GB) |
| Shared memory | `<khuyến nghị --ipc=host hoặc --shm-size=16g / không yêu cầu (đã kiểm thử T3)>` |
| Mạng | không cần (chạy với `--network none`) |
| Thời gian dự kiến | `<~45>` phút cho ~500 câu / 15 doanh nghiệp, gồm nạp model (`<…>` phút) |

## 5. Cách chạy

```bash
docker load -i datamate_dma_final.tar.gz          # sha256: <…>
docker run --rm --gpus all --network none --ipc=host \
  -v /path/to/test_split:/data:ro \
  -v /path/to/result:/result \
  dma:final
```

- Đầu vào: `/data` (chỉ đọc) chứa `questions.jsonl`, `rulebook.md`, `calendar.csv`, `data/<business_id>/…` (tự dò tối đa 3 cấp).
- Đầu ra: `/result/submission.jsonl` (ghi từng dòng ngay khi xong mỗi câu).
- Log: `/result/logs/<id>.json` (câu hỏi, đáp án cuối, thời gian, kế hoạch DMQL + SQL, code đã chạy + output, các ứng viên, xác suất, quyết định), `/result/logs/run_manifest.json`, `/result/logs/predict.log`, `/result/logs/bkp/<business>/` (semantic model đã dựng).

## 6. Tái lập (Rule 8)

Seed sinh từ nội dung câu hỏi; tham số sampling cố định; ngân sách theo token/bước (không theo đồng hồ); mọi xếp hạng có tie-break; số tiền tính bằng DECIMAL. Đo nội bộ qua 3 lần chạy cùng image: tỷ lệ đổi đáp án `<…>%`, ΔPA `<…>`.

## 7. Huấn luyện & dữ liệu (Rule 6)

| Thành phần | Vị trí trong image | Dữ liệu dùng |
|---|---|---|
| Calibrator | `/app/training/calibrator/` | train (leave-business-out CV) + dữ liệu tự tổng hợp |
| Exemplar memory | `/app/assets/exemplars.jsonl` | chỉ train + dữ liệu tự tổng hợp |
| RuleSpec tham chiếu | `/app/assets/train_rulespec.json` | rulebook train |
| Fine-tune (nếu có) | `/app/training/finetune/` | `<RFT trên train + synth; teacher open-weight: …>` |
| Generator dữ liệu tổng hợp | `/app/synth/` | không dùng dữ liệu test |
| Dữ liệu train | `/app/data_train/` | bộ train BTC cung cấp |

Không sử dụng dữ liệu public/private test cho huấn luyện, ghi nhớ hay tạo luật.

## 8. Tuyên bố tuân thủ

- Suy luận hoàn toàn offline; không gọi API/dịch vụ ngoài; không tải thêm trọng số/dữ liệu.
- Không có logic theo ID câu hỏi/doanh nghiệp; không hard-code đáp án (kiểm bằng `scripts/check_no_id_logic.py` và kiểm thử đổi tên ID ngẫu nhiên).
- Thư mục `/data` chỉ được đọc; file tạm ghi vào `/tmp/dma`.

## 9. Cấu trúc thư mục image

```text
/app/predict.sh  /app/README.md  /app/params.json
/app/src/dma/            mã suy luận
/app/configs/final.yaml  cấu hình chạy
/app/assets/             exemplars, calibrator, rulespec tham chiếu, ontology
/app/training/           mã huấn luyện
/app/synth/              generator dữ liệu tổng hợp
/app/data_train/         dữ liệu huấn luyện
/opt/models/main/        trọng số model (+ SHA256SUMS)
```
