# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603479
- **Họ và tên:** Phạm Trung Kiên
- **SEED** (in ở ô đầu notebook): 6155

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg.png) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

Bảng so sánh cuối Phần 5 (SEED = 6155):

| Phương án | F1 (test) | F1 (SUB 40) | p95 ms/tin | USD / 1k tin |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg (tự code) | 0.856 | 0.889 | 0.106 | 0.000 |
| 2. BoW + MLP (deep learning) | 0.923 | 0.947 | 0.374 | 0.000 |
| 3a. Encoder (MiniLM) + kNN | 0.907 | 0.974 | 113.086 | 0.000 |
| 3b. Encoder (MiniLM) + LogReg | 0.931 | 0.947 | 37.035 | 0.000 |
| 3c. Encoder (MiniLM) + MLP head | 0.939 | 0.947 | 36.268 | 0.000 |
| 4a. LLM nhỏ zero-shot | 0.976 | 0.976 | 1738.733 | 0.008 |
| 4b. LLM lớn zero-shot | 0.950 | 0.950 | 1488.139 | 0.084 |
| 4c. LLM nhỏ few-shot | 0.952 | 0.952 | 1169.035 | 0.017 |

**Tính tay (hand_calc):** baseline accuracy ≈ 86.59%; σ(0) = 0.50; F1 ví dụ = 0.73; cos(A,B) ≈ 0.60; chi phí LLM ≈ $10.50/ngày.

## 3. Dự đoán và thực tế

Ba dự đoán `predict(...)` lệch nhiều nhất so với kết quả thực tế:

1. **`llm_models`** — Đoán model lớn F1 cao hơn model nhỏ khoảng 0.02–0.05, và LLM sẽ ngang hoặc hơn 3b (Encoder + LogReg). Thực tế: LLM nhỏ zero-shot F1 = **0.976**, LLM lớn chỉ **0.950** (ngược chiều). Lý do lệch: trên bộ SUB 40 tin, model lớn không nhất thiết tốt hơn; zero-shot còn phụ thuộc prompt và độ ổn định nhãn, nên “lớn hơn = chính xác hơn” không đúng.

2. **`llm_fewshot`** — Đoán F1 tăng và chi phí tăng khoảng 5 lần. Thực tế: F1 **giảm** từ 0.976 (zero-shot) xuống **0.952** (few-shot); USD/1k tin chỉ tăng khoảng **2×** (0.008 → 0.017). Lý do lệch: few-shot làm prompt dài hơn (tốn token) nhưng ví dụ đưa vào chưa giúp rõ hơn zero-shot trên bộ này; thậm chí có thể nhiễu nếu ví dụ không đại diện.

3. **`hard_eval`** — Đoán accuracy BoW ~65–75%, Encoder ~75–85%, LLM ~80–90%. Thực tế trên 50 case khó: BoW + LogReg chỉ **52%**, Encoder khoảng **62–72%**, LLM nhỏ đạt **100%**. Lý do lệch: ước lượng còn lạc quan với BoW/Encoder (các bẫy né từ khóa, ham giống spam làm tụt mạnh), đồng thời đánh giá thấp khả năng zero-shot của LLM trên bộ stress test này.

## 4. Phân tích 50 case khó

**Model tụt nhiều nhất:** họ **BoW** — đặc biệt **BoW + MLP** (F1 SUB 0.947 → F1 50 case khó **0.389**) và **BoW + LogReg** (0.889 → **0.333**). Accuracy trên 50 case chỉ còn ~52–56%, trong khi LLM nhỏ vẫn **100%**.

**Nhóm case làm BoW sai nhiều nhất:**
- **Spam giọng tự nhiên** — BoW+LogReg 10%, BoW+MLP **0%**
- **Né từ khoá** — BoW+LogReg 17%, BoW+MLP **0%**
- **Thao túng / lừa đảo** — BoW+LogReg chỉ 12%

Encoder tụt nhẹ hơn (vẫn ~62–72%) nhưng cũng yếu ở spam giọng tự nhiên; LLM gần như không bị bẫy trên bộ này.

**2 case cụ thể:**

1. **#29 [Né từ khoá]** — *“C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd…”* (spam).  
   **Cơ chế sai:** BoW tách token theo từ đã thấy khi train. Chữ bị thay số (`C0ngr4tulat10ns`, `w0n`, `c0m`) → hầu như không khớp vocab spam quen thuộc → mô hình không “nhìn thấy” tín hiệu và dễ đoán **ham**.

2. **#12 [Spam giọng tự nhiên]** — *“hi there, sorry to bother you, my friend gave me your number… check my profile at meetlocal.club”* (spam).  
   **Cơ chế sai:** Không có từ khoá spam điển hình (`FREE`, `WIN`, `URGENT`…). BoW chỉ đếm từ rời; câu lịch sự + lời chào nghe như tin thường → thiếu feature spam → đoán **ham**. Embedding/LLM đọc được ý định dụ click nên dễ bắt hơn.

## 5. Thử thách cá nhân

**Đề #6** (SEED 6155 % 6 = 5): Sửa `build_classification_prompt` để LLM trả JSON `{"label": ..., "reason": ...}`, viết parser JSON, chạy trên 10 tin SUB; so token/chi phí với bản trả lời 1 từ — có đáng không?

**Cách làm:**
- Đổi prompt yêu cầu chỉ trả JSON 2 trường `label` và `reason`.
- Viết `parse_json_label` để trích JSON từ response và lấy nhãn.
- Chạy song song bản 1 từ và bản JSON trên 10 tin đầu của SUB; so F1, token trung bình, USD.

**Kết quả (10 tin SUB):**

| | Bản 1 từ | Bản JSON |
| --- | ---: | ---: |
| F1-score | 0.857 | 0.857 |
| Tokens (TB) | 584.0 | 778.7 |
| Chi phí USD | 0.000064 | 0.000134 |
| Invalid | 0 | 0 |

JSON **đắt gấp ~2.11 lần**, F1 **không tăng**. Hai bản cùng nhãn trên 10 tin (sai chung tin #6: ham bị đoán spam vì nội dung gợi dục).

**Giải thích / kết luận:** Phần `reason` hữu ích khi cần audit hoặc debug, nhưng với pipeline chỉ cần nhãn spam/ham thì bản 1 từ đủ và rẻ hơn. Không đáng dùng JSON nếu mục tiêu chính là phân loại hàng loạt với chi phí thấp.

## 6. Phương án cho sản phẩm thật

Yêu cầu: **10 triệu SMS/ngày** (~116 tin/s trung bình), **p95 < 50 ms/tin**.

**Đạt SLA p95 < 50 ms (theo bảng của mình):**
- BoW + LogReg — p95 **0.106 ms**, ~31k tin/s, F1 SUB 0.889
- BoW + MLP — p95 **0.374 ms**, ~32k tin/s, F1 SUB **0.947**
- Encoder + LogReg — p95 **37.0 ms**, ~53 tin/s, F1 SUB 0.947
- Encoder + MLP head — p95 **36.3 ms**, ~53 tin/s, F1 SUB 0.947

LLM (p95 > 1 s, ~1 tin/s) **không** đạt SLA nếu gọi từng tin.

**Chọn:** **BoW + MLP** — vừa đủ nhanh (p95 0.37 ms, throughput dư sức so với 116 tin/s), vừa F1 SUB cao (0.947), chi phí API = 0. Encoder đạt p95 nhưng throughput đo được chỉ ~53 tin/s nên cần scale thêm tài nguyên mới ổn định ở 10M tin/ngày.

**Đánh đổi chấp nhận:** BoW yếu trên case khó (né từ khoá, spam giọng tự nhiên — F1 hard tụt còn ~0.39). Đổi lấy độ trễ cực thấp và chi phí 0. Có thể bổ sung router: tin “lưỡng lự” (p ∈ [0.3, 0.7]) mới gọi LLM.

**Nếu dùng LLM thuần:** theo USD/1k tin đo được → LLM nhỏ ~**$2.5k/tháng**, LLM lớn ~**$25k/tháng**, vẫn chậm hơn SLA rất nhiều.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời / viết gì | Bạn đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Khái niệm (Phần 1) | Vì sao dùng F1 thay accuracy khi dữ liệu lệch ham/spam? Learning rate lớn (`lr=50`) sẽ ra sao? | Giải thích F1 cân precision–recall; lr lớn làm loss nhảy mạnh, có thể phân kỳ hoặc hội tụ bất ổn | Đối chiếu với đồ thị loss tự chạy (`lr=0.001 / 0.5 / 50`); thấy `lr=50` loss tăng rồi vẫn tụ về thấp |
| Khái niệm (Phần 2–3) | Bỏ ReLU thì MLP thành gì? Cosine khác tích vô hướng thế nào? | Không ReLU ≈ chồng Linear → gần mô hình tuyến tính; cosine cần chuẩn hoá vector | Tự chạy thí nghiệm không ReLU (F1 ~0.928); kiểm tra `top_k_similar` có chia `‖A‖‖B‖` |
| Phần 4 — `parse_label` | Viết hàm chỉ nhận đúng `"spam"` / `"ham"`, còn lại `None` | Code `strip().lower()` + if/else; cảnh báo không dùng `"spam" in text` | Sửa lại: không mặc định ham khi parse fail; chạy batch và đếm `invalid=0` |
| Phần 4 — gọi API | Viết `call_openai` + `llm_classify_batch` (zero-shot), `temperature=0`, đo USD | Skeleton gọi OpenRouter, đếm token, trả `preds`/`usd` | Đổi model đúng `gpt-6-luna` / `gpt-6.1-terra`; đối chiếu F1 và USD/1k với bảng RESULTS |
| Phần 4 — few-shot | Thêm ví dụ spam/ham vào prompt, dự đoán F1/chi phí đổi thế nào | Gợi ý ghép shots vào prompt; nói F1 thường tăng nhưng đắt hơn | Chạy thật: F1 **giảm** 0.976→0.952, phí ~×2 (không ×5 như đoán) |
| Phần 5 — hard cases | Prompt injection là gì, làm sao phòng thủ ngoài prompt? | Giải thích tách DATA/INSTRUCTION, thẻ `<sms>`, không để LLM là lớp duy nhất | So 4a vs 4d trên 50 case; cả hai 100% trên bộ này; vẫn ghi rủi ro nếu chỉ dựa prompt |
| Thử thách #6 | Sửa prompt trả JSON `{label, reason}`, parse JSON, so với bản 1 từ | Template prompt JSON + `json.loads` / tìm `{...}` trong response | Tự đo trên 10 tin SUB: F1 bằng nhau (0.857), JSON đắt **2.11×** → kết luận không đáng nếu chỉ cần nhãn |

**Điều hiểu ra hôm nay:**
1. Model lớn hơn / few-shot nhiều hơn **không tự động** cho F1 cao hơn — trên SUB của mình LLM nhỏ zero-shot thắng LLM lớn và few-shot.
2. Điểm trên test thường có thể **quá lạc quan**: BoW F1 SUB ~0.95 nhưng trên 50 case khó tụt còn ~0.33–0.39 vì dựa từ khoá.
3. Chọn mô hình sản phẩm phải nhìn **cùng lúc** F1, p95 và chi phí — Pareto/SLA quan trọng hơn “model mới nhất”.
4. Dùng AI viết code được, nhưng phải **review + chạy đo lại**; nếu tin AI mù quáng sẽ sai về chi phí (few-shot) và về “model lớn hơn = tốt hơn”.
