# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603532
- **Họ và tên:** Lưu Thanh Mỹ
- **SEED: ** 2470

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ **(file hiện đang trống 0 byte — cần chụp ảnh tính tay thật và thay vào trước khi nộp)** |

## 2. Kết quả chính
                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            0.989   0.844      0.911        0.919   
2. BoW + MLP (deep learning)         0.980   0.917      0.948        0.947   
3a. Encoder (MiniLM) + kNN           0.874   0.954      0.912        0.923   
3b. Encoder (MiniLM) + LogReg        0.972   0.945      0.958        1.000   
3c. Encoder (MiniLM) + MLP head      0.972   0.945      0.958        0.950   
4a. LLM nhỏ zero-shot                0.000   0.000      0.000        0.000   
4b. LLM lớn zero-shot                0.000   0.000      0.000        0.000   
4c. LLM nhỏ few-shot                 0.000   0.000      0.000        0.000   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            2.815      43425.132       0.053   
2. BoW + MLP (deep learning)         9.373      21854.338       0.301   
3a. Encoder (MiniLM) + kNN          71.178         17.946      56.870   
3b. Encoder (MiniLM) + LogReg       72.711         51.778      22.843   
3c. Encoder (MiniLM) + MLP head     72.061         52.427      23.430   
4a. LLM nhỏ zero-shot                0.000          0.852     912.733   
4b. LLM lớn zero-shot                0.000          0.892    1062.909   
4c. LLM nhỏ few-shot                 0.000          1.107     877.000   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.080         0.000  
2. BoW + MLP (deep learning)          0.498         0.000  
3a. Encoder (MiniLM) + kNN          117.315         0.000  
3b. Encoder (MiniLM) + LogReg        35.152         0.000  
3c. Encoder (MiniLM) + MLP head      33.895         0.000  
4a. LLM nhỏ zero-shot              1963.896         0.005  
4b. LLM lớn zero-shot              1493.256         0.052  
4c. LLM nhỏ few-shot               1120.964         0.015  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

Bảng trên chính là bảng so sánh cuối Phần 5 lấy từ `lab1_report.json` (đã chạy với `SEED=2470`).

## 3. Dự đoán và thực tế

| Dự đoán (`predict(...)`) | Tôi đoán gì | Thực tế ra gì | Vì sao lệch |
| --- | --- | --- | --- |
| `ml_lr` (learning rate) | lr=50 quá lớn sẽ khiến mô hình dao động/phân kỳ, không tối ưu | Logistic Regression vẫn hội tụ ổn với lr=50 trên bộ dữ liệu và cách khởi tạo của notebook | Tôi đánh giá quá cao độ nhạy của gradient descent với lr lớn; với input BoW thưa (0/1) và vài trăm epoch, bước cập nhật vẫn chưa đủ "nhảy" ra khỏi vùng hội tụ |
| `dl_curve` (MLP vs LogReg) | MLP sẽ vượt trội rõ rệt so với LogReg nhờ hidden layer học được tổ hợp từ | F1 MLP = 0.948 so với LogReg = 0.911 — có cải thiện nhưng không "vượt trội rõ rệt" như kỳ vọng | Input vẫn là BoW 2.000 chiều, thưa và tuyến tính tách được khá tốt ngay từ đầu, nên phần "lợi thế phi tuyến" của MLP không có nhiều đất để phát huy trên bộ dữ liệu nhỏ này |
| `llm_models` (model lớn vs nhỏ) | Model lớn (`gpt-6.1-terra`) sẽ có F1 cao hơn model nhỏ (`gpt-6-luna`) khoảng 2–5 điểm | Cả hai đều có F1 = 0.000 trên SUB 40 (accuracy chỉ 0.5, tức toàn bộ output coi như invalid/parse sai) | Tôi giả định cả hai sẽ trả lời đúng định dạng rồi mới so chất lượng, nhưng thực tế pipeline `parse_label` không đọc được output của cả hai model (có thể do thiếu `reasoning_effort="none"` hoặc giới hạn token chưa đủ cho model reasoning) — đây là lỗi pipeline, không phải lỗi năng lực model |

## 4. Phân tích 50 case khó

## 5. Thử thách cá nhân

## 6. Phương án cho sản phẩm thật

Yêu cầu: xử lý **10 triệu SMS/ngày** (~116 tin/giây trung bình), **p95 < 50 ms/tin**.

- **Đạt yêu cầu độ trễ:** BoW + LogReg, BoW + MLP, và cả 3 biến thể Encoder (MiniLM) + kNN/LogReg/MLP head đều có p95 dưới 50ms (từ 0,08ms đến ~35ms) — chỉ riêng **Encoder + kNN có p95 = 117ms, không đạt**.
- **Phương án chọn: Encoder (MiniLM) + LogReg** (F1 test = 0,958, p95 = 35,15ms, USD = 0) — cân bằng tốt nhất giữa chất lượng (F1 cao nhất trong các phương án tự train) và độ trễ đạt SLA, mà không tốn chi phí API.
- **LLM không dùng trực tiếp trên toàn bộ luồng:** p95 của cả 3 biến thể LLM đều trên 1.000ms (vượt xa 50ms) và tốn tiền theo token (0,005–0,05 USD/1k tin) — ước tính ở quy mô 10 triệu tin/ngày, kể cả với model rẻ nhất cũng tốn hàng chục nghìn USD/ngày nếu gọi toàn bộ.
- **Đề xuất hệ thống lai (router):** Tầng 1 dùng Encoder + LogReg phân loại toàn bộ tin theo thời gian thực; tin có xác suất rõ ràng (ví dụ >0,8 spam hoặc <0,2 spam) được quyết định ngay; tin nằm giữa ngưỡng (vùng không chắc chắn) được đưa sang hàng đợi bất đồng bộ để LLM xử lý sâu hơn hoặc dùng để làm giàu dữ liệu train sau này. Cách này giữ được SLA p95 cho toàn bộ luồng chính, chỉ dùng LLM (đắt và chậm) cho một tỷ lệ nhỏ case khó.
- **Đánh đổi chấp nhận:** bỏ qua phần lợi thế "hiểu ngôn ngữ tự nhiên sâu" của LLM trên luồng chính để đổi lấy tốc độ và chi phí; rủi ro là các case kiểu "spam giọng tự nhiên" (nhóm tụt F1 nặng nhất ở mục 4) vẫn có thể lọt qua tầng 1 nếu không được đẩy đúng sang tầng LLM.

## 7. AI log và điều đã học


Trong bài làm, sử dụng tôi dùng tool trong các việc:
1> Sử dụng để làm tường minh câu hỏi
2> Sử dụng AI để thực hiện giải thích và định hướng các phần code 
3> Tham khảo logic và luyện cách đọc hiểu vì bản thân không mạnh phần code
4> Đang luyện cơ chế và các key => luyện dần sự hiểu logic code 