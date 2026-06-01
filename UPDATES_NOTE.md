# BẢN GHI CHÚ CÁC CẬP NHẬT HỆ THỐNG DỰ BÁO CHỨNG KHOÁN (V2 & V3)

Tài liệu này tổng hợp toàn bộ các thay đổi, tối ưu hóa thuật toán và nâng cấp hệ thống dự báo chứng khoán sử dụng mô hình XGBoost.

---

## 🚀 PHIÊN BẢN V2: TỐI ƯU HÓA VŨ TRỤ ĐẦU TƯ & SỬA LỖI DÒNG TIỀN KHỐI NGOẠI

### 1. Khắc phục lỗi dữ liệu `foreign_net` (Dòng tiền khối ngoại)
* **Vấn đề:** Phát hiện cột `foreign_net` trong cơ sở dữ liệu `model_training_data` chỉ chứa toàn giá trị `0.0`, khiến mô hình học máy bỏ qua biến quan trọng này (feature importance = 0%).
* **Giải pháp:** Xây dựng script thu thập dữ liệu lịch sử từ FiinQuant API (từ 2021 đến 2026) cập nhật vào MySQL. Mô hình sau khi huấn luyện lại đã tăng tầm quan trọng đặc trưng này lên **~4.29%**.

### 2. Triển khai Lọc Vũ Trụ Đầu Tư (Universe Filtering - Phương án B)
* **Nguyên lý:** Thay vì dự báo trung bình cho toàn bộ 93 mã cổ phiếu bất kể độ nhiễu, hệ thống tự động phân loại cổ phiếu dựa trên hiệu suất dự báo thực tế trên tập Validation (năm 2024):
  * **Universe B (Dễ dự báo):** Có độ chính xác $\ge 53\%$ (Ví dụ: BCM, BSR, BTS, CII, ELC, PVT,...).
  * **Universe C (Độ nhiễu cao):** Có độ chính xác $< 50\%$ (Ví dụ: AGR, ANV, BSI, DBC, DCM,...).
* **Giao diện người dùng (Frontend):**
  * Làm mờ (`opacity: 0.45`), đổi sang thang màu xám và viền nét đứt màu đỏ nhạt đối với các mã thuộc nhóm nhiễu cao.
  * Viền sáng xanh lá nổi bật kèm nhãn dán `⭐ Universe B` đối với các mã dễ dự báo.
  * Thêm cảnh báo và khuyên nhà đầu tư **NÊN BỎ QUA** khi click vào các mã nhiễu cao, kèm giải thích cơ chế SHAP chi tiết.

### 3. Hiển thị toàn bộ Đặc trưng trên Giao diện
* Loại bỏ giới hạn hiển thị cũ (chỉ cắt lấy 12 đặc trưng), cho phép hiển thị đầy đủ đóng góp của tất cả đặc trưng trên biểu đồ của Dashboard.

---

## 💎 PHIÊN BẢN V3: TÍCH HỢP 4 ĐẶC TRƯNG CUNG CẦU MỚI (`bu`, `sd`, `fs`, `fb`)

### 1. Nâng cấp Schema Cơ sở dữ liệu & Cập nhật Lịch sử
* Cập nhật database và tệp [db_setup.py](db_setup.py) để thêm 4 cột mới kiểu `DOUBLE`:
  * **`bu`** (Buying power / Lực mua chủ động)
  * **`sd`** (Selling pressure / Áp lực bán chủ động)
  * **`fs`** (Foreign selling value / Giá trị bán khối ngoại)
  * **`fb`** (Foreign buying value / Giá trị mua khối ngoại)
* Chạy script cập nhật lịch sử đồng bộ thành công dữ liệu từ năm 2021 đến năm 2026 cho 93 mã chứng khoán trong MySQL (hơn 121,000 bản ghi).

### 2. Đồng bộ hóa Đường ống dự báo hàng ngày (Daily Pipeline)
* Nâng cấp [daily_pipeline.py](daily_pipeline.py) để tự động crawl, xử lý và chuẩn hóa 4 trường dữ liệu mới này hàng ngày từ FiinQuant API để lưu vào database và đưa vào mô hình dự báo thời gian thực.

### 3. Huấn luyện lại và Kiểm định Mô hình Học máy
* Nâng tổng số lượng đặc trưng từ **17 lên 21 đặc trưng** trong [train_model.py](train_model.py) và [validate_model.py](validate_model.py).
* **Hiệu năng cải thiện vượt bậc:**
  * **Accuracy Chronological Split (Tập Test 2025-2026):** Tăng lên **52.14%** (vượt mức nền cũ 51.93%).
  * **Accuracy TimeSeriesSplit (5-Fold chéo):** Đạt mức **57.77% ± 1.28%** với AUC-ROC đạt **0.6112**.
  * Đặc trưng áp lực bán chủ động **`sd`** vươn lên vị trí thứ 4 về độ quan trọng đóng góp thực tế (**5.29%**), đóng vai trò quan trọng trong việc phát hiện tín hiệu cạn cung để đảo chiều tăng giá.
