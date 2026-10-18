# LAB2: PREDICTING PRODUCT SALES

##  1. Giới thiệu bài toán
Bài toán áp dụng các thuật toán **Hồi quy (Regression)** trong Machine Learning nhằm dự đoán chính xác số lượng sản phẩm bán ra (`Units_Sold`) dựa trên dữ liệu thương mại điện tử. Mục tiêu giúp doanh nghiệp tối ưu hóa kế hoạch quản lý kho hàng, định giá và hoạch định chiến lược marketing hiệu quả.

##  2. Cấu trúc thư mục dự án
* `sales_data.csv`: Bộ dữ liệu lịch sử bán hàng và các thông tin chiến dịch, khách hàng.
* `main.py`: Script chính thực hiện toàn bộ quy trình:
  - Làm sạch dữ liệu và xử lý ngoại lai (Outliers) trên tập train.
  - Trích xuất đặc trưng (Feature Engineering): bóc tách mùa vụ từ ngày tháng, tính giá thực tế sau giảm giá (`net_price`), log giá (`log_price`).
  - Xây dựng Pipeline huấn luyện và so sánh các mô hình: *Baseline (Mean)*, *Linear Regression*, *Decision Tree*, *Random Forest*, và *Gradient Boosting*.
  - Tối ưu hóa siêu tham số bằng `GridSearchCV` kết hợp `TimeSeriesSplit`.
  - Đánh giá mô hình qua các chỉ số: **RMSE**, **MSE**, **MAE**, **R2 Score**.
  - Lưu mô hình tốt nhất vào file `.joblib`.
* `predict_sales.py`: Script tái sử dụng mô hình đã huấn luyện để dự đoán số lượng bán cho các kịch bản mới (hỗ trợ nhập tay nhanh hoặc dự đoán hàng loạt qua file CSV).
* `sales_model.joblib`: Tệp chứa mô hình hồi quy tối ưu đã được nén và lưu trữ.

---

##  3. Hướng dẫn cài đặt và sử dụng

### Bước 1: Cài đặt các thư viện phụ thuộc
Đảm bảo môi trường Python của bạn đã cài đặt các thư viện cần thiết:
pip install pandas numpy scikit-learn joblib

### Bước 2: Huấn luyện mô hình 
Chạy file main.py để tiến hành đọc dữ liệu, huấn luyện, đánh giá và lưu mô hình tốt nhất
  python main.py

### Bước 3: Dự đoán kịch bản mới
Xem các cột dữ liệu mô hình yêu cầu
  python predict_sales.py --info

Dự đoán nhanh bằng cách nhập vào một thông tin về hàng hóa bất kì
    ví dụ: python predict_sales.py --set Date=15-08-2026 Product_Category=Electronics Price=1200 Discount=10 Customer_Segment=Premium Marketing_Spend=8000

Dự đoán hàng loạt thông tin bằng cách đưa dữ liệu vào một file csv
    python predict_sales.py -c ten_file_kich_ban.csv -o ket_qua_du_doan.csv


