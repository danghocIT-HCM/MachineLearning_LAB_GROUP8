LAB1: SPAM EMAIL CLASSIFICATION

## Giới thiệu bài toán
Xây dựng hệ thống Machine Learning tự động phân loại tin nhắn/email thành hai nhãn: **Spam** (Thư rác) hoặc **Ham** (Thư bình thường) sử dụng các kỹ thuật xử lý ngôn ngữ tự nhiên (NLP) và thuật toán phân loại học máy.

## Cấu trúc dự án
- `emails.csv`: Bộ dữ liệu văn bản tin nhắn/email.
- `main.py`: Tiền xử lý văn bản (TF-IDF), huấn luyện các mô hình (Logistic Regression, SVM, Naive Bayes), tối ưu hóa (`GridSearchCV`) và lưu mô hình.
- `predict_email.py`: Tái sử dụng mô hình đã lưu để kiểm tra và dự đoán tin nhắn mới.
- `spam_model.joblib`: File mô hình đã được huấn luyện.

## Hướng dẫn chạy chương trình

1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install pandas scikit-learn joblib

2. Chạy và huấn luyện mô hình
   python main.py

3. Dự đoán email spam hay không spam bằng file predict_sales.py
   python predict_email.py --message "Congratulations you have won a free ticket"
   