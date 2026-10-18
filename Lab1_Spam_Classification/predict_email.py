"""
Tái sử dụng mô hình đã huấn luyện (spam_model.joblib) để dự đoán email mới.

Chạy sau khi đã chạy xong spam_classifier.py (file mô hình được tạo ra ở đó).

Cách dùng:
  python predict_email.py                         # chế độ nhập tay (tương tác)
  python predict_email.py -t "Win a FREE prize!"  # dự đoán một đoạn văn bản
  python predict_email.py -f email1.txt email2.txt  # dự đoán từ các file .txt
  python predict_email.py -c new_emails.csv --col text -o ket_qua.csv  # dự đoán cả file CSV
"""

import argparse
import re
import sys

import joblib
import pandas as pd

MODEL_PATH = "spam_model.joblib"


def clean_text(text: str) -> str:
    """PHẢI giống hệt hàm clean_text trong spam_classifier.py,
    vì mô hình được huấn luyện trên văn bản đã làm sạch theo cách này."""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_model(path: str = MODEL_PATH):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        sys.exit(f"Không tìm thấy '{path}'. Hãy chạy spam_classifier.py trước "
                 f"và chạy script này trong cùng thư mục chứa file mô hình.")


def predict(model, texts):
    """Trả về danh sách (nhãn, độ tin cậy) cho từng email."""
    cleaned = [clean_text(t) for t in texts]
    labels = model.predict(cleaned)

    scores = [None] * len(cleaned)
    if hasattr(model, "predict_proba"):            # Logistic Regression, Naive Bayes
        scores = model.predict_proba(cleaned)[:, 1]  # xác suất là spam
    elif hasattr(model, "decision_function"):      # SVM: điểm > 0 nghiêng về spam
        scores = model.decision_function(cleaned)

    return [("SPAM" if l == 1 else "NOT SPAM", s) for l, s in zip(labels, scores)]


def show(text, label, score):
    preview = text.strip().replace("\n", " ")[:80]
    extra = f" (điểm: {score:.3f})" if score is not None else ""
    print(f"[{label}]{extra}  {preview}")


def interactive(model):
    print("Nhập nội dung email rồi nhấn Enter (gõ 'q' để thoát).")
    while True:
        text = input("\n> ").strip()
        if text.lower() in {"q", "quit", "exit"}:
            break
        if not text:
            continue
        (label, score), = predict(model, [text])
        show(text, label, score)


def main():
    parser = argparse.ArgumentParser(description="Dự đoán email spam bằng mô hình đã lưu")
    parser.add_argument("-m", "--model", default=MODEL_PATH, help="đường dẫn file mô hình")
    parser.add_argument("-t", "--text", help="một đoạn văn bản email")
    parser.add_argument("-f", "--files", nargs="+", help="một hoặc nhiều file .txt")
    parser.add_argument("-c", "--csv", help="file CSV chứa các email cần dự đoán")
    parser.add_argument("--col", default="text", help="tên cột nội dung trong CSV")
    parser.add_argument("-o", "--output", help="file CSV để lưu kết quả (dùng với --csv)")
    args = parser.parse_args()

    model = load_model(args.model)

    if args.text:
        (label, score), = predict(model, [args.text])
        show(args.text, label, score)

    elif args.files:
        texts = []
        for path in args.files:
            with open(path, encoding="utf-8", errors="ignore") as f:
                texts.append(f.read())
        for path, text, (label, score) in zip(args.files, texts, predict(model, texts)):
            print(f"{path}:", end=" ")
            show(text, label, score)

    elif args.csv:
        df = pd.read_csv(args.csv)
        results = predict(model, df[args.col].fillna("").tolist())
        df["prediction"] = [r[0] for r in results]
        df["score"] = [r[1] for r in results]
        print(df[["prediction", "score"]].value_counts("prediction"))
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Đã lưu kết quả vào {args.output}")
        else:
            print(df.head(10))

    else:
        interactive(model)


if __name__ == "__main__":
    main()