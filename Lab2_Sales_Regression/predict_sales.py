import argparse
import sys

import joblib
import pandas as pd

from main import MODEL_PATH, TARGET_COL, predict_sales


def load_bundle(path) -> dict:
    try:
        return joblib.load(path)
    except FileNotFoundError:
        sys.exit(f"Không tìm thấy '{path}'. Hãy chạy main.py trước để huấn luyện và lưu mô hình.")


def parse_assignments(pairs, raw_columns):
    """['Price=19.9', 'Product_Category=Toys'] -> {'Price': 19.9, 'Product_Category': 'Toys'}
    Tên cột không phân biệt hoa/thường."""
    lookup = {c.lower(): c for c in raw_columns}
    row = {}
    for pair in pairs:
        if "=" not in pair:
            sys.exit(f"Sai định dạng '{pair}'. Dùng dạng TenCot=GiaTri.")
        key, value = pair.split("=", 1)
        col = lookup.get(key.strip().lower())
        if col is None:
            sys.exit(f"Cột '{key}' không có trong mô hình. Các cột hợp lệ: {raw_columns}")
        try:
            row[col] = float(value)
        except ValueError:
            row[col] = value.strip()
    return row


def main():
    parser = argparse.ArgumentParser(description="Dự đoán số lượng bán bằng mô hình đã lưu")
    parser.add_argument("-m", "--model", default=str(MODEL_PATH), help="đường dẫn file mô hình")
    parser.add_argument("--info", action="store_true", help="in ra các cột mô hình cần")
    parser.add_argument("--set", nargs="+", metavar="COT=GIATRI",
                        help="một kịch bản, vd Price=500 Product_Category=Toys")
    parser.add_argument("-c", "--csv", help="file CSV chứa nhiều kịch bản")
    parser.add_argument("-o", "--output", help="file CSV lưu kết quả (dùng với --csv)")
    args = parser.parse_args()

    bundle = load_bundle(args.model)

    if args.info:
        print("Mô hình:", bundle["model_name"])
        print(f"Các cột dữ liệu gốc mô hình sử dụng (dự đoán {TARGET_COL}):")
        for col in bundle["raw_columns"]:
            print("  -", col)
        print("Cột nào không cung cấp sẽ được điền giá trị mặc định (median / phổ biến nhất).")

    elif args.set:
        df = pd.DataFrame([parse_assignments(args.set, bundle["raw_columns"])])
        print(f"{TARGET_COL} dự đoán: {predict_sales(bundle, df)[0]:.1f}")

    elif args.csv:
        df = pd.read_csv(args.csv)
        df["predicted_units_sold"] = predict_sales(bundle, df).round(1)
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Đã lưu kết quả vào {args.output}")
        else:
            print(df.head(10).to_string())

    else:
        parser.print_help()


if __name__ == "__main__":
    main()