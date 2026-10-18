from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

# CẤU HÌNH
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "sales_data.csv"
MODEL_PATH = BASE_DIR / "sales_model.joblib"

TARGET_COL = "Units_Sold"   
DATE_COL = "Date"           
DATE_FORMAT = "%d-%m-%Y"    
DROP_COLS = []              

PRICE_COL = "Price"         
DISCOUNT_COL = "Discount"   
RATING_COL = None           
REVIEWS_COL = None

TEST_SIZE = 0.2
OUTLIER_FACTOR = 3.0        
RANDOM_STATE = 42


def _has(df: pd.DataFrame, col) -> bool:
    return col is not None and col in df.columns


def parse_dates(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format=DATE_FORMAT, errors="coerce")
    if parsed.isna().any():
        iso = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
        parsed = parsed.fillna(iso)
    return parsed


# 1. DATA PREPARATION
def load_data() -> pd.DataFrame:
    if not Path(DATA_PATH).exists():
        csv_files = [p.name for p in Path(BASE_DIR).glob("*.csv")]
        raise SystemExit(
            f"Không tìm thấy file dữ liệu: {DATA_PATH}\n"
            f"Các file .csv trong thư mục {BASE_DIR}: {csv_files or 'không có'}\n"
            f"Hãy đặt file CSV vào thư mục trên hoặc sửa DATA_PATH."
        )

    df = pd.read_csv(DATA_PATH)

    needed = [TARGET_COL] + ([DATE_COL] if DATE_COL else [])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Không thấy cột {missing} trong file. "
                         f"Các cột hiện có: {df.columns.tolist()}")

    df = df.drop(columns=DROP_COLS, errors="ignore").drop_duplicates()

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL])

    if _has(df, DATE_COL):
        df[DATE_COL] = parse_dates(df[DATE_COL])
        df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL)
    return df.reset_index(drop=True)


def split_data(df: pd.DataFrame):
    if _has(df, DATE_COL):
        cut = int(len(df) * (1 - TEST_SIZE))
        return df.iloc[:cut].copy(), df.iloc[cut:].copy()
    return train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def remove_target_outliers(train: pd.DataFrame) -> pd.DataFrame:
    q1, q3 = train[TARGET_COL].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - OUTLIER_FACTOR * iqr, q3 + OUTLIER_FACTOR * iqr
    kept = train[train[TARGET_COL].between(lo, hi)]
    print(f"Loại {len(train) - len(kept)} dòng ngoại lai khỏi tập train")
    return kept


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])


# 2. FEATURE ENGINEERING
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if _has(df, DATE_COL):
        d = parse_dates(df[DATE_COL])
        df["year"] = d.dt.year
        df["month"] = d.dt.month
        df["quarter"] = d.dt.quarter
        df["day_of_week"] = d.dt.dayofweek
        df["day_of_year"] = d.dt.dayofyear
        df["is_weekend"] = (d.dt.dayofweek >= 5).astype(float)
        df = df.drop(columns=[DATE_COL])

    if _has(df, PRICE_COL) and _has(df, DISCOUNT_COL):
        df["net_price"] = df[PRICE_COL] * (1 - df[DISCOUNT_COL] / 100)

    if _has(df, PRICE_COL):
        df["log_price"] = np.log1p(df[PRICE_COL].clip(lower=0))

    if _has(df, RATING_COL) and _has(df, REVIEWS_COL):
        df["popularity"] = df[RATING_COL] * np.log1p(df[REVIEWS_COL].clip(lower=0))
    return df


# 3. MODEL SELECTION
BASELINE_NAME = "Baseline (đoán bằng trung bình)"


def build_models(X_train: pd.DataFrame) -> dict:
    def pipe(estimator):
        return Pipeline([("prep", build_preprocessor(X_train)), ("model", estimator)])

    return {
        BASELINE_NAME: pipe(DummyRegressor(strategy="mean")),
        "Linear Regression": pipe(LinearRegression()),
        "Decision Tree": pipe(DecisionTreeRegressor(max_depth=5, min_samples_leaf=10,
                                                    random_state=RANDOM_STATE)),
        "Random Forest": pipe(RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                                    n_jobs=-1, random_state=RANDOM_STATE)),
        "Gradient Boosting": pipe(GradientBoostingRegressor(random_state=RANDOM_STATE)),
    }


# 4. MODEL TRAINING (+ HYPERPARAMETER TUNING)
def tune_gradient_boosting(pipeline, X_train, y_train, time_based: bool):
    grid = {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.03, 0.1],
        "model__max_depth": [2, 3],
    }
    cv = TimeSeriesSplit(n_splits=5) if time_based else 5
    search = GridSearchCV(pipeline, grid, cv=cv, n_jobs=-1,
                          scoring="neg_root_mean_squared_error")
    search.fit(X_train, y_train)
    print("Tham số tốt nhất:", search.best_params_,
          "| RMSE (CV):", round(-search.best_score_, 4))
    return search.best_estimator_


# 5. MODEL EVALUATION
def evaluate(name, model, X_test, y_test) -> dict:
    pred = np.clip(model.predict(X_test), 0, None)  
    mse = mean_squared_error(y_test, pred)
    return {"model": name, "MSE": mse, "RMSE": np.sqrt(mse),
            "MAE": mean_absolute_error(y_test, pred), "R2": r2_score(y_test, pred)}


# 6. MODEL DEPLOYMENT
def predict_sales(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    X = add_features(df)
    X = X.reindex(columns=bundle["feature_columns"])  
    return np.clip(bundle["model"].predict(X), 0, None)


# MAIN
def main():
    df = load_data()
    print(f"Đã đọc {len(df)} dòng từ {DATA_PATH}")
    train, test = split_data(df)
    train = remove_target_outliers(train)

    raw_columns = train.drop(columns=[TARGET_COL]).columns.tolist()
    X_train, y_train = add_features(train.drop(columns=[TARGET_COL])), train[TARGET_COL]
    X_test, y_test = add_features(test.drop(columns=[TARGET_COL])), test[TARGET_COL]
    print(f"Train: {len(X_train)} dòng | Test: {len(X_test)} dòng | "
          f"{X_train.shape[1]} đặc trưng")

    models = build_models(X_train)
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        results.append(evaluate(name, model, X_test, y_test))

    tuned = tune_gradient_boosting(models["Gradient Boosting"], X_train, y_train,
                                   time_based=_has(df, DATE_COL))
    models["Gradient Boosting (tuned)"] = tuned
    results.append(evaluate("Gradient Boosting (tuned)", tuned, X_test, y_test))

    summary = pd.DataFrame(results).set_index("model").round(4)
    print("\n=== BẢNG SO SÁNH (RMSE/MAE càng thấp càng tốt, R2 càng cao càng tốt) ===")
    print(summary.to_string())

    candidates = summary.drop(index=BASELINE_NAME)
    best_name = candidates["RMSE"].idxmin()
    if candidates.loc[best_name, "RMSE"] >= summary.loc[BASELINE_NAME, "RMSE"]:
        print("\nLƯU Ý: chưa mô hình nào đánh bại cách đoán bằng trung bình. "
              "Các đặc trưng hiện có gần như không giải thích được biến mục tiêu.")

    bundle = {
        "model": models[best_name],
        "feature_columns": X_train.columns.tolist(),  
        "raw_columns": raw_columns,                   
        "model_name": best_name,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nĐã lưu mô hình tốt nhất: {best_name} -> {MODEL_PATH}")


if __name__ == "__main__":
    main()