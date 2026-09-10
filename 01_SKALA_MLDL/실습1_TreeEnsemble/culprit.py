"""과적합의 범인은 n_estimators 인가 max_depth 인가"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

df = pd.read_csv("wine_quality.csv")
X, y = pd.get_dummies(df.drop(columns=["quality"]), drop_first=True), df["quality"]
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
B = dict(learning_rate=0.3, subsample=1, colsample_bytree=1, reg_lambda=0, gamma=0, random_state=42)

CASES = [
    ("Perfect 원본  (depth15, 2000트리)", dict(B, max_depth=15, min_child_weight=1, n_estimators=2000)),
    ("트리만 줄임   (depth15,   30트리)", dict(B, max_depth=15, min_child_weight=1, n_estimators=30)),
    ("깊이만 줄임   (depth 3, 2000트리)", dict(B, max_depth=3,  min_child_weight=1, n_estimators=2000)),
    ("깊이만 줄임   (depth 3,   30트리)", dict(B, max_depth=3,  min_child_weight=1, n_estimators=30)),
]
print(f"{'':<34}{'Train MAE':>11}{'Val MAE':>10}{'Test MAE':>10}{'Train R²':>10}{'Test R²':>9}")
print("-" * 84)
for name, p in CASES:
    m = XGBRegressor(**p).fit(X_train, y_train)
    pr = {k: m.predict(v) for k, v in {"tr": X_train, "va": X_val, "te": X_test}.items()}
    print(f"{name:<34}{mean_absolute_error(y_train, pr['tr']):>11.4f}"
          f"{mean_absolute_error(y_val, pr['va']):>10.4f}{mean_absolute_error(y_test, pr['te']):>10.4f}"
          f"{r2_score(y_train, pr['tr']):>10.4f}{r2_score(y_test, pr['te']):>9.4f}")
