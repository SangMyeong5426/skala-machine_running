"""손으로 추론해서 고른 값들이 250개 탐색 결과에 얼마나 근접하나"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

df = pd.read_csv("wine_quality.csv")
X, y = pd.get_dummies(df.drop(columns=["quality"]), drop_first=True), df["quality"]
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

CASES = {
 "① Baseline (주어진 값)":        dict(n_estimators=100, max_depth=3, learning_rate=0.1,
                                     min_child_weight=10, subsample=1, colsample_bytree=1),
 "② 깊이만 3→6":                  dict(n_estimators=100, max_depth=6, learning_rate=0.1,
                                     min_child_weight=10, subsample=1, colsample_bytree=1),
 "③ +트리 100→500, lr 0.1→0.05":  dict(n_estimators=500, max_depth=6, learning_rate=0.05,
                                     min_child_weight=10, subsample=1, colsample_bytree=1),
 "④ +샘플링 0.8, mcw 10→5":       dict(n_estimators=500, max_depth=6, learning_rate=0.05,
                                     min_child_weight=5, subsample=0.8, colsample_bytree=0.8),
 "⑤ 교과서적 상식값":              dict(n_estimators=1000, max_depth=6, learning_rate=0.05,
                                     min_child_weight=1, subsample=0.8, colsample_bytree=0.8),
 "★ 탐색으로 찾은 Best":          dict(n_estimators=800, max_depth=8, learning_rate=0.05,
                                     min_child_weight=3, subsample=0.7, colsample_bytree=0.7,
                                     reg_lambda=20),
}
print(f"{'':<30}{'Train MAE':>11}{'Val MAE':>10}{'Test MAE':>10}{'Gap':>9}{'Test R²':>10}")
print("-" * 80)
for n, p in CASES.items():
    m = XGBRegressor(**p, random_state=42).fit(X_train, y_train)
    tr, va, te = (m.predict(v) for v in (X_train, X_val, X_test))
    a, b, c = (mean_absolute_error(x, z) for x, z in
               ((y_train, tr), (y_val, va), (y_test, te)))
    print(f"{n:<30}{a:>11.4f}{b:>10.4f}{c:>10.4f}{b-a:>9.4f}{r2_score(y_test, te):>10.4f}")
