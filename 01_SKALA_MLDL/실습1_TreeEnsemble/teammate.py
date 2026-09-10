"""조원 모델 vs 내 모델 vs Perfect 비교"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

df = pd.read_csv("wine_quality.csv")
X, y = pd.get_dummies(df.drop(columns=["quality"]), drop_first=True), df["quality"]
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

CASES = {
 "Baseline":            dict(n_estimators=100, max_depth=3, learning_rate=0.1,
                             min_child_weight=10, subsample=1, colsample_bytree=1),
 "Task1 Perfect(우리)":  dict(n_estimators=2000, max_depth=15, learning_rate=0.3,
                             min_child_weight=1, subsample=1, colsample_bytree=1,
                             reg_lambda=0, gamma=0),
 "Task2 Best(우리)":     dict(n_estimators=800, max_depth=8, learning_rate=0.05,
                             min_child_weight=3, subsample=0.7, colsample_bytree=0.7,
                             reg_lambda=20, gamma=0),
 "★ 조원 모델":          dict(n_estimators=1500, max_depth=15, learning_rate=0.03,
                             min_child_weight=5, subsample=0.7, colsample_bytree=0.8),
}
print(f"{'':<22}{'Train MAE':>11}{'Val MAE':>10}{'Test MAE':>10}{'Gap':>9}"
      f"{'Train R²':>10}{'Val R²':>9}{'Test R²':>9}{'|Val-Test|':>11}")
print("-" * 101)
for n, p in CASES.items():
    m = XGBRegressor(**p, random_state=42).fit(X_train, y_train)
    tr, va, te = (m.predict(v) for v in (X_train, X_val, X_test))
    a, b, c = (mean_absolute_error(x, z) for x, z in ((y_train,tr),(y_val,va),(y_test,te)))
    r = [r2_score(x, z) for x, z in ((y_train,tr),(y_val,va),(y_test,te))]
    print(f"{n:<22}{a:>11.4f}{b:>10.4f}{c:>10.4f}{b-a:>9.4f}"
          f"{r[0]:>10.4f}{r[1]:>9.4f}{r[2]:>9.4f}{abs(b-c):>11.4f}")
