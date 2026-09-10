"""MAE 는 좋아졌는데 R² 는 나빠진 이유 : 오차 분포를 뜯어본다"""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor

df = pd.read_csv("wine_quality.csv")
X, y = pd.get_dummies(df.drop(columns=["quality"]), drop_first=True), df["quality"]
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

M = {"Baseline": dict(n_estimators=100, max_depth=3, learning_rate=0.1, min_child_weight=10,
                      subsample=1, colsample_bytree=1, random_state=42),
     "Perfect":  dict(n_estimators=2000, max_depth=15, learning_rate=0.3, min_child_weight=1,
                      subsample=1, colsample_bytree=1, reg_lambda=0, gamma=0, random_state=42)}
E = {}
for n, p in M.items():
    pred = XGBRegressor(**p).fit(X_train, y_train).predict(X_test)
    E[n] = (np.abs(y_test.to_numpy() - pred), pred)

print(f"{'':<12}{'MAE':>8}{'RMSE':>8}{'R²':>8}{'최대오차':>9}"
      f"{'|e|>1':>8}{'|e|>1.5':>9}{'|e|>2':>8}{'예측범위':>16}")
print("-" * 86)
for n, (e, pred) in E.items():
    print(f"{n:<12}{e.mean():>8.4f}{np.sqrt((e**2).mean()):>8.4f}"
          f"{r2_score(y_test, pred):>8.4f}{e.max():>9.2f}"
          f"{(e>1).mean()*100:>7.1f}%{(e>1.5).mean()*100:>8.1f}%{(e>2).mean()*100:>7.1f}%"
          f"{f'{pred.min():.2f} ~ {pred.max():.2f}':>16}")
print(f"{'실제 y_test':<12}{'':>41}{'':>24}{f'{y_test.min()} ~ {y_test.max()}':>16}")

b, p = E["Baseline"][0], E["Perfect"][0]
print(f"\n작은 오차(|e|≤1) 구간 평균 : Baseline {b[b<=1].mean():.4f} → Perfect {p[p<=1].mean():.4f}"
      f"  ({(p[p<=1].mean()-b[b<=1].mean())/b[b<=1].mean()*100:+.1f}%)")
print(f"큰 오차(|e|>1) 구간 평균  : Baseline {b[b>1].mean():.4f} → Perfect {p[p>1].mean():.4f}"
      f"  ({(p[p>1].mean()-b[b>1].mean())/b[b>1].mean()*100:+.1f}%)")
print(f"큰 오차가 전체 SSE 에서 차지하는 비중 : "
      f"Baseline {(b[b>1]**2).sum()/(b**2).sum()*100:.1f}% → Perfect {(p[p>1]**2).sum()/(p**2).sum()*100:.1f}%")
