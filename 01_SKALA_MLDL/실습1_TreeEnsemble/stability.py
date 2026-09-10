"""Gap 이 큰 모델은 정말 불안정한가 — 데이터 분할을 5번 바꿔서 확인"""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

df = pd.read_csv("wine_quality.csv")
X, y = pd.get_dummies(df.drop(columns=["quality"]), drop_first=True), df["quality"]

M = {
 "Baseline (Gap 0.05)":    dict(n_estimators=100, max_depth=3, learning_rate=0.1,
                                min_child_weight=10, subsample=1, colsample_bytree=1),
 "우리 Best (Gap 0.36)":    dict(n_estimators=800, max_depth=8, learning_rate=0.05,
                                min_child_weight=3, subsample=0.7, colsample_bytree=0.7,
                                reg_lambda=20),
 "조원 Best (Gap 0.43)":    dict(n_estimators=1500, max_depth=15, learning_rate=0.03,
                                min_child_weight=5, subsample=0.7, colsample_bytree=0.8),
 "Perfect  (Gap 0.51)":    dict(n_estimators=2000, max_depth=15, learning_rate=0.3,
                                min_child_weight=1, subsample=1, colsample_bytree=1,
                                reg_lambda=0),
}
SEEDS = [42, 0, 1, 7, 123]
res = {n: {"te": [], "va": []} for n in M}
for s in SEEDS:
    Xt, Xte, yt, yte = train_test_split(X, y, test_size=0.2, random_state=s)
    Xtr, Xv, ytr, yv = train_test_split(Xt, yt, test_size=0.2, random_state=s)
    for n, p in M.items():
        m = XGBRegressor(**p, random_state=42).fit(Xtr, ytr)
        res[n]["te"].append(mean_absolute_error(yte, m.predict(Xte)))
        res[n]["va"].append(mean_absolute_error(yv, m.predict(Xv)))

print("분할 seed 를 5번 바꿔가며 측정한 Test MAE")
print(f"{'':<22}" + "".join(f"{f'seed={s}':>9}" for s in SEEDS)
      + f"{'평균':>9}{'표준편차':>10}{'최대-최소':>11}")
print("-" * 100)
for n in M:
    v = np.array(res[n]["te"])
    print(f"{n:<22}" + "".join(f"{x:>9.4f}" for x in v)
          + f"{v.mean():>9.4f}{v.std():>10.4f}{v.max()-v.min():>11.4f}")

print("\nseed=42 에서 최고였던 모델이 다른 분할에서도 최고인가")
print(f"{'seed':<8}{'1위 모델':<24}{'Val 1위 = Test 1위?':>22}")
print("-" * 56)
for i, s in enumerate(SEEDS):
    bt = min(M, key=lambda n: res[n]["te"][i])
    bv = min(M, key=lambda n: res[n]["va"][i])
    print(f"{s:<8}{bt:<24}{'예' if bt == bv else f'아니오 (Val 1위: {bv})':>22}")
