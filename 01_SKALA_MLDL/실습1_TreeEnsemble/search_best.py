"""Task 2 - Best Fitting : 복잡도 축을 넓게 잡은 랜덤 탐색 + Pareto frontier

앞선 그리드(n_estimators=2000 + early stopping)는 모든 후보가 Train 을 외워 버려
Gap 이 0.43 근처에 몰렸다. n_estimators / 정규화를 명시적 복잡도 축으로 넣어
(Val MAE, Gap) 의 상충 관계 자체를 그린다.
"""
import json, os, random
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "wine_quality.csv"))
X, y = pd.get_dummies(df.drop(columns=["quality"]), drop_first=True), df["quality"]
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

SPACE = dict(
    n_estimators=[50, 100, 200, 400, 800],
    max_depth=[2, 3, 4, 5, 6, 8, 10],
    learning_rate=[0.02, 0.05, 0.1, 0.2],
    min_child_weight=[1, 3, 5, 10, 20, 50],
    subsample=[0.6, 0.7, 0.8, 0.9, 1.0],
    colsample_bytree=[0.6, 0.7, 0.8, 0.9, 1.0],
    reg_lambda=[0, 1, 5, 20, 100],
    gamma=[0, 0.1, 0.5],
)
rng = random.Random(42)
seen, configs = set(), []
while len(configs) < 250:
    c = {k: rng.choice(v) for k, v in SPACE.items()}
    key = tuple(sorted(c.items()))
    if key not in seen:
        seen.add(key); configs.append(c)

rows = []
for i, c in enumerate(configs):
    m = XGBRegressor(**c, random_state=42, eval_metric="mae")
    m.fit(X_train, y_train, verbose=False)
    p = {k: m.predict(v) for k, v in {"train": X_train, "val": X_val, "test": X_test}.items()}
    a = {"train": y_train, "val": y_val, "test": y_test}
    r = dict(c)
    for k in p:
        r[f"{k}_mae"] = mean_absolute_error(a[k], p[k]); r[f"{k}_r2"] = r2_score(a[k], p[k])
    r["gap"] = r["val_mae"] - r["train_mae"]
    rows.append(r)
    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{len(configs)} …", flush=True)

res = pd.DataFrame(rows)
res.to_csv(os.path.join(HERE, "search_results.csv"), index=False)

# Pareto frontier : Val MAE 와 Gap 을 동시에 최소화 (지배당하지 않는 점)
v, g = res["val_mae"].to_numpy(), res["gap"].to_numpy()
pareto = np.array([not np.any((v <= v[i]) & (g <= g[i]) & ((v < v[i]) | (g < g[i])))
                   for i in range(len(res))])
res["pareto"] = pareto

TOL = 0.01                                   # Val MAE 최저 대비 허용 손실
cand = res[res.val_mae <= res.val_mae.min() + TOL]
best = cand.loc[cand.gap.idxmin()]           # 추천 Best : 성능 상위권 중 Gap 최소
lowest = res.loc[res.val_mae.idxmin()]       # 비교용 : Val 만 보고 고른 모델

KEYS = ["n_estimators", "max_depth", "learning_rate", "min_child_weight",
        "subsample", "colsample_bytree", "reg_lambda", "gamma"]
def show(tag, r):
    print(f"\n[{tag}]")
    print("  " + "  ".join(f"{k}={r[k]:g}" for k in KEYS))
    print(f"  Train MAE {r.train_mae:.4f} | Val MAE {r.val_mae:.4f} | Test MAE {r.test_mae:.4f}"
          f" | Gap {r.gap:.4f}")
    print(f"  Train R²  {r.train_r2:.4f} | Val R²  {r.val_r2:.4f} | Test R²  {r.test_r2:.4f}")

print(f"\n탐색 {len(res)}개 / Pareto 최적 {int(pareto.sum())}개")
show(f"Val MAE 만 최소화", lowest)
show(f"추천 Best (Val ≤ 최저+{TOL}, 그 중 Gap 최소)", best)

print("\n=== Pareto frontier (Val MAE 오름차순) ===")
pf = res[res.pareto].sort_values("val_mae")
print(f"{'n_est':>6}{'depth':>6}{'lr':>6}{'mcw':>5}{'sub':>5}{'col':>5}{'lmbd':>6}{'gam':>5}"
      f"{'Train':>9}{'Val':>8}{'Test':>8}{'Gap':>8}")
for _, r in pf.iterrows():
    print(f"{r.n_estimators:>6g}{r.max_depth:>6g}{r.learning_rate:>6g}{r.min_child_weight:>5g}"
          f"{r.subsample:>5g}{r.colsample_bytree:>5g}{r.reg_lambda:>6g}{r.gamma:>5g}"
          f"{r.train_mae:>9.4f}{r.val_mae:>8.4f}{r.test_mae:>8.4f}{r.gap:>8.4f}")

fig, ax = plt.subplots(figsize=(8.5, 6))
sc = ax.scatter(res.gap, res.val_mae, c=res.max_depth, cmap="viridis", s=26, alpha=0.75)
ax.plot(pf.gap, pf.val_mae, "r-o", ms=5, lw=1.2, alpha=0.8, label="Pareto frontier")
ax.scatter([best.gap], [best.val_mae], s=260, facecolors="none", edgecolors="red",
           lw=2.2, label="recommended Best")
ax.set_xlabel("Gap  =  Val MAE − Train MAE   (Variance ↑)")
ax.set_ylabel("Validation MAE   (Bias+Variance ↓ 가 좋음)")
ax.set_title("Bias-Variance Tradeoff : 250 random configs")
plt.colorbar(sc, label="max_depth"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "pareto.png"), dpi=140)

json.dump({"best": {k: float(best[k]) for k in KEYS + ["train_mae","val_mae","test_mae","gap",
                                                       "train_r2","val_r2","test_r2"]},
           "lowest_val": {k: float(lowest[k]) for k in KEYS + ["train_mae","val_mae","test_mae",
                                                               "gap","train_r2","val_r2","test_r2"]}},
          open(os.path.join(HERE, "best.json"), "w"), indent=2)
print("\n저장: search_results.csv / pareto.png / best.json")
