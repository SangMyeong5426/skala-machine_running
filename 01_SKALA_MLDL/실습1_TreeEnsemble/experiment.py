"""실습1 - Tree Ensemble(XGBoost) : Perfect Fitting / Best Fitting 실험

노트북(실습_TreeEnsemble.ipynb)과 동일한 전처리·분할을 쓰고,
하이퍼파라미터 세트만 바꿔가며 Train/Val/Test 성능을 표로 뽑는다.
"""
import json, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "wine_quality.csv")

# ---------------------------------------------------------------- 1. 데이터
if os.path.exists(CACHE):
    df_raw = pd.read_csv(CACHE)
else:
    from ucimlrepo import fetch_ucirepo
    wq = fetch_ucirepo(id=186)
    df_raw = pd.concat([wq.data.features, wq.data.targets["quality"]], axis=1)
    df_raw.to_csv(CACHE, index=False)

X = df_raw.drop(columns=["quality"])
y = df_raw["quality"]
print(f"Data Shape        : {df_raw.shape}")
print(f"Categorical Vars  : {X.select_dtypes(include=['object']).columns.tolist()}")
print(f"Target range      : {y.min()} ~ {y.max()}  (unique={sorted(y.unique())})")

# ---------------------------------------------------------------- 2. 전처리
X = pd.get_dummies(X, drop_first=True)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
print(f"\nTrain {len(X_train)} / Val {len(X_val)} / Test {len(X_test)}   (features={X.shape[1]})")

# ------------------------------------------- 2-b. Perfect Fitting 의 이론적 하한
# feature 가 완전히 같은데 target 이 다른 행은 어떤 트리도 분리할 수 없다.
tr = X_train.copy()
tr["__y"] = y_train.values
g = tr.groupby(list(X_train.columns), sort=False)["__y"]
n_groups, n_rows = g.ngroups, len(tr)
conflict = g.nunique() > 1
n_conflict_groups = int(conflict.sum())
n_conflict_rows = int(g.size()[conflict].sum()) if n_conflict_groups else 0
floor_mean = float((tr["__y"] - g.transform("mean")).abs().mean())   # squared-error 목적함수의 하한
floor_median = float((tr["__y"] - g.transform("median")).abs().mean())  # MAE 이론 하한
print(f"\n=== Perfect Fitting 하한 (Train 기준) ===")
print(f"고유 feature 조합 : {n_groups} / {n_rows} 행")
print(f"target 이 충돌하는 조합 : {n_conflict_groups}개 ({n_conflict_rows}행)")
print(f"도달 가능한 최소 Train MAE : {floor_mean:.4f} (leaf=mean) / {floor_median:.4f} (leaf=median)")

# ---------------------------------------------------------------- 3. 실행 헬퍼
def run(name, params, es=None):
    p = dict(params)
    kw = dict(eval_metric="mae")
    if es:
        kw["early_stopping_rounds"] = es   # xgboost 2.x+ : 생성자 인자
    m = XGBRegressor(**p, **kw)
    m.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
    pr = {k: m.predict(v) for k, v in
          {"train": X_train, "val": X_val, "test": X_test}.items()}
    act = {"train": y_train, "val": y_val, "test": y_test}
    r = {"name": name, "params": p,
         "best_iteration": getattr(m, "best_iteration", None),
         "n_trees": m.get_booster().num_boosted_rounds()}
    for k in pr:
        r[f"{k}_mae"] = mean_absolute_error(act[k], pr[k])
        r[f"{k}_r2"] = r2_score(act[k], pr[k])
    r["gap_mae"] = r["val_mae"] - r["train_mae"]
    r["_curve"] = m.evals_result()
    return r

BASE = dict(n_estimators=100, max_depth=3, learning_rate=0.1,
            min_child_weight=10, subsample=1, colsample_bytree=1, random_state=42)

# ---------------------------------------------------------------- 4. Task 1
# "제약을 푼다" 를 단계적으로 : 깊이↑ 개수↑ 정규화↓
ESCALATION = [
    ("Baseline",              BASE),
    ("① depth 3→6",           {**BASE, "max_depth": 6}),
    ("② +mcw 10→1",           {**BASE, "max_depth": 6, "min_child_weight": 1}),
    ("③ +depth 6→10",         {**BASE, "max_depth": 10, "min_child_weight": 1}),
    ("④ +n_est 100→1000",     {**BASE, "max_depth": 10, "min_child_weight": 1,
                               "n_estimators": 1000}),
    ("⑤ Perfect(정규화 해제)", {**BASE, "max_depth": 15, "min_child_weight": 1,
                               "n_estimators": 2000, "learning_rate": 0.3,
                               "reg_lambda": 0, "gamma": 0}),
]
print("\n" + "=" * 96)
print("Task 1) Perfect Fitting — 제약을 단계적으로 풀었을 때")
print("=" * 96)
print(f"{'설정':<24}{'Train MAE':>11}{'Val MAE':>10}{'Test MAE':>10}"
      f"{'Gap':>9}{'Train R2':>10}{'Val R2':>9}{'Test R2':>9}")
esc = []
for name, p in ESCALATION:
    r = run(name, p); esc.append(r)
    print(f"{name:<24}{r['train_mae']:>11.4f}{r['val_mae']:>10.4f}{r['test_mae']:>10.4f}"
          f"{r['gap_mae']:>9.4f}{r['train_r2']:>10.4f}{r['val_r2']:>9.4f}{r['test_r2']:>9.4f}")

# ---------------------------------------------------------------- 5. Task 2
print("\n" + "=" * 96)
print("Task 2) Best Fitting — Validation 기준 그리드 탐색 (early stopping 50)")
print("=" * 96)
grid, results = [], []
for md in [3, 4, 5, 6, 8]:
    for lr in [0.05, 0.1]:
        for mcw in [1, 5, 10]:
            for sub in [0.8, 1.0]:
                for col in [0.8, 1.0]:
                    grid.append(dict(BASE, n_estimators=2000, max_depth=md,
                                     learning_rate=lr, min_child_weight=mcw,
                                     subsample=sub, colsample_bytree=col))
for i, p in enumerate(grid):
    results.append(run(f"grid{i}", p, es=50))
results.sort(key=lambda r: r["val_mae"])
print(f"탐색한 조합 : {len(grid)}개\n")
print(f"{'순위':<5}{'depth':>6}{'lr':>6}{'mcw':>5}{'sub':>5}{'col':>5}{'trees':>7}"
      f"{'Train MAE':>11}{'Val MAE':>10}{'Test MAE':>10}{'Gap':>9}")
for i, r in enumerate(results[:10], 1):
    p = r["params"]
    print(f"{i:<5}{p['max_depth']:>6}{p['learning_rate']:>6}{p['min_child_weight']:>5}"
          f"{p['subsample']:>5}{p['colsample_bytree']:>5}{(r['best_iteration'] or 0) + 1:>7}"
          f"{r['train_mae']:>11.4f}{r['val_mae']:>10.4f}{r['test_mae']:>10.4f}{r['gap_mae']:>9.4f}")

best = results[0]
# Gap 이 작으면서 Val 이 최상위권(최저 Val MAE 대비 +1% 이내)인 대안도 같이 본다
near = [r for r in results if r["val_mae"] <= best["val_mae"] * 1.01]
lean = min(near, key=lambda r: r["gap_mae"])
print(f"\nVal 최저          : depth={best['params']['max_depth']} "
      f"lr={best['params']['learning_rate']} mcw={best['params']['min_child_weight']} "
      f"sub={best['params']['subsample']} col={best['params']['colsample_bytree']} "
      f"→ Val {best['val_mae']:.4f} / Gap {best['gap_mae']:.4f}")
print(f"Val 상위+Gap 최소 : depth={lean['params']['max_depth']} "
      f"lr={lean['params']['learning_rate']} mcw={lean['params']['min_child_weight']} "
      f"sub={lean['params']['subsample']} col={lean['params']['colsample_bytree']} "
      f"→ Val {lean['val_mae']:.4f} / Gap {lean['gap_mae']:.4f}  "
      f"({len(near)}개 후보 중)")

# ---------------------------------------------------------------- 6. 그래프
def curve(ax, r, title, logx=False):
    tr_c = r["_curve"]["validation_0"]["mae"]
    va_c = r["_curve"]["validation_1"]["mae"]
    ep = range(1, len(tr_c) + 1)
    ax.plot(ep, tr_c, color="tab:blue", lw=1.8, label="Train MAE")
    ax.plot(ep, va_c, color="tab:red", lw=1.8, label="Validation MAE")
    if r.get("best_iteration") is not None:
        ax.axvline(r["best_iteration"] + 1, color="gray", ls="--", lw=1,
                   label=f"best iter = {r['best_iteration']+1}")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel("Boosting Round"); ax.set_ylabel("MAE")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_ylim(0, max(max(va_c), max(tr_c)) * 1.1)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
curve(axes[0], esc[0], f"(a) Baseline\nTrain {esc[0]['train_mae']:.3f} / "
      f"Val {esc[0]['val_mae']:.3f} / Gap {esc[0]['gap_mae']:.3f}")
curve(axes[1], esc[-1], f"(b) Task1 Perfect Fitting\nTrain {esc[-1]['train_mae']:.3f} / "
      f"Val {esc[-1]['val_mae']:.3f} / Gap {esc[-1]['gap_mae']:.3f}", logx=True)
curve(axes[2], best, f"(c) Task2 Best Fitting\nTrain {best['train_mae']:.3f} / "
      f"Val {best['val_mae']:.3f} / Gap {best['gap_mae']:.3f}")
plt.tight_layout(); plt.savefig(os.path.join(HERE, "learning_curves.png"), dpi=140)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
labels = [r["name"] for r in esc]
xs = range(len(esc))
ax[0].plot(xs, [r["train_mae"] for r in esc], "o-", color="tab:blue", label="Train MAE")
ax[0].plot(xs, [r["val_mae"] for r in esc], "o-", color="tab:red", label="Validation MAE")
ax[0].plot(xs, [r["test_mae"] for r in esc], "o--", color="tab:orange", label="Test MAE")
ax[0].axhline(floor_mean, color="gray", ls=":", label=f"train floor {floor_mean:.3f}")
ax[0].set_xticks(list(xs)); ax[0].set_xticklabels([f"{i}" for i in xs])
ax[0].set_xlabel("complexity step (0=Baseline … 5=Perfect)"); ax[0].set_ylabel("MAE")
ax[0].set_title("Bias-Variance Tradeoff : 복잡도를 올릴수록"); ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)
ax[1].bar(list(xs), [r["gap_mae"] for r in esc], color="tab:purple", alpha=0.75)
ax[1].set_xticks(list(xs)); ax[1].set_xticklabels([f"{i}" for i in xs])
ax[1].set_xlabel("complexity step"); ax[1].set_ylabel("Val MAE - Train MAE")
ax[1].set_title("Variance 지표 : Train-Val 성능 차이"); ax[1].grid(alpha=0.3)
for a in ax:
    for lbl in a.get_xticklabels():
        lbl.set_fontsize(9)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "tradeoff.png"), dpi=140)

# ---------------------------------------------------------------- 7. 저장
out = {"floor_mean": floor_mean, "floor_median": floor_median,
       "n_conflict_groups": n_conflict_groups, "n_conflict_rows": n_conflict_rows,
       "n_groups": n_groups, "n_rows": n_rows,
       "escalation": [{k: v for k, v in r.items() if k != "_curve"} for r in esc],
       "best": {k: v for k, v in best.items() if k != "_curve"},
       "lean": {k: v for k, v in lean.items() if k != "_curve"},
       "grid_top10": [{k: v for k, v in r.items() if k != "_curve"} for r in results[:10]]}
with open(os.path.join(HERE, "results.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=str)
print("\n저장: learning_curves.png / tradeoff.png / results.json")
