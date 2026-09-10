"""리포트에 실을 최종 3개 모델 : Baseline / Task1 Perfect / Task2 Best"""
import json, os
import pandas as pd, numpy as np
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

MODELS = {
 "Baseline": dict(n_estimators=100, max_depth=3, learning_rate=0.1, min_child_weight=10,
                  subsample=1, colsample_bytree=1, random_state=42),
 "Task1_Perfect": dict(n_estimators=2000, max_depth=15, learning_rate=0.3, min_child_weight=1,
                       subsample=1, colsample_bytree=1, reg_lambda=0, gamma=0, random_state=42),
 "Task2_Best": dict(n_estimators=800, max_depth=8, learning_rate=0.05, min_child_weight=3,
                    subsample=0.7, colsample_bytree=0.7, reg_lambda=20, gamma=0, random_state=42),
}
out, curves = {}, {}
for name, p in MODELS.items():
    m = XGBRegressor(**p, eval_metric="mae")
    m.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
    pr = {k: m.predict(v) for k, v in {"train":X_train,"val":X_val,"test":X_test}.items()}
    ac = {"train":y_train, "val":y_val, "test":y_test}
    r = {"params": p}
    for k in pr:
        r[f"{k}_mae"] = float(mean_absolute_error(ac[k], pr[k]))
        r[f"{k}_r2"]  = float(r2_score(ac[k], pr[k]))
    r["gap_mae"] = r["val_mae"] - r["train_mae"]
    r["gap_r2"]  = r["train_r2"] - r["val_r2"]
    out[name] = r; curves[name] = m.evals_result()

W = 16
print(f"{'':<16}{'Train MAE':>11}{'Val MAE':>10}{'Test MAE':>10}{'Gap(MAE)':>11}"
      f"{'Train R²':>10}{'Val R²':>9}{'Test R²':>9}")
print("-" * 76)
for n, r in out.items():
    print(f"{n:<16}{r['train_mae']:>11.4f}{r['val_mae']:>10.4f}{r['test_mae']:>10.4f}"
          f"{r['gap_mae']:>11.4f}{r['train_r2']:>10.4f}{r['val_r2']:>9.4f}{r['test_r2']:>9.4f}")

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
for ax, (name, r) in zip(axes, out.items()):
    tr, va = curves[name]["validation_0"]["mae"], curves[name]["validation_1"]["mae"]
    ep = np.arange(1, len(tr) + 1)
    ax.plot(ep, tr, color="tab:blue", lw=1.8, label="Train MAE")
    ax.plot(ep, va, color="tab:red", lw=1.8, label="Validation MAE")
    k = int(np.argmin(va))
    ax.plot(k + 1, va[k], "k*", ms=13, label=f"Val min {va[k]:.4f} @ {k+1}")
    if len(tr) > 500: ax.set_xscale("log")
    ax.set_xlabel("Boosting Round"); ax.set_ylabel("MAE"); ax.set_ylim(0, 0.75)
    ax.set_title(f"{name}\nTrain {r['train_mae']:.4f} / Val {r['val_mae']:.4f} / "
                 f"Gap {r['gap_mae']:.4f}", fontsize=10.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "learning_curves.png"), dpi=150)
json.dump(out, open(os.path.join(HERE, "final_models.json"), "w"), indent=2)
print("\n저장: learning_curves.png / final_models.json")
