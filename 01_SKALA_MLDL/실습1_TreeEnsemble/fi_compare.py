"""Perfect vs Best : Feature Importance 분포 비교"""
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

df = pd.read_csv("wine_quality.csv")
X, y = pd.get_dummies(df.drop(columns=["quality"]), drop_first=True), df["quality"]
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

M = {
 "Baseline": dict(n_estimators=100, max_depth=3, learning_rate=0.1, min_child_weight=10,
                  subsample=1, colsample_bytree=1, random_state=42),
 "Perfect":  dict(n_estimators=2000, max_depth=15, learning_rate=0.3, min_child_weight=1,
                  subsample=1, colsample_bytree=1, reg_lambda=0, gamma=0, random_state=42),
 "Best":     dict(n_estimators=800, max_depth=8, learning_rate=0.05, min_child_weight=3,
                  subsample=0.7, colsample_bytree=0.7, reg_lambda=20, gamma=0, random_state=42),
}
imp = {}
for n, p in M.items():
    m = XGBRegressor(**p).fit(X_train, y_train)
    imp[n] = pd.Series(m.feature_importances_, index=X.columns)
t = pd.DataFrame(imp).sort_values("Perfect", ascending=False)
print(t.round(4).to_string())
print()
for n in M:
    s = t[n].sort_values(ascending=False)
    # 상위 1개 비중 / 상위 3개 비중 / 지니 계수(집중도)
    top1, top3 = s.iloc[0], s.iloc[:3].sum()
    v = np.sort(s.values); i = np.arange(1, len(v)+1)
    gini = (2*(i*v).sum())/(len(v)*v.sum()) - (len(v)+1)/len(v)
    print(f"{n:<9} 1위={s.index[0]:<12} 비중 {top1:.3f} | 상위3 {top3:.3f} | 집중도(Gini) {gini:.3f}")
