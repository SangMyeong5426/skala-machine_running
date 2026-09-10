"""추가한 reg_lambda / gamma 가 실제로 일을 했나"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

df = pd.read_csv("wine_quality.csv")
X, y = pd.get_dummies(df.drop(columns=["quality"]), drop_first=True), df["quality"]
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

BEST = dict(n_estimators=800, max_depth=8, learning_rate=0.05, min_child_weight=3,
            subsample=0.7, colsample_bytree=0.7, random_state=42)
PERF = dict(n_estimators=2000, max_depth=15, learning_rate=0.3, min_child_weight=1,
            subsample=1, colsample_bytree=1, random_state=42)

def run(name, p):
    m = XGBRegressor(**p).fit(X_train, y_train)
    tr, va, te = (m.predict(v) for v in (X_train, X_val, X_test))
    a, b, c = (mean_absolute_error(x, z) for x, z in ((y_train,tr),(y_val,va),(y_test,te)))
    print(f"{name:<36}{a:>11.4f}{b:>10.4f}{c:>10.4f}{b-a:>9.4f}{r2_score(y_test,te):>10.4f}")

print(f"{'':<36}{'Train MAE':>11}{'Val MAE':>10}{'Test MAE':>10}{'Gap':>9}{'Test R²':>10}")
print("-" * 86)
print("[ Best 모델 ]")
run("λ 생략 (기본값 1)",              BEST)
run("λ=20  ← 내가 쓴 값",             dict(BEST, reg_lambda=20))
run("λ=20, gamma=0 (내가 쓴 그대로)",  dict(BEST, reg_lambda=20, gamma=0))
print("\n[ Perfect 모델 ]")
run("λ 생략 (기본값 1)",              PERF)
run("λ=0   ← 내가 쓴 값",             dict(PERF, reg_lambda=0))
