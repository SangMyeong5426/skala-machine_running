# 실습2 KNN — 실습 계획

> **[정정 안내]** 아래 §1 의 수치는 `sklearn.datasets.load_breast_cancer` 로 미리 찔러본 값이다.
> 실제 과제는 `ucimlrepo(id=17)` + `LabelEncoder(B=0/M=1)` 를 쓰는데, 라벨 방향이 달라
> `stratify` 분할이 달라지므로 **실행값이 다르다.** 최종 수치는 `실습_KNN.ipynb` 와
> `조별토의_정리.md` 의 Day2 를 기준으로 한다. (거리 기여도 71.7%/99.6% 는 거의 동일)
>
> 주요 차이: 최적 k **5 → 3**, seed별 argmax `[3,7,3,7,7,7,5,5,11,9]` → `[3,5,3,3,3,7,5,3,3,3]`,
> CV 표준편차 ±0.015 → **±0.020**, 스케일링 효과 +0.03 → **+0.036**

> 교재: `머신러닝 및 딥러닝 이해 Day2 - Distance based model`
> 노트북: `실습_KNN.ipynb` (12 cells)
> Task 데이터: Breast Cancer Wisconsin Diagnostic (UCI id=17), 569 × 30, B 357 / M 212

---

## 0. 현황 — 노트북에서 내가 채워야 할 곳

| Cell | 내용 | 상태 |
| --- | --- | --- |
| 2 | numpy 저수준 KNN 구현 | 완성 (읽고 이해만) |
| 4~8 | Wine 데이터 예제 (k 비교 / 스케일링 비교 / 성과분석) | 완성 (실행값 있음) |
| 10 | `!pip install ucimlrepo` | 완성 |
| **11** | **Task: BCW 로드 → train_test_split 까지만 있음** | **← 여기부터 전부 내 몫** |

Task 요구: *"최적의 K값 탐색 등 KNN을 활용하여 예측성능 확보하기"*

---

## 1. 시작 전 확인한 것 — 이 데이터의 두 가지 사실

### ① 원척도에서 KNN은 사실상 "면적 1개짜리 모델"이다

제곱 유클리드 거리에 각 변수가 기여하는 비중(= 분산 비중):

| 변수 | 거리 기여도 |
| --- | ---: |
| worst area | **71.73%** |
| mean area | **27.41%** |
| area error | 0.46% |
| worst perimeter | 0.25% |
| **상위 3개 합계** | **99.60%** |

나머지 27개 변수가 나눠 갖는 몫이 0.4%다. 표준화 없이 30개 변수를 넣었다고 생각하지만
실제로는 `area` 계열 3개로 이웃을 찾고 있다. 교재 p.31 Cons —
*"X는 각 Data Point의 거리가 일관성 있고(정규화), 연속 변수로 채워진 Vector여야 함"* — 의
정량적 증거이고, 실습1과 달리 **여기서는 전처리가 성능의 주 원인**이다.

### ② "최적의 k"는 이 데이터에 존재하지 않는다

Train(455건)만으로 5-fold CV, seed만 0~9로 바꿔가며 최적 k를 뽑은 결과:

```
[3, 7, 3, 7, 7, 7, 5, 5, 11, 9]      ← seed 하나 바꾸면 답이 바뀐다
```

5-fold × 10 repeat 로 안정화시키면:

| k | CV Accuracy | 표준편차 |
| ---: | ---: | ---: |
| 1 | 0.9508 | ±0.0172 |
| 3 | 0.9646 | ±0.0146 |
| **5** | **0.9664** | ±0.0174 |
| 7 | 0.9659 | ±0.0154 |
| 9 | 0.9651 | ±0.0133 |
| 11 | 0.9626 | ±0.0142 |
| 21 | 0.9565 | ±0.0154 |
| 31 | 0.9512 | ±0.0170 |

k=3~9 구간의 차이는 **0.0018**, 표준편차는 **±0.015**. 차이가 흔들림의 1/8이다.
→ **결론은 "최적 k는 5"가 아니라 "k는 3~9 구간이면 되고 그 안에서는 구별 불가"**여야 한다.
이걸 하나로 찍어서 보고하면 실습1 리포트에서 지적한 ③번(*Train 성능은 근거가 될 수 없다*)과
같은 종류의 오류가 된다.

---

## 2. 주어진 예제 코드(cell 7~8)의 함정 — 실습1 교훈 연결

Task 를 풀기 전에 **예제 코드를 그대로 베끼면 안 되는 이유**를 먼저 정리한다.

### 함정 ① `best_k` 를 Test Accuracy 로 고른다 → 선택 오염

```python
# cell 7
for k in k_values:
    knn.fit(X_train_scaled, y_train)
    accuracy = accuracy_score(y_test, knn.predict(X_test_scaled))   # ← test로 평가
    accuracies_with_scaling.append(accuracy)

best_k_idx = np.argmax(accuracies_with_scaling)                     # ← 그 test 점수로 선택
```

Test 를 보고 고른 k 의 Test 성능은 더 이상 일반화 성능이 아니다.
실습1 리포트 ③ — *"Validation 을 보고 골랐으므로 Validation 도 오염된다.
진짜 성능은 한 번도 쓰지 않은 Test 로만 확인된다"* — 와 정확히 같은 구조다.
→ **Task 에서는 k 선택을 Train 내부 CV 로만 하고, Test 는 마지막에 딱 한 번 연다.**

### 함정 ② `Accuracy = 1.0000` 의 정체

Wine 예제의 Test 는 36건이다. 1건 = 2.78%p. 1.0000 은 "완벽"이 아니라
**"36번 던져서 안 틀렸다"**이며 신뢰구간이 사실상 없다.
BCW Test 는 114건(1건 = 0.88%p)이라 조금 낫지만 여전히 소수점 셋째 자리는 의미 없다.

### 함정 ③ `scaler` 재fit 오염 (cell 8, 4-5)

```python
X_all_scaled = scaler.fit_transform(X)   # ← 전체 X로 scaler를 다시 fit
```

시각화용이라 결과는 안 바뀌지만 `scaler` 객체가 Test 정보로 오염된다.
이후 이 `scaler` 를 다시 쓰면 leak. → **Task 에서는 `Pipeline` 으로 묶어서 원천 차단.**

---

## 3. 교재 ↔ `KNeighborsClassifier` 파라미터 매핑

실습에서 건드릴 손잡이가 교재 어느 페이지인지 먼저 붙여둔다.

| 교재 | 개념 | 코드 |
| --- | --- | --- |
| p.11 | Manhattan (L1) | `metric='manhattan'` = `p=1` |
| p.12 | Euclidean (L2) | `metric='euclidean'` = `p=2` (기본) |
| p.26 | Minkowski, p에 따라 특화 | `metric='minkowski', p=...` |
| p.15~16 | Cosine (크기 무시, 방향만) | `metric='cosine'` |
| p.25 | *"거리가 가까울수록 가중치를 부여할지"* | `weights='distance'` |
| p.25 | *"이웃 수가 적으면 이상치 민감 / 많으면 결과 왜곡"* | `n_neighbors` |
| p.27 | Feature Selection / Feature Weighting | 변수 부분집합 실험 |
| p.27 | Instance Selection (Editing) | 이상치 제거 실험 |
| p.29 | Lazy learning, *"학습 과정이 매우 빠름"* | fit / predict 시간 측정 |
| p.30 | 정규화 필수, 클래스 불균형, inference 비용 | Step 1 · 6 · 7 |

---

## 4. 실습 설계 — 7 Step

전 단계 공통 원칙: **`Pipeline(scaler → knn)` + `StratifiedKFold` 로 Train 안에서만 평가.
Test(114건)는 Step 6 에서 최종 1회만 사용.**

### Step 1. 스케일링이 왜 필수인지 "증명"한다 (15분)
말로 하지 말고 위 §1-① 표를 코드로 재현한다.

```python
v = X_train.var(axis=0); share = v / v.sum()
```

그 다음 원척도 vs 표준화 CV 를 나란히. (예상: 0.934 → 0.966, 약 +0.03)
Wine 예제는 +0.19 가 나왔는데 BCW 는 +0.03 이다. **왜 차이가 나는지**가 토의거리 —
Wine 은 `proline`(수백~1500) 하나가 지배해서 원척도가 아예 망가졌고,
BCW 는 지배 변수인 `area` 가 마침 악성/양성을 잘 가르는 변수라 원척도로도 93%는 나온다.
→ *"스케일링을 안 해도 그럭저럭 나오는" 경우가 더 위험하다.* 우연히 맞은 것이기 때문.

### Step 2. k 탐색을 올바른 프로토콜로 (25분) ★ 핵심
```python
ks = [1,3,5,7,9,11,15,21,31,51]
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
# Pipeline(StandardScaler, KNN(k)) 로 cross_val_score → mean ± std
```
산출물 2개:
- **k vs CV accuracy 곡선 + 표준편차 밴드(errorbar)** ← 밴드가 겹치는 걸 눈으로 보여주는 게 목적
- **seed 0~9 최적 k 리스트** → `[3,7,3,7,7,7,5,5,11,9]` 재현

동시에 **k=1 의 Train accuracy = 1.0000** 확인. 실습1의 "Perfect Fitting" 과제와 같은 것이
KNN 에서는 파라미터 한 개로 공짜로 나온다(자기 자신이 최근접 이웃). Train 성능이 왜 근거가
못 되는지의 가장 짧은 증명.

→ **결론 형식: "k=5 가 최적" (X) / "k=3~9 구간, 대표값 k=5, 구간 내 차이는 노이즈" (O)**

### Step 3. 거리 척도 × 가중치 격자 (20분)
| metric | weights |
| --- | --- |
| euclidean (p=2) | uniform / distance |
| manhattan (p=1) | uniform / distance |
| minkowski p=3 | uniform / distance |
| cosine | uniform / distance |

`GridSearchCV(Pipeline, param_grid, cv=cv, scoring=...)` 한 방.
관찰 포인트:
- **manhattan 이 euclidean 보다 나은가?** 30차원 고차원에서는 L1 이 유리하다는 것이 알려져 있음
  (제곱이 큰 차이를 과대반영하지 않음). 교재 p.13 *"맨하탄 거리는 항상 유클리드 거리보다 같거나 크다"* 와
  연결해서 **거리의 크기가 아니라 이웃의 순서만 바뀌면 된다**는 점을 짚기.
- **cosine 은 왜 여기서 애매한가?** 교재 p.16 — cosine 은 크기를 버리고 방향만 본다.
  문서 벡터에는 맞지만 BCW 는 **크기(종양이 크다) 자체가 악성 신호**다. 버리면 안 되는 정보를 버리는 것.
- `weights='distance'` 는 k 를 키워도 성능이 덜 떨어지게 만든다 → k 민감도 완화 효과 확인.

### Step 4. 스케일러 비교 (10분)
`StandardScaler` / `MinMaxScaler` / `RobustScaler` / (없음).
BCW 는 `*_worst` 변수들이 오른쪽 꼬리가 긴 분포라 MinMax 는 이상치 1개에 전체가 눌린다.
→ **"정규화" 라고 뭉뚱그리지 말고 어느 정규화인지가 결과를 바꾼다**는 것 확인.

### Step 5. 차원 / 변수 선택 (20분) — 교재 p.27
30차원에서 거리는 서로 비슷해진다(거리 집중). 두 가지로 확인:
- (a) 상위 상관 변수 k개만 / `SelectKBest` 로 5·10·15·20·30개 → CV 곡선
- (b) `PCA(n_components=…)` 를 Pipeline 중간에 끼워서 비교
- (c) `mean/se/worst` 세 그룹 중 **`worst` 10개만** 써보기 (도메인적으로 가장 판별력 높은 그룹)

관찰 포인트: **변수를 줄여도 성능이 안 떨어지면**, 그건 교재 p.27 Feature Selection 이
가중치를 0/1 로 주는 것이라는 설명의 실증이 된다. 게다가 KNN 은 예측 비용이 데이터 크기에
비례하므로(p.30) 변수 축소는 성능 유지 + 비용 절감이다.

### Step 6. 최종 평가 — 여기서 Test 를 처음 연다 (20분)
Step 2~5 로 고른 설정 **하나**를 Train 전체에 재학습 → Test 114건 1회 평가.

의료 데이터이므로 **accuracy 로 끝내면 안 된다**:
- `classification_report` — 특히 **malignant(악성) recall**
- Confusion Matrix — **FN(악성을 양성이라 함) 개수를 명시적으로 언급**
- ROC-AUC / PR-AUC
- `predict_proba` 는 **k개 이웃의 투표 비율**이라 k=5면 0, 0.2, 0.4, … 5단계뿐이다.
  → 임계값 조정의 해상도가 k 에 묶여 있다는 점 기록. (k 를 키우면 확률이 촘촘해진다)
- 클래스 불균형 63:37 → 교재 p.30 *"클래스 분포가 불균형할 경우 다수 투표를 지배"* 확인.
  전부 benign 이라 찍으면 62.7%. **62.7% 가 진짜 baseline.**

### Step 7. Lazy Learning 비용 측정 (5분) — 교재 p.29 / p.30
```
fit 0.63ms   /   predict 0.57ms (114건)
```
fit 이 predict 와 같은 수준이다 = **학습을 안 한다**는 것의 직접 증거.
XGBoost(실습1)와 비교해서 **비용이 학습이 아니라 추론에 있다**를 한 줄로 대비시킨다.
여유 있으면 데이터를 복제해 5배·10배로 늘려 predict 시간이 선형 증가하는 것까지.

---

## 5. 예상 결과 (미리 잡아둔 기준선)

| 구간 | 값 |
| --- | --- |
| 전부 benign 찍기 (진짜 baseline) | 0.627 |
| 원척도 KNN, CV | ≈ 0.934 |
| 표준화 KNN, CV | ≈ 0.966 |
| 튜닝 후 CV | 0.97 전후 |
| 최종 Test | 0.96 ~ 0.98 |

**0.99 나 1.0 을 목표로 잡지 않는다.** 이 데이터에서 CV 0.966 은 이미 상단이고,
그 위는 Test 를 훔쳐본 결과일 가능성이 높다. Test 114건에서 1.0 이 나오면
축하할 게 아니라 **의심해야 할 신호**다.

---

## 6. 시간 배분 (총 2시간 가정)

| | 내용 | 시간 |
| --- | --- | ---: |
| 0 | cell 2 저수준 구현 읽기 + wine 예제 실행 | 10분 |
| 1 | 스케일링 정량 증명 | 15분 |
| 2 | k 탐색 (올바른 프로토콜) ★ | 25분 |
| 3 | metric × weights 격자 | 20분 |
| 4 | 스케일러 비교 | 10분 |
| 5 | 변수 선택 / 차원 | 20분 |
| 6 | 최종 Test 평가 | 20분 |
| 7 | Lazy 비용 측정 | 5분 |
| — | 정리 / 리포트 | 15분 |

시간이 모자라면 **Step 4 와 5(b) PCA 를 먼저 버린다.** Step 1·2·6 은 못 버린다.

---

## 7. 조별 토의용 논점 (실습1 형식 따라감)

1. **"최적의 k" 를 하나로 답한 조와 구간으로 답한 조 중 누가 맞나** — seed 10개에서 답이
   3·5·7·9·11 로 다 나온다는 것을 근거로. 조원들 답이 갈렸다면 그게 곧 증거다.
2. **스케일링 효과가 Wine(+0.19) 과 BCW(+0.03) 에서 왜 다른가** —
   "스케일링 안 해도 잘 나오는 경우"의 위험성.
3. **거리 척도 선택은 데이터의 무엇을 보고 하는가** — cosine 이 문서에는 맞고 종양 크기에는
   안 맞는 이유 (크기가 정보인가 잡음인가).
4. **accuracy 96% 모델의 FN 이 몇 명인가** — 의료 맥락에서 지표 선택.
   실습1 리포트 ④ *"지표 하나만 보면 판단이 뒤집힌다"* 의 KNN 판.
5. **KNN 의 k 와 XGBoost 의 max_depth 는 같은 역할인가** —
   실습1 결론 ① *"max_depth 는 용량 상한선"* 과 대비.
   k 는 **작을수록** 복잡(k=1 이 최대 복잡), depth 는 **클수록** 복잡 → 방향이 반대다.
   KNN 에는 "제동 장치"가 사실상 `k` 와 `weights` 뿐이라 조합 폭발이 없다.

---

## 8. 산출물

| 파일 | 내용 |
| --- | --- |
| `실습_KNN.ipynb` (cell 11 이후) | Step 1~7 구현, 제출용 |
| `knn_search.py` | Step 2~5 격자 탐색 (실습1 `search_best.py` 대응) |
| `knn_stability.py` | seed 별 최적 k 흔들림 (실습1 `stability.py` 대응) |
| `실습2_조별토의_정리.md` | §7 논점 정리, 조원 결과 비교표 |

**그림 4장** (실습1 처럼 png 로):
1. k vs CV accuracy + 표준편차 밴드
2. 원척도 vs 표준화 거리 기여도 막대 (worst area 71.7%)
3. metric × weights 히트맵
4. Confusion Matrix + ROC/PR 곡선
