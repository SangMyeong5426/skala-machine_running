# 교육 실습 작업 폴더

```
├── 01_SKALA_MLDL/          머신러닝 및 딥러닝 이해 (3일, 완료)
└── 02_ML_ToyProject/       Machine Learning 이해 및 Toy Project (2일, 진행 중)
```

---

## 01_SKALA_MLDL — 머신러닝 및 딥러닝 이해 (완료)

| 폴더 | 실습 | 데이터 | 최종 결과 |
| --- | --- | --- | --- |
| `실습1_TreeEnsemble` | Day1 · XGBoost | Wine Quality | 조별 Best: depth15/1500트리/lr0.03 · **Test R² 0.4602** |
| `실습2_KNN` | Day2 · KNN | Breast Cancer | euclidean/uniform/**k=7** · **Test AUC 0.9825** |
| `실습3_Clustering` | Day2 · 군집화 | Wholesale Customers | log+KMeans **k=4** · 4개 세그먼트 **← 제출** |
| `실습4_ANN` | Day3 · 신경망 | Bank Marketing | 41→64→32→1 +Dropout/Early · **Test AUC 0.7913** |

**제출본** `실습3_Clustering/보고서/광주_3반_박상명_Day2_실습_Clustering_보고서.pdf` (2page)

### 폴더별 내용

- **실습1_TreeEnsemble** — 노트북 + 통제실험 스크립트 9개(`culprit.py`, `search_best.py`, `stability.py` …) + 결과 JSON/CSV
- **실습2_KNN** — 노트북(50셀) · `_원본` = 강사 배포본 · 계획서 · 그림 7장 + `거리_설명.png`
- **실습3_Clustering** — 노트북 3종(간결 36셀 / 상세 43셀 / 원본 20셀) · `보고서/` · 그림 8장
- **실습4_ANN** — 노트북(33셀) · `딥러닝_개념정리.md` · 그림 6장

### 실습을 관통한 결론

| 실습 | 배운 것 |
| --- | --- |
| 1 | **Gap 은 진단 지표이지 최소화 대상이 아니다** — Gap 최소 모델이 가장 나빴다 |
| 2 | **차이가 노이즈보다 큰지 보라** — metric 차이 0.0004 vs 표준편차 0.020 |
| 3 | **지표 최대화는 목적이 아니다** — Silhouette 1위가 [437, 1, 1, 1] |
| 4 | **정확도가 지표가 못 될 때가 있다** — 전부 '미가입' 으로 찍어도 88.3% |
| 공통 | **전처리가 모델 선택보다 결과를 좌우했다** (스케일링·로그변환) |

---

## 02_ML_ToyProject — Machine Learning 이해 및 Toy Project (진행 중)

교재 구성: 기계학습 개요 · 지도학습(회귀/분류) · 비지도학습(군집화/차원축소) ·
성능평가(회귀·분류 지표, 교차검증, 과적합) · Toy Project 실습

> 이미 다룬 주제와 상당 부분 겹친다 — KNN(실습2), K-Means/DBSCAN/Hierarchical(실습3),
> Decision Tree/RF(실습1), 분류 성능평가·ROC-AUC(실습4), 과적합/교차검증(실습1·2·4).

---

## 환경

`.venv` — sklearn 1.9 · pandas · numpy · matplotlib · seaborn · **xgboost 3.2** ·
**tensorflow 2.21** · scipy · ucimlrepo · markdown

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace <노트북>   # 실행
```

**PDF 변환** — Chrome headless 사용 (`pandoc`/`wkhtmltopdf` 불필요)
```bash
.venv/bin/python -c "import markdown, ..."   # md -> html (한글 폰트 CSS 포함)
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --disable-gpu --no-pdf-header-footer --print-to-pdf=out.pdf file:///.../in.html
```

> 노트북은 **자기 폴더 기준**으로 그림을 저장하므로, 폴더째 옮겨도 그대로 재실행된다.
