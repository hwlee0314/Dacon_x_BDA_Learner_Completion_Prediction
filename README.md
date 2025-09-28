# 데이콘 x BDA 학습자 수료 예측 AI 경진대회 

### **2025-07-07 ~ 2025.08.25**
### [Competition Link](https://dacon.io/competitions/official/236519/overview/description)
- 온라인 중심의 교육 특성상 중도 이탈이 발생할 수밖에 없음
- BDA 8기의 학습 데이터를 분석하여 9기 학습자의 수료 여부를 예측하는 AI 알고리즘을 개발하는 것이 목표
- 학습자의 설문 정보를 바탕으로 교육 과정을 수료할 학습자를 식별하는 모델을 설계


- 평가 산식: f1-score  

- 2rd out of 275 teams 🏆

- 주최: 빅데이터분석학회  
  주관: 데이콘
---

### **주요 특징**

- **멀티모달 데이터 융합**: 테이블 형태의 개인정보와 자연어 텍스트 데이터를 **TF-IDF** 벡터화를 통해 결합하여 학습자의 다면적 특성을 포착했습니다.
- **교육 도메인 특화 피처 엔지니어링**: 수강 동기, 희망 직무, 관심 회사 등 교육 맥락에 특화된 파생변수를 생성하여 중도 탈락 패턴을 식별했습니다.
- **결측치 전략적 처리**: 결측 과다 컬럼은 제거하고, 나머지는 도메인 지식을 활용해 의미있는 값으로 대체하여 데이터 품질을 향상시켰습니다.
- **키워드 기반 관심사 탐지**: 원데이 클래스 주제에서 'Python', '머신러닝', 'SQL' 등 핵심 키워드를 추출하여 학습자의 기술적 관심도를 수치화했습니다.
- **Optuna 하이퍼파라미터 최적화**: 베이지안 최적화를 통해 도출된 최적 파라미터를 고정하여 **RandomForest** 모델의 성능을 극대화했습니다.

---

### **개발 환경**

- **운영체제**: macos 26.0
- **언어**: Python 3.11

---

### **프로젝트 구조**

```
Dacon_x_BDA_Learner_Completion_Prediction/
├── data/               
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── TF-IDF+RandomForest+Optuna.ipynb          
├── submission_tfidf_rf_fixed.csv
├── README.md           
└── requirements.txt    
```

---

### **모델링 접근법**

BDA 수강생의 중도 탈락을 예측하기 위해 자연어 처리(TF-IDF)와 RandomForest를 결합한 분류 모델을 구축했습니다.

***

### 1. 데이터 전처리 파이프라인
- **결측 과다 컬럼 삭제**: class2~4, previous_class_3~7 등 결측치가 많은 컬럼 제거
- **결측치 처리**: 텍스트형 변수는 'missing', 수치형 변수는 0으로 대체
- **이진 변수 처리**: contest_award, idea_contest, contest_participation을 0/1 이진화
- **자연어 처리**: whyBDA, what_to_gain, onedayclass_topic, expected_domain을 결합하여 TF-IDF 벡터화

### 2. 파생변수 생성
- **텍스트 길이**: 각종 텍스트 컬럼의 길이 계산
- **학교 빈도**: school1의 빈도수 기반 파생변수
- **키워드 탐지**: onedayclass_topic에서 'Python', '머신러닝', 'SQL' 키워드 포함 여부
- **전공 조합**: major_field와 major1_1의 조합 변수
- **희망 직무 일치**: desired_job과 desired_job_except_data 일치 여부

### 3. 모델 학습 전략
- **TF-IDF Vectorizer**: 자연어 데이터를 1000차원 벡터로 변환
- **RandomForest Classifier**: 테이블 데이터와 TF-IDF 벡터를 결합하여 학습
- **고정 하이퍼파라미터**: Optuna 최적화 결과를 바탕으로 고정된 최적 파라미터 사용

### 4. 최적 하이퍼파라미터
```python
best_params_fixed = {
    'n_estimators': 771,
    'max_depth': 25,
    'min_samples_split': 12,
    'min_samples_leaf': 20,
    'max_features': None,
    'bootstrap': True,
    'random_state': 42,
    'n_jobs': -1
}
```

---

### **파일 설명**

#### `TF-IDF+RandomForest+Optuna.ipynb`
- **데이터 로딩**: train.csv, test.csv, sample_submission.csv 불러오기
- **데이터 전처리**:
  - 결측 과다 컬럼 삭제 (class2~4, previous_class_3~7)
  - 결측치 처리 (텍스트: 'missing', 수치: 0)
  - 이진 변수 생성 (contest_award, idea_contest, contest_participation)
- **자연어 처리**:
  - 4개 텍스트 컬럼을 결합하여 통합 텍스트 생성
  - TF-IDF Vectorizer로 1000차원 벡터 변환
- **파생변수 생성**:
  - `re_reg_time_mult`: 시간과 재등록 상호작용
  - `school1_len`, `school1_freq`: 학교명 길이와 빈도
  - `onedayclass_topic_len`: 원데이 클래스 주제 길이
  - 키워드 탐지 변수들 (Python, 머신러닝, SQL)
  - `interested_company_len`, `interested_company_count`: 관심 회사 관련
  - `major_comb`: 전공 조합 변수
  - `desired_job_same`: 희망 직무 일치 여부
- **Label Encoding**: 모든 범주형 변수를 수치형으로 변환
- **모델 학습**: 
  - 테이블 데이터와 TF-IDF 벡터를 결합
  - 고정된 최적 하이퍼파라미터로 RandomForest 학습
  - Train/Validation 분할로 성능 검증
- **예측 및 저장**: `submission_tfidf_rf_fixed.csv` 생성

#### `data/`
- `train.csv`: BDA 수강생 학습 데이터 (ID, 개인정보, 수강동기 등, withdrawal 라벨)
- `test.csv`: BDA 수강생 테스트 데이터 (ID, 개인정보, 수강동기 등)
- `sample_submission.csv`: 제출 파일 템플릿 (ID, withdrawal)

#### `submission_tfidf_rf_fixed.csv`
- TF-IDF + RandomForest 모델의 최종 예측 결과 파일


