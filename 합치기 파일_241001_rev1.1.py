# 필요한 라이브러리 불러오기 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap


# 데이터 불러오기
defect = pd.read_csv("../data/1주_실습데이터.csv")

# 데이터 살펴보기
defect.head()
defect.shape
defect.columns
defect.describe()
defect.isnull().sum().max() # null 값 확인
print('Defect',round(defect['Y'].value_counts()[1]/len(defect) * 100,2), '% of the dataset')



### EDA
# 타겟변수 시각화 
colors = ["#ABC2E2", "#CBA5D1"]
sns.countplot(x = 'Y', data = defect, palette = colors)

# KDE plot을 통해 각 변수의 분포 시각화
plt.figure(figsize=(15, 12))
for i, column in enumerate(defect.iloc[:, :-1]):  # 마지막 열 Y는 제외하고 X 변수만 사용
    plt.subplot(5, 4, i+1)
    sns.kdeplot(defect[column], fill=True)
    plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.show()



### 데이터 나누기 ###
defect_X = defect.drop('Y', axis = 1)
defect_y = defect['Y']



# 독립변수 간 상관관계 알아보기 
defect_X.corr()

# 독립변수 히트맵
plt.figure(figsize=(10,10))
sns.heatmap(defect_X.corr(),annot=True,fmt=".2f")


# X6와 X20, X8과 X18, X12와 X19 컬럼들이 동일한 지 알아보기
print((defect_X["X6"]-defect_X["X20"]).unique())
print((defect_X["X8"]-defect_X["X18"]).unique())
print((defect_X["X12"]-defect_X["X19"]).unique())

# X4, X13의 값 알아보기
print(defect_X['X4'].unique())
print(defect_X['X13'].unique())



### 컬럼 드롭 ###
defect_X_dropped = defect_X.drop(['X20', 'X18', 'X19', 'X4', 'X13'], axis= 1)



# 다중공선성 확인해보기
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(
    defect_X_dropped.values, i) for i in range(defect_X_dropped.shape[1])]
vif["features"] = defect_X_dropped.columns
vif



### 트레인, 테스트 나누기 ###
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(defect_X_dropped, defect_y, test_size=0.3, shuffle=True, stratify=defect_y, random_state=4)


##################################
##### 기본샘플 + 샘플링 5종류 #####
from collections import Counter # 샘플링 갯수 결과 보기 위해 필요한 모듈
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

### 0. 기본 데이터
# Original Data
X_train
y_train

## Logistic Regression (로지스틱 회귀 분석)
from sklearn.linear_model import LogisticRegression
model_Original_LogisticRegression = LogisticRegression()
model_Original_LogisticRegression.fit(X_train, y_train)

y_pred_Original_LogisticRegression = model_Original_LogisticRegression.predict(X_test)

## Random Forest (랜덤포레스트)
from sklearn.ensemble import RandomForestClassifier
model_Original_RandomForest = RandomForestClassifier(n_estimators=100, random_state=4)
model_Original_RandomForest.fit(X_train, y_train)

y_pred_Original_RandomForest = model_Original_RandomForest.predict(X_test)

## XGBoost
from xgboost import XGBClassifier
model_Original_XGBoost = XGBClassifier(n_estimators=100, random_state=4)
model_Original_XGBoost.fit(X_train, y_train)

y_pred_Original_XGBoost = model_Original_XGBoost.predict(X_test)

### Light GBM
from lightgbm import LGBMClassifier
model_Original_LightGBM = LGBMClassifier(n_estimators=100, random_state=4)
model_Original_LightGBM.fit(X_train, y_train)

y_pred_Original_LightGBM = model_Original_LightGBM.predict(X_test)



### 1. 랜덤 언더 샘플링
# Random Under Sampling
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=4)
X_train_RandomUnderSample, y_train_RandomUnderSample = rus.fit_resample(X_train, y_train)

print(f"Before Sampling: {Counter(y_train)}") # 결과 확인
print(f"After Sampling, Random Under Sampling: {Counter(y_train_RandomUnderSample)}")
# {0: 39900, 1: 39900}

## Logistic Regression (로지스틱 회귀 분석)
from sklearn.linear_model import LogisticRegression
model_Under_LogisticRegression = LogisticRegression()
model_Under_LogisticRegression.fit(X_train_RandomUnderSample, y_train_RandomUnderSample)

y_pred_Under_LogisticRegression = model_Under_LogisticRegression.predict(X_test)

## Random Forest (랜덤포레스트)
from sklearn.ensemble import RandomForestClassifier
model_Under_RandomForest = RandomForestClassifier(n_estimators=100, random_state=4)
model_Under_RandomForest.fit(X_train_RandomUnderSample, y_train_RandomUnderSample)

y_pred_Under_RandomForest = model_Under_RandomForest.predict(X_test)

## XGBoost
from xgboost import XGBClassifier
model_Under_XGBoost = XGBClassifier(n_estimators=100, random_state=4)
model_Under_XGBoost.fit(X_train_RandomUnderSample, y_train_RandomUnderSample)

y_pred_Under_XGBoost = model_Under_XGBoost.predict(X_test)

### Light GBM
from lightgbm import LGBMClassifier
model_Under_LightGBM = LGBMClassifier(n_estimators=100, random_state=4)
model_Under_LightGBM.fit(X_train_RandomUnderSample, y_train_RandomUnderSample)

y_pred_Under_LightGBM = model_Under_LightGBM.predict(X_test)



### 2. 토멕링크
# Tomek Links
from imblearn.under_sampling import TomekLinks
tl = TomekLinks(sampling_strategy='majority')
X_train_TomekLinks, y_train_TomekLinks = tl.fit_resample(X_train,y_train)

print(f"Before Sampling: {Counter(y_train)}")
print(f"After Sampling, Tomek Links: {Counter(y_train_TomekLinks)}")
# {0: 328971, 1: 39900}


## Logistic Regression (로지스틱 회귀 분석)
from sklearn.linear_model import LogisticRegression
model_Tomek_LogisticRegression = LogisticRegression()
model_Tomek_LogisticRegression.fit(X_train_TomekLinks, y_train_TomekLinks)

y_pred_Tomek_LogisticRegression = model_Tomek_LogisticRegression.predict(X_test)

## Random Forest (랜덤포레스트)
from sklearn.ensemble import RandomForestClassifier
model_Tomek_RandomForest = RandomForestClassifier(n_estimators=100, random_state=4)
model_Tomek_RandomForest.fit(X_train_TomekLinks, y_train_TomekLinks)

y_pred_Tomek_RandomForest = model_Tomek_RandomForest.predict(X_test)

## XGBoost
from xgboost import XGBClassifier
model_Tomek_XGBoost = XGBClassifier(n_estimators=100, random_state=4)
model_Tomek_XGBoost.fit(X_train_TomekLinks, y_train_TomekLinks)

y_pred_Tomek_XGBoost = model_Tomek_XGBoost.predict(X_test)

### Light GBM
from lightgbm import LGBMClassifier
model_Tomek_LightGBM = LGBMClassifier(n_estimators=100, random_state=4)
model_Tomek_LightGBM.fit(X_train_TomekLinks, y_train_TomekLinks)

y_pred_Tomek_LightGBM = model_Tomek_LightGBM.predict(X_test)



### 3. 랜덤 오버 샘플링
# Random Over Sampling
from imblearn.over_sampling import RandomOverSampler # 랜덤오버샘플링 해주는거.
ros = RandomOverSampler(random_state=4)
X_train_RandomOverSample, y_train_RandomOverSample = ros.fit_resample(X_train, y_train)

print(f"Before Sampling: {Counter(y_train)}")
print(f"After Sampling, Random Over Sampling: {Counter(y_train_RandomOverSample)}")
# {0: 329000, 1: 329000}


## Logistic Regression (로지스틱 회귀 분석)
from sklearn.linear_model import LogisticRegression
model_Over_LogisticRegression = LogisticRegression()
model_Over_LogisticRegression.fit(X_train_RandomOverSample, y_train_RandomOverSample)

y_pred_Over_LogisticRegression = model_Over_LogisticRegression.predict(X_test)

## Random Forest (랜덤포레스트)
from sklearn.ensemble import RandomForestClassifier
model_Over_RandomForest = RandomForestClassifier(n_estimators=100, random_state=4)
model_Over_RandomForest.fit(X_train_RandomOverSample, y_train_RandomOverSample)

y_pred_Over_RandomForest = model_Over_RandomForest.predict(X_test)

## XGBoost
from xgboost import XGBClassifier
model_Over_XGBoost = XGBClassifier(n_estimators=100, random_state=4)
model_Over_XGBoost.fit(X_train_RandomOverSample, y_train_RandomOverSample)

y_pred_Over_XGBoost = model_Over_XGBoost.predict(X_test)

### Light GBM
from lightgbm import LGBMClassifier
model_Over_LightGBM = LGBMClassifier(n_estimators=100, random_state=4)
model_Over_LightGBM.fit(X_train_RandomOverSample, y_train_RandomOverSample)

y_pred_Over_LightGBM = model_Over_LightGBM.predict(X_test)



### 4. 스모트
# SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state = 4)
X_train_SMOTE, y_train_SMOTE = smote.fit_resample(X_train, y_train)

print(f"Before Sampling: {Counter(y_train)}")
print(f"After Sampling, SMOTE: {Counter(y_train_SMOTE)}")
# {0: 329000, 1: 329000}


## Logistic Regression (로지스틱 회귀 분석)
from sklearn.linear_model import LogisticRegression
model_SMOTE_LogisticRegression = LogisticRegression()
model_SMOTE_LogisticRegression.fit(X_train_SMOTE, y_train_SMOTE)

y_pred_SMOTE_LogisticRegression = model_SMOTE_LogisticRegression.predict(X_test)

## Random Forest (랜덤포레스트)
from sklearn.ensemble import RandomForestClassifier
model_SMOTE_RandomForest = RandomForestClassifier(n_estimators=100, random_state=4)
model_SMOTE_RandomForest.fit(X_train_SMOTE, y_train_SMOTE)

y_pred_SMOTE_RandomForest = model_SMOTE_RandomForest.predict(X_test)

## XGBoost
from xgboost import XGBClassifier
model_SMOTE_XGBoost = XGBClassifier(n_estimators=100, random_state=4)
model_SMOTE_XGBoost.fit(X_train_SMOTE, y_train_SMOTE)

y_pred_SMOTE_XGBoost = model_SMOTE_XGBoost.predict(X_test)

### Light GBM
from lightgbm import LGBMClassifier
model_SMOTE_LightGBM = LGBMClassifier(n_estimators=100, random_state=4)
model_SMOTE_LightGBM.fit(X_train_SMOTE, y_train_SMOTE)

y_pred_SMOTE_LightGBM = model_SMOTE_LightGBM.predict(X_test)



### 5. 스모트 토멕
# SMOTE TOMEK
from imblearn.combine import SMOTETomek
smote_tomek = SMOTETomek(random_state=4)
X_train_SmoteTomek, y_train_SmoteTomek = smote_tomek.fit_resample(X_train, y_train)

print(f"Before Sampling: {Counter(y_train)}")
print(f"After Sampling, SMOTE TOMEK: {Counter(y_train_SmoteTomek)}")
# {0: 328953, 1: 328953}


## Logistic Regression (로지스틱 회귀 분석)
from sklearn.linear_model import LogisticRegression
model_SmoteTomek_LogisticRegression = LogisticRegression()
model_SmoteTomek_LogisticRegression.fit(X_train_SmoteTomek, y_train_SmoteTomek)

y_pred_SmoteTomek_LogisticRegression = model_SmoteTomek_LogisticRegression.predict(X_test)

## Random Forest (랜덤포레스트)
from sklearn.ensemble import RandomForestClassifier
model_SmoteTomek_RandomForest = RandomForestClassifier(n_estimators=100, random_state=4)
model_SmoteTomek_RandomForest.fit(X_train_SmoteTomek, y_train_SmoteTomek)

y_pred_SmoteTomek_RandomForest = model_SmoteTomek_RandomForest.predict(X_test)

## XGBoost
from xgboost import XGBClassifier
model_SmoteTomek_XGBoost = XGBClassifier(n_estimators=100, random_state=4)
model_SmoteTomek_XGBoost.fit(X_train_SmoteTomek, y_train_SmoteTomek)

y_pred_SmoteTomek_XGBoost = model_SmoteTomek_XGBoost.predict(X_test)

### Light GBM
from lightgbm import LGBMClassifier
model_SmoteTomek_LightGBM = LGBMClassifier(n_estimators=100, random_state=4)
model_SmoteTomek_LightGBM.fit(X_train_SmoteTomek, y_train_SmoteTomek)

y_pred_SmoteTomek_LightGBM = model_SmoteTomek_LightGBM.predict(X_test)
############ 샘플링, 모델링 끝 #######################


######################################
########### 성능 비교 #################

### F1-score
from sklearn.metrics import f1_score

f1_Original_LogisticRegression = f1_score(y_test, y_pred_Original_LogisticRegression)
f1_Original_RandomForest = f1_score(y_test, y_pred_Original_RandomForest)
f1_Original_XGBoost = f1_score(y_test, y_pred_Original_XGBoost)
f1_Original_LightGBM = f1_score(y_test, y_pred_Original_LightGBM)

f1_Under_LogisticRegression = f1_score(y_test, y_pred_Under_LogisticRegression)
f1_Under_RandomForest = f1_score(y_test, y_pred_Under_RandomForest)
f1_Under_XGBoost = f1_score(y_test, y_pred_Under_XGBoost)
f1_Under_LightGBM = f1_score(y_test, y_pred_Under_LightGBM)

f1_Tomek_LogisticRegression = f1_score(y_test, y_pred_Tomek_LogisticRegression)
f1_Tomek_RandomForest = f1_score(y_test, y_pred_Tomek_RandomForest)
f1_Tomek_XGBoost = f1_score(y_test, y_pred_Tomek_XGBoost)
f1_Tomek_LightGBM = f1_score(y_test, y_pred_Tomek_LightGBM)

f1_Over_LogisticRegression = f1_score(y_test, y_pred_Over_LogisticRegression)
f1_Over_RandomForest = f1_score(y_test, y_pred_Over_RandomForest)
f1_Over_XGBoost = f1_score(y_test, y_pred_Over_XGBoost)
f1_Over_LightGBM = f1_score(y_test, y_pred_Over_LightGBM)

f1_SMOTE_LogisticRegression = f1_score(y_test, y_pred_SMOTE_LogisticRegression)
f1_SMOTE_RandomForest = f1_score(y_test, y_pred_SMOTE_RandomForest)
f1_SMOTE_XGBoost = f1_score(y_test, y_pred_SMOTE_XGBoost)
f1_SMOTE_LightGBM = f1_score(y_test, y_pred_SMOTE_LightGBM)

f1_SmoteTomek_LogisticRegression = f1_score(y_test, y_pred_SmoteTomek_LogisticRegression)
f1_SmoteTomek_RandomForest = f1_score(y_test, y_pred_SmoteTomek_RandomForest)
f1_SmoteTomek_XGBoost = f1_score(y_test, y_pred_SmoteTomek_XGBoost)
f1_SmoteTomek_LightGBM = f1_score(y_test, y_pred_SmoteTomek_LightGBM)

f1score_table = pd.DataFrame({
    'sampling' : ['Original', 'Under', 'Tomek', 'Over', 'SMOTE', 'SmoteTomek'],
    'Logistic Regression' : [f1_Original_LogisticRegression, f1_Under_LogisticRegression, f1_Tomek_LogisticRegression, f1_Over_LogisticRegression, f1_SMOTE_LogisticRegression, f1_SmoteTomek_LogisticRegression],
    'Random Forest' : [f1_Original_RandomForest, f1_Under_RandomForest, f1_Tomek_RandomForest, f1_Over_RandomForest, f1_SMOTE_RandomForest, f1_SmoteTomek_RandomForest],
    'XGBoost' : [f1_Original_XGBoost, f1_Under_XGBoost, f1_Tomek_XGBoost, f1_Over_XGBoost, f1_SMOTE_XGBoost, f1_SmoteTomek_XGBoost],
    'Light GBM' : [f1_Original_LightGBM, f1_Under_LightGBM, f1_Tomek_LightGBM, f1_Over_LightGBM, f1_SMOTE_LightGBM, f1_SmoteTomek_LightGBM]
})
f1score_table

### G-Mean
from imblearn.metrics import geometric_mean_score

gmean_Original_LogisticRegression = geometric_mean_score(y_test, y_pred_Original_LogisticRegression)
gmean_Original_RandomForest = geometric_mean_score(y_test, y_pred_Original_RandomForest)
gmean_Original_XGBoost = geometric_mean_score(y_test, y_pred_Original_XGBoost)
gmean_Original_LightGBM = geometric_mean_score(y_test, y_pred_Original_LightGBM)

gmean_Under_LogisticRegression = geometric_mean_score(y_test, y_pred_Under_LogisticRegression)
gmean_Under_RandomForest = geometric_mean_score(y_test, y_pred_Under_RandomForest)
gmean_Under_XGBoost = geometric_mean_score(y_test, y_pred_Under_XGBoost)
gmean_Under_LightGBM = geometric_mean_score(y_test, y_pred_Under_LightGBM)

gmean_Tomek_LogisticRegression = geometric_mean_score(y_test, y_pred_Tomek_LogisticRegression)
gmean_Tomek_RandomForest = geometric_mean_score(y_test, y_pred_Tomek_RandomForest)
gmean_Tomek_XGBoost = geometric_mean_score(y_test, y_pred_Tomek_XGBoost)
gmean_Tomek_LightGBM = geometric_mean_score(y_test, y_pred_Tomek_LightGBM)

gmean_Over_LogisticRegression = geometric_mean_score(y_test, y_pred_Over_LogisticRegression)
gmean_Over_RandomForest = geometric_mean_score(y_test, y_pred_Over_RandomForest)
gmean_Over_XGBoost = geometric_mean_score(y_test, y_pred_Over_XGBoost)
gmean_Over_LightGBM = geometric_mean_score(y_test, y_pred_Over_LightGBM)

gmean_SMOTE_LogisticRegression = geometric_mean_score(y_test, y_pred_SMOTE_LogisticRegression)
gmean_SMOTE_RandomForest = geometric_mean_score(y_test, y_pred_SMOTE_RandomForest)
gmean_SMOTE_XGBoost = geometric_mean_score(y_test, y_pred_SMOTE_XGBoost)
gmean_SMOTE_LightGBM = geometric_mean_score(y_test, y_pred_SMOTE_LightGBM)

gmean_SmoteTomek_LogisticRegression = geometric_mean_score(y_test, y_pred_SmoteTomek_LogisticRegression)
gmean_SmoteTomek_RandomForest = geometric_mean_score(y_test, y_pred_SmoteTomek_RandomForest)
gmean_SmoteTomek_XGBoost = geometric_mean_score(y_test, y_pred_SmoteTomek_XGBoost)
gmean_SmoteTomek_LightGBM = geometric_mean_score(y_test, y_pred_SmoteTomek_LightGBM)

gmean_table = pd.DataFrame({
    'sampling' : ['Original', 'Under', 'Tomek', 'Over', 'SMOTE', 'SmoteTomek'],
    'Logistic Regression' : [gmean_Original_LogisticRegression, gmean_Under_LogisticRegression, gmean_Tomek_LogisticRegression, gmean_Over_LogisticRegression, gmean_SMOTE_LogisticRegression, gmean_SmoteTomek_LogisticRegression],
    'Random Forest' : [gmean_Original_RandomForest, gmean_Under_RandomForest, gmean_Tomek_RandomForest, gmean_Over_RandomForest, gmean_SMOTE_RandomForest, gmean_SmoteTomek_RandomForest],
    'XGBoost' : [gmean_Original_XGBoost, gmean_Under_XGBoost, gmean_Tomek_XGBoost, gmean_Over_XGBoost, gmean_SMOTE_XGBoost, gmean_SmoteTomek_XGBoost],
    'Light GBM' : [gmean_Original_LightGBM, gmean_Under_LightGBM, gmean_Tomek_LightGBM, gmean_Over_LightGBM, gmean_SMOTE_LightGBM, gmean_SmoteTomek_LightGBM]
})
gmean_table

### FN 개수
# FN = 실제는 Positive인데 예측을 Nagative로 분류 한 것 = 정답은 1인데 0으로 분류 한 것 = 불량인데 양품이라고 분류한 것
from sklearn.metrics import confusion_matrix
'''
컨퓨젼 메트릭스가 나타내는 값
[[TN, FP],
 [FN, TP]]
'''

fn_Original_LogisticRegression = confusion_matrix(y_test, y_pred_Original_LogisticRegression)[1, 0]
fn_Original_RandomForest = confusion_matrix(y_test, y_pred_Original_RandomForest)[1, 0]
fn_Original_XGBoost = confusion_matrix(y_test, y_pred_Original_XGBoost)[1, 0]
fn_Original_LightGBM = confusion_matrix(y_test, y_pred_Original_LightGBM)[1, 0]

fn_Under_LogisticRegression = confusion_matrix(y_test, y_pred_Under_LogisticRegression)[1, 0]
fn_Under_RandomForest = confusion_matrix(y_test, y_pred_Under_RandomForest)[1, 0]
fn_Under_XGBoost = confusion_matrix(y_test, y_pred_Under_XGBoost)[1, 0]
fn_Under_LightGBM = confusion_matrix(y_test, y_pred_Under_LightGBM)[1, 0]

fn_Tomek_LogisticRegression = confusion_matrix(y_test, y_pred_Tomek_LogisticRegression)[1, 0]
fn_Tomek_RandomForest = confusion_matrix(y_test, y_pred_Tomek_RandomForest)[1, 0]
fn_Tomek_XGBoost = confusion_matrix(y_test, y_pred_Tomek_XGBoost)[1, 0]
fn_Tomek_LightGBM = confusion_matrix(y_test, y_pred_Tomek_LightGBM)[1, 0]

fn_Over_LogisticRegression = confusion_matrix(y_test, y_pred_Over_LogisticRegression)[1, 0]
fn_Over_RandomForest = confusion_matrix(y_test, y_pred_Over_RandomForest)[1, 0]
fn_Over_XGBoost = confusion_matrix(y_test, y_pred_Over_XGBoost)[1, 0]
fn_Over_LightGBM = confusion_matrix(y_test, y_pred_Over_LightGBM)[1, 0]

fn_SMOTE_LogisticRegression = confusion_matrix(y_test, y_pred_SMOTE_LogisticRegression)[1, 0]
fn_SMOTE_RandomForest = confusion_matrix(y_test, y_pred_SMOTE_RandomForest)[1, 0]
fn_SMOTE_XGBoost = confusion_matrix(y_test, y_pred_SMOTE_XGBoost)[1, 0]
fn_SMOTE_LightGBM = confusion_matrix(y_test, y_pred_SMOTE_LightGBM)[1, 0]

fn_SmoteTomek_LogisticRegression = confusion_matrix(y_test, y_pred_SmoteTomek_LogisticRegression)[1, 0]
fn_SmoteTomek_RandomForest = confusion_matrix(y_test, y_pred_SmoteTomek_RandomForest)[1, 0]
fn_SmoteTomek_XGBoost = confusion_matrix(y_test, y_pred_SmoteTomek_XGBoost)[1, 0]
fn_SmoteTomek_LightGBM = confusion_matrix(y_test, y_pred_SmoteTomek_LightGBM)[1, 0]

fn_table = pd.DataFrame({
    'sampling' : ['Original', 'Under', 'Tomek', 'Over', 'SMOTE', 'SmoteTomek'],
    'Logistic Regression' : [fn_Original_LogisticRegression, fn_Under_LogisticRegression, fn_Tomek_LogisticRegression, fn_Over_LogisticRegression, fn_SMOTE_LogisticRegression, fn_SmoteTomek_LogisticRegression],
    'Random Forest' : [fn_Original_RandomForest, fn_Under_RandomForest, fn_Tomek_RandomForest, fn_Over_RandomForest, fn_SMOTE_RandomForest, fn_SmoteTomek_RandomForest],
    'XGBoost' : [fn_Original_XGBoost, fn_Under_XGBoost, fn_Tomek_XGBoost, fn_Over_XGBoost, fn_SMOTE_XGBoost, fn_SmoteTomek_XGBoost],
    'Light GBM' : [fn_Original_LightGBM, fn_Under_LightGBM, fn_Tomek_LightGBM, fn_Over_LightGBM, fn_SMOTE_LightGBM, fn_SmoteTomek_LightGBM]
})
fn_table

### 샘플링 성능지표 만들기
sampling_score_Original = pd.DataFrame({
    'model' : ['Logistic Regression', 'Random Forest', 'XGBoost', 'Light GBM'],
    'F1-score' : [f1_Original_LogisticRegression, f1_Original_RandomForest, f1_Original_XGBoost, f1_Original_LightGBM],
    'G-means' : [gmean_Original_LogisticRegression, gmean_Original_RandomForest, gmean_Original_XGBoost, gmean_Original_LightGBM],
    'FN count' : [fn_Original_LogisticRegression, fn_Original_RandomForest, fn_Original_XGBoost, fn_Original_LightGBM]
})

sampling_score_RandomUnder = pd.DataFrame({
    'model' : ['Logistic Regression', 'Random Forest', 'XGBoost', 'Light GBM'],
    'F1-score' : [f1_Under_LogisticRegression, f1_Under_RandomForest, f1_Under_XGBoost, f1_Under_LightGBM],
    'G-means' : [gmean_Under_LogisticRegression, gmean_Under_RandomForest, gmean_Under_XGBoost, gmean_Under_LightGBM],
    'FN count' : [fn_Under_LogisticRegression, fn_Under_RandomForest, fn_Under_XGBoost, fn_Under_LightGBM]
})

sampling_score_TomekLinks = pd.DataFrame({
    'model' : ['Logistic Regression', 'Random Forest', 'XGBoost', 'Light GBM'],
    'F1-score' : [f1_Tomek_LogisticRegression, f1_Tomek_RandomForest, f1_Tomek_XGBoost, f1_Tomek_LightGBM],
    'G-means' : [gmean_Tomek_LogisticRegression, gmean_Tomek_RandomForest, gmean_Tomek_XGBoost, gmean_Tomek_LightGBM],
    'FN count' : [fn_Tomek_LogisticRegression, fn_Tomek_RandomForest, fn_Tomek_XGBoost, fn_Tomek_LightGBM]
})

sampling_score_RandomOver = pd.DataFrame({
    'model' : ['Logistic Regression', 'Random Forest', 'XGBoost', 'Light GBM'],
    'F1-score' : [f1_Over_LogisticRegression, f1_Over_RandomForest, f1_Over_XGBoost, f1_Over_LightGBM],
    'G-means' : [gmean_Over_LogisticRegression, gmean_Over_RandomForest, gmean_Over_XGBoost, gmean_Over_LightGBM],
    'FN count' : [fn_Over_LogisticRegression, fn_Over_RandomForest, fn_Over_XGBoost, fn_Over_LightGBM]
})

sampling_score_SMOTE = pd.DataFrame({
    'model' : ['Logistic Regression', 'Random Forest', 'XGBoost', 'Light GBM'],
    'F1-score' : [f1_SMOTE_LogisticRegression, f1_SMOTE_RandomForest, f1_SMOTE_XGBoost, f1_SMOTE_LightGBM],
    'G-means' : [gmean_SMOTE_LogisticRegression, gmean_SMOTE_RandomForest, gmean_SMOTE_XGBoost, gmean_SMOTE_LightGBM],
    'FN count' : [fn_SMOTE_LogisticRegression, fn_SMOTE_RandomForest, fn_SMOTE_XGBoost, fn_SMOTE_LightGBM]
})

sampling_score_SmoteTomek = pd.DataFrame({
    'model' : ['Logistic Regression', 'Random Forest', 'XGBoost', 'Light GBM'],
    'F1-score' : [f1_SmoteTomek_LogisticRegression, f1_SmoteTomek_RandomForest, f1_SmoteTomek_XGBoost, f1_SmoteTomek_LightGBM],
    'G-means' : [gmean_SmoteTomek_LogisticRegression, gmean_SmoteTomek_RandomForest, gmean_SmoteTomek_XGBoost, gmean_SmoteTomek_LightGBM],
    'FN count' : [fn_SmoteTomek_LogisticRegression, fn_SmoteTomek_RandomForest, fn_SmoteTomek_XGBoost, fn_SmoteTomek_LightGBM]
})

sampling_score_Original
sampling_score_RandomUnder
sampling_score_TomekLinks
sampling_score_RandomOver
sampling_score_SMOTE
sampling_score_SmoteTomek

# F1스코어가 가장 높은 녀석은? SmoteTomek * XGBoost
# G-means가 가장 높은 녀석은? SMOTE * XGBoost
# FN 갯수가 가장 낮은 녀석은? SMOTE * XGBoost
# 가장 괜찮은 셈플링, 모델 조합은 SMOTE * XGBoost이다.
# model_SMOTE_XGBoost
# 이 조합일 때 중요한 변수는 X3였다.

