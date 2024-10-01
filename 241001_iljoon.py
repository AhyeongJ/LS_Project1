# 필요한 라이브러리 불러오기 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import shap
from sklearn.metrics import classification_report, confusion_matrix, roc_curve 
from imblearn.under_sampling import *
from imblearn.over_sampling import *

# 데이터 불러오기, 살펴보기
defect = pd.read_csv("../data/1주_실습데이터.csv")
defect.head()
defect.shape # (527000, 21)
defect.info()
defect.describe()
defect['Y'].sum()



### 데이터 분해
defect_X = defect.drop('Y', axis = 1)
defect_y = defect['Y']
defect_X_dropped = defect_X.drop(['X20', 'X18', 'X19', 'X4', 'X13'], axis= 1)



### 트레인, 테스트 나누기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(defect_X_dropped, defect_y, test_size=0.3, shuffle=True, stratify=defect_y, random_state=4)



### 랜덤 오버 셈플링
# Random Over Sampling
from imblearn.over_sampling import RandomOverSampler # 랜덤오버샘플링 해주는거.
from collections import Counter # 샘플링 갯수 결과 보기 위해 필요한 모듈

ros = RandomOverSampler(random_state=4)
X_train_ROS, y_train_ROS = ros.fit_resample(X_train, y_train)

# 결과 확인
print(f"Before oversampling: {Counter(y_train)}")
print(f"After oversampling: {Counter(y_train_ROS)}")



### Logistic Regression (로지스틱 회귀 분석)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# 로지스틱 회귀 모델 생성
model_LR = LogisticRegression()
model_LR.fit(X_train_ROS, y_train_ROS)

# 테스트 데이터에 대한 예측
y_pred_LR_ROS = model_LR.predict(X_test)



### 랜덤포레스트 분류 모델 생성
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# 랜덤포레스트 분류 모델 생성
model_RF = RandomForestClassifier(n_estimators=100, random_state=42)
model_RF.fit(X_train_ROS, y_train_ROS)

# 테스트 데이터에 대한 예측
y_pred_RF_ROS = model_RF.predict(X_test)



### XGBoost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# XGBoost 분류 모델 생성
model_XGB = XGBClassifier(n_estimators=100, random_state=4)
model_XGB.fit(X_train_ROS, y_train_ROS)

# 테스트 데이터에 대한 예측
y_pred_XGB_ROS = model_XGB.predict(X_test)



### Light GBM
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# LightGBM 분류 모델 생성
model_LGBM = LGBMClassifier(n_estimators=100, random_state=4)
model_LGBM.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
y_pred_LGBM_ROS = model_LGBM.predict(X_test)



### 모델평가
y_pred_LR_ROS
y_pred_RF_ROS
y_pred_XGB_ROS
y_pred_LGBM_ROS

# 한글 폰트 설정
plt.rcParams.update({"font.family" : "Malgun Gothic"})

# Confusion Matrix(혼동행렬)
cm = confusion_matrix(y_test, y_pred_LR_ROS)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for 랜덤 오버 셈플링*로지스틱 회귀 분석')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

cm = confusion_matrix(y_test, y_pred_RF_ROS)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for 랜덤 오버 셈플링*랜덤포레스트')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

cm = confusion_matrix(y_test, y_pred_XGB_ROS)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for 랜덤 오버 셈플링*XGBoost')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

cm = confusion_matrix(y_test, y_pred_LGBM_ROS)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for 랜덤 오버 셈플링*Light GBM')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


#f1_score
from sklearn.metrics import f1_score

# 데이터 스플릿으로 y_valid와 모델 예측으로 y_pred를 구한 후 실행
# y_pred만 모델에 맞게 수정
f1 = f1_score(y_test, y_pred_LR_ROS)
print(f"F1_score for 랜덤 오버 셈플링*로지스틱 회귀 분석: {round(f1,4)}")
f1 = f1_score(y_test, y_pred_RF_ROS)
print(f"F1_score for 랜덤 오버 셈플링*랜덤포레스트: {round(f1,4)}")
f1 = f1_score(y_test, y_pred_XGB_ROS)
print(f"F1_score for 랜덤 오버 셈플링*XGBoost: {round(f1,4)}")
f1 = f1_score(y_test, y_pred_LGBM_ROS)
print(f"F1_score for 랜덤 오버 셈플링*Light GBM: {round(f1,4)}")


# G-Mean
# y_pred만 모델에 맞게 수정
from imblearn.metrics import geometric_mean_score
g_mean = geometric_mean_score(y_test, y_pred_LR_ROS) # 예측값, 실제값
print(f"G-Mean for 랜덤 오버 셈플링*로지스틱 회귀 분석: {round(g_mean,4)}")
g_mean = geometric_mean_score(y_test, y_pred_RF_ROS)
print(f"G-Mean for 랜덤 오버 셈플링*랜덤포레스트: {round(g_mean,5)}")
g_mean = geometric_mean_score(y_test, y_pred_XGB_ROS)
print(f"G-Mean for 랜덤 오버 셈플링*XGBoost: {round(g_mean,5)}")
g_mean = geometric_mean_score(y_test, y_pred_LGBM_ROS)
print(f"G-Mean for 랜덤 오버 셈플링*Light GBM: {round(g_mean,4)}")

# 특성 중요도 시각화
feature_importance = model_XGB.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure(figsize=(12, 6))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X_train.columns[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Feature Importance in XGBoost')
plt.tight_layout()
plt.show()