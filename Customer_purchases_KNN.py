import pandas as pd
import numpy as np
import time #計算執行時間
from sklearn.preprocessing import StandardScaler #資料標準化
from sklearn.model_selection import train_test_split, cross_val_score #資料分割, 交叉驗證
from sklearn.ensemble import AdaBoostClassifier #Adaptive Boost
from sklearn.metrics import accuracy_score #準確率
# from sklearn.preprocessing import LabelEncoder 將目標變量進行整數編碼
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout #Dense, Dropout
from tensorflow.keras.optimizers import Adam #Adam 演算法
from sklearn.model_selection import StratifiedKFold #交叉驗證切割資料
from sklearn.neighbors import KNeighborsClassifier #KNN
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

#資料載入
data=pd.read_csv('Customer_purchases.csv')

#資料前處理
print(data.isnull().sum())
data= data.drop(["id"], axis=1)

#特徵工程
#Frequency encoding: 地區, 產品種類
frequency_region=data["region"].value_counts()
data["region"]=data["region"].map(frequency_region)
frequency_product_category=data["product_category"].value_counts()
data["product_category"]=data["product_category"].map(frequency_product_category)
#Label encoder and Order encoder：教育程度, 購物頻率, 性別
sorted_categories= {'gender':{'Female':0, 'Male':1},
                    'education':{'HighSchool':0, 'College':1, 'Bachelor':2,
                    'Masters':3},
                    'loyalty_status':{'Regular':0, 'Silver':1, 'Gold':2},
                    'purchase_frequency':{'rare':0, 'occasional':1, 'frequent':2}} #定義每個欄位的類別順序
for col in sorted_categories.keys():
 data[col]= data[col].map(sorted_categories[col]) #欄位.map(dict)：根據 dict 轉換資料


#分成解釋變數(X)和目標變數(Y)
X= data.drop(["loyalty_status"], axis=1)
Y= data['loyalty_status']
columns_name=X.columns

#將數據集分為訓練集和測試集
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size= 0.1, random_state= 202)

#RFE特徵選取
model=LogisticRegression()
rfe = RFE(model, n_features_to_select=2)
rfe = rfe.fit(X_train, Y_train)
#選取特徵
selected_features = X_train.columns[rfe.support_]
print(selected_features)
X_train=X_train[selected_features]
X_test=X_test[selected_features]

#X 標準化
scaler1= StandardScaler().fit(X_train)
X_train= scaler1.transform(X_train)
scaler2= StandardScaler().fit(X_test)
X_test= scaler2.transform(X_test)

# 參數設定
num_neighbors= np.arange(1, 11, 1)


class3_fp_insample_vector = []
class3_fp_cv_vector = []
knn_insample_recall_vector = []
knn_random_recall = []
# 定義召回率比重
class_weights = np.array([1, 2, 3])


for num_neighbor in num_neighbors:
    knn_classifier = KNeighborsClassifier(n_neighbors=num_neighbor)
    knn_classifier.fit(X_train, Y_train)

    # 訓練集上的預測結果
    knn_insample_predict = knn_classifier.predict(X_train)
    # 訓練集召回率（加權平均）
    recall_per_class = recall_score(Y_train, knn_insample_predict, average=None)
    weighted_recall_insample = np.dot(recall_per_class, class_weights) / class_weights.sum()
    knn_insample_recall_vector.append(round(weighted_recall_insample, 6))
    # 計算混淆矩陣並擷取類別3的假陽個數
    cm = confusion_matrix(Y_train, knn_insample_predict)
    class3_fp_insample = cm[0:2, 2].sum()  # 類別3的假陽個數是第一、二行的第三列元素和
    class3_fp_insample_vector.append(class3_fp_insample)

    # 交叉驗證預測
    y_pred_cv = cross_val_predict(knn_classifier, X_train, Y_train, cv=10)
    # 交叉驗證召回率（加權平均）
    recall_per_class_cv = recall_score(Y_train, y_pred_cv, average=None)
    weighted_recall_cv = np.dot(recall_per_class_cv, class_weights) / class_weights.sum()
    knn_random_recall.append(round(weighted_recall_cv, 6))
    # 計算交叉驗證中的類別3假陽個數
    cm_cv = confusion_matrix(Y_train, y_pred_cv)
    class3_fp_cv = cm_cv[0:2, 2].sum()  # 同樣計算類別3假陽個數
    class3_fp_cv_vector.append(class3_fp_cv)


# 打印结果
for class3_fp_insample, class3_fp_cv, apparent_recall, CV_recall in zip(
    class3_fp_insample_vector, class3_fp_cv_vector,
    knn_insample_recall_vector, knn_random_recall):
    print(f'Class 3 FP - Apparent: {class3_fp_insample}, CV: {class3_fp_cv}, Recall - Apparent: {apparent_recall}, CV: {CV_recall}')

# 繪製圖形：類別3假陽個數與加權召回率
plt.figure(figsize=(10, 8))

# 繪製類別3假陽個數
plt.scatter(num_neighbors, num_neighbors, color='blue', label='Data Points')
for i, (class3_fp_insample, class3_fp_cv) in enumerate(zip(class3_fp_insample_vector, class3_fp_cv_vector)):
    plt.text(num_neighbors[i], num_neighbors[i], f'Class 3 FP({class3_fp_insample}, {class3_fp_cv})', fontsize=12, ha='center', va='bottom', color='blue')

# 繪製加權召回率
for i, (apparent_recall, CV_recall) in enumerate(zip(knn_insample_recall_vector, knn_random_recall)):
    plt.text(num_neighbors[i], num_neighbors[i], f'Recall({apparent_recall}, {CV_recall})', fontsize=12, ha='center', va='top', color='green')

plt.title('Class 3 False Positives and Weighted Recall for KNN', fontsize=24)
plt.xlabel('Number of Neighbors', fontsize=24)
plt.ylabel('', fontsize=24)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.show()


