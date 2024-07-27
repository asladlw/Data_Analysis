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
import xgboost as xgb #XGboost
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#資料載入
data=pd.read_csv('Customer_purchases.csv')

#資料前處理
 #無遺失值
print(data.isnull().sum())

 #刪除欄位"id"
data= data.drop(["id"], axis=1)

 #one-hot encoder
oh_encoder= pd.get_dummies(data[['region','product_category']]) #使用get_dummies函數進行one-hot encoding
data= pd.concat([data.drop(['region', 'product_category'], axis=1), oh_encoder], axis=1) #使用 concat 函數按列合併 DataFrame

 #Label encoder and Order encoder
sorted_categories= {'gender':{'Female':0, 'Male':1},
                    'education':{'HighSchool':0, 'College':1, 'Bachelor':2, 'Masters':3},
                    'loyalty_status':{'Regular':0, 'Silver':1, 'Gold':2},
                    'purchase_frequency':{'rare':0, 'occasional':1, 'frequent':2}} #定義每個欄位的類別順序
 
for col in sorted_categories.keys():
    data[col]= data[col].map(sorted_categories[col]) #欄位.map(dict)：根據 dict 轉換資料

#模型訓練
 #參數設定
num_grids=25

  #AdaBoost
# adaboost_n_iterations= np.random.randint(50, 500, num_grids)
# adaboost_learning_rate= 0.5 * np.random.rand(num_grids)
# adaboost_paras= zip(adaboost_n_iterations, adaboost_learning_rate)

  #XGBoost
xgboost_learning_rate= np.linspace(0.01, 0.1, num_grids)
xgboost_max_depth= np.random.randint(3, 6, num_grids)
xgboost_paras= zip(xgboost_learning_rate, xgboost_max_depth)

  #DNN
batch_size= np.random.choice([64, 128, 256, 512], num_grids)
DNN_learning_rate= np.random.rand(num_grids)
DNN_paras=zip(batch_size, DNN_learning_rate)

  #K Nearest Neighbor
num_neighbors= np.arange(10, 161, 10)

 #分成解釋變數(X)和目標變數(Y)
X= data.drop(["loyalty_status"], axis=1)
Y= data['loyalty_status']

 #X 標準化
scaler= StandardScaler().fit(X)
X= scaler.transform(X)
Y=np.array(Y)

#  #將數據集分為訓練集和測試集
# X_train, X_test, Y_train, Y_test= train_test_split(std_X, Y, test_size= 0.2, 
# random_state= 202)

 #開始訓練(取得訓練資料的 CV 準確度)
#   #AdaBoost
# adaboost_random_accuracy= []
# for itera, lr in adaboost_paras:
#  adaboost_classifier= AdaBoostClassifier(n_estimators= itera, learning_rate= lr)
#  # adaboost_classifier.fit(X, Y)
#  cv_scores= cross_val_score(adaboost_classifier,X, Y, cv=10)
#  adaboost_random_accuracy.append(round(cv_scores.mean(), 6)) #.append()只會作用不會回傳

  #XGBoost
# dtrain= xgb.DMatrix(X_train, label= Y_train)
xgboost_random_accuracy= []
for lr, dep in xgboost_paras:
    xgboost_classifier= xgb.XGBClassifier(
                        objective='multi:softmax',
                        num_class=3,  #類別數
                        val_metric='mlogloss',  #多分類交叉熵損失
                        learning_rate=lr,  #學習率
                        max_depth=dep,  #樹的最大深度
                        use_label_encoder=False  #避免警告
             )
    cv_scores= cross_val_score(xgboost_classifier,X, Y, cv=10, scoring='accuracy')
    xgboost_random_accuracy.append(round(cv_scores.mean(), 6))

  #DNN
model= Sequential([
             Dense(32, activation= 'relu', input_shape= (X_train.shape[1], )),
             Dropout(0.2),
             Dense(32, activation= 'relu'),
             Dropout(0.2),
             Dense(7, activation= 'softmax')
      ])
kfold= StratifiedKFold(n_splits= 10) #隨機分割資料成十份
DNN_random_accuracy= []
for bz, lr in DNN_paras:
    optimizer= Adam(learning_rate= lr)
    model.compile(optimizer= optimizer,
    loss= 'sparse_categorical_crossentropy',
    metrics= ['accuracy'])
    cv_scores= []
    for train_index, val_index in kfold.split(X, Y):
        x_train, x_val= X[train_index], X[val_index]
        y_train, y_val= Y[train_index], Y[val_index]
        model.fit(x_train, y_train, epochs= 50, batch_size= bz, verbose= 0)
        _,accuracy= model.evaluate(x_val, y_val, verbose= 0)
        cv_scores.append(accuracy)
    mean_accuracy= np.mean(cv_scores)
    DNN_random_accuracy.append(round(mean_accuracy, 6))
 
  #K Nearest Neighbor
diff_nebor_accuracy= []
for n in num_neighbors:
    knn= KNeighborsClassifier(n_neighbors= n)
    cv_scores= cross_val_score(knn, X, Y, cv=10)
    diff_nebor_accuracy.append(round(cv_scores.mean(), 6)) 
 
 #畫調節參數與準確度的圖形
#   #AdaBoost
# plt.figure(figsize= (8, 6))
# plt.scatter(adaboost_n_iterations, adaboost_learning_rate, color= 'blue', label= 'Data 
# Points')
# for i, ga in enumerate(adaboost_random_accuracy):
#  plt.text(adaboost_n_iterations[i], adaboost_learning_rate[i], ga, fontsize= 16, 
# ha= 'center', va= 'bottom')
# plt.title('Cross-Validation Accuracy for AdaBoost', fontsize= 24)
# plt.xlabel('Number of Iterations', fontsize= 24)
# plt.ylabel('Learning Rate', fontsize= 24)
# plt.xticks(fontsize= 16)
# plt.yticks(fontsize=16)
# plt.grid(True)
# plt.show()

  #XGBoost
lt.figure(figsize= (8, 6))
plt.scatter(xgboost_max_depth, xgboost_learning_rate, color= 'blue', label= 'Data Points')
for i, ga in enumerate(xgboost_random_accuracy):
    plt.text(xgboost_max_depth[i], xgboost_learning_rate[i], ga, fontsize= 16, ha= 'center', va= 'bottom')
plt.title('Cross-Validation Accuracy for XGBoost', fontsize= 24)

  #DNN
plt.figure(figsize=(8, 6))
plt.scatter(batch_size, DNN_learning_rate, color= 'blue', label= 'Data Points')
for i, ga in enumerate(DNN_random_accuracy):
    plt.text(batch_size[i], DNN_learning_rate[i], ga, fontsize= 16, ha= 'center', va='bottom')
plt.title('Cross-Validation Accuracy for NN(Softmax)', fontsize= 24)
plt.xlabel('Batch Size', fontsize= 24)
plt.ylabel('Learning Rate', fontsize= 24)
plt.xticks(fontsize= 16)
plt.yticks(fontsize= 16)
plt.grid(True)
plt.show()

  #K Nearest Neighbor
plt.figure(figsize=(8, 6))
plt.plot(num_neighbors, diff_nebor_accuracy)
for i in range(num_neighbors.size):
    plt.text(num_neighbors[i], diff_nebor_accuracy[i], diff_nebor_accuracy[i], fontsize= 16, ha= 'center', va= 'bottom')
plt.title('Cross-Validation Accuracy for KNN', fontsize= 24)
plt.xlabel('Number of Neighbors', fontsize= 24)
plt.ylabel('Accuracy', fontsize= 24)
plt.xticks(fontsize= 16)
plt.yticks(fontsize= 16)
plt.grid(True)
plt.show()
