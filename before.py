# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 20:30:08 2018

@author: YUBO
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 18:04:41 2018

@author: YUBO
"""
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import sklearn
from collections import Counter
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.metrics import f1_score
data=pd.read_csv("model_sample.csv")
data.drop("user_id",inplace=True,axis=1)
#去掉各类银行持卡信息
data.drop(["x_034","x_035","x_036","x_037","x_038","x_039","x_040"],inplace=True,axis=1)
#去掉输错密码的信息
data.drop(["x_092","x_093","x_094","x_095"],inplace=True,axis=1)
data.info()
# 统计每列属性缺失值比例
list1=data.select_dtypes(include=["int64"]).describe().T.assign(missing_pct=data.apply(lambda x : (len(x)-x.count())/(float(len(x)))))
list2=data.select_dtypes(include=["float64"]).describe().T.assign(missing_pct=data.apply(lambda x : (len(x)-x.count())/(float(len(x)))))
#丢掉属性缺失比例超过89%的部分
data.drop(list2.ix[list2.missing_pct>0.89,:].index,inplace=True,axis=1)
#去掉x_012这个属性，因为全是0
data.drop("x_012",inplace=True,axis=1)
#将性别和年龄做缺失值插补，年龄用均值代替，性别选择众数代替,便于后面生成多项式特征
from sklearn.preprocessing import Imputer
imp1 = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
data.x_001=imp1.fit_transform(data.x_001.values.reshape(-1,1))
imp2 = Imputer(missing_values='NaN', strategy='mean', axis=0)
data.x_002=Series(map(lambda x:int(x),imp2.fit_transform(data.x_002.values.reshape(-1,1))))
#对个人身份及财产信息生成多项式特征
poly = PolynomialFeatures(2)
poly_features=DataFrame(poly.fit_transform(data.ix[:,1:19]))
data=pd.concat([poly_features,data.drop(data.ix[:,1:19],axis=1)],axis=1)

#选择100个特征
X_new=data.drop("y",axis=1)
Y_new=data.y
#分割数据集
X_new_train,X_new_test,Y_new_train,Y_new_test=train_test_split(X_new,Y_new,test_size=0.3,random_state=42)
#训练xgboost模型
#设置初试参数
xgb1 = XGBClassifier(booster="gbtree",
 learning_rate =0.04,
 n_estimators=990,
 max_depth=4,
 min_child_weight=8,
 gamma=0,
 reg_alpha=42,
 reg_lambda=36,
 subsample=0.87,
 colsample_bytree=0.6,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=4,
 seed=27)
xgb1.fit(X_new_train,Y_new_train,eval_metric="auc")
pred=xgb1.predict(X_new_test)
np.mean(f1_score(Y_new_test,pred,average=None))
