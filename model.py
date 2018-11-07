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
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import VarianceThreshold,SelectFromModel

data=pd.read_csv("model_sample.csv")
data.drop("user_id",inplace=True,axis=1)
#将原始样本中的特征缺失数量计算出来当做一个特征
miss_num=data.apply(lambda x : len(x)-x.count(),axis=1)
#对身份信息利用xgboost做特征选择
user_info=data.ix[:,3:19]
xgb1 = XGBClassifier(booster="gbtree",
    learning_rate =0.04,
    n_estimators=500,
    max_depth=4,
    min_child_weight=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=4,
    scale_pos_weight=4,
    seed=27)
xgb1.fit(user_info,data.y,eval_metric="auc")
model1 = SelectFromModel(xgb1, prefit=True,threshold=0.04)
user_info_new=DataFrame(model1.transform(user_info))
user_info_new.columns=["x_003","x_004","x_005","x_006","x_009",
                   "x_010","x_012","x_013","x_015","x_016","x_017"]
#将性别和年龄做缺失值插补，年龄用均值代替，性别用-1代替，表示未知
data.x_001.fillna(value=-1,inplace=True)
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
data.x_002=Series(map(lambda x:int(x),imp.fit_transform(data.x_002.values.reshape(-1,1))))
#合并身份信息
user_info_new=pd.concat([data.ix[:,1:3],user_info_new],axis=1,ignore_index=True)
#对个人身份及财产信息生成多项式特征
poly = PolynomialFeatures(2)
poly_features=DataFrame(poly.fit_transform(user_info_new.ix[:,0:]))
data.info()
# 统计每列属性缺失值比例
list1=data.select_dtypes(include=["int64"]).describe().T.assign(missing_pct=data.apply(lambda x : (len(x)-x.count())/(float(len(x)))))
list2=data.select_dtypes(include=["float64"]).describe().T.assign(missing_pct=data.apply(lambda x : (len(x)-x.count())/(float(len(x)))))

#丢掉属性缺失比例超过0.72的部分
data.drop(list2.ix[list2.missing_pct>0.72,:].index,inplace=True,axis=1)
#缺失比例较大的属性名
#['x_062', 'x_063', 'x_064', 'x_065', 'x_066', 'x_067', 'x_068', 'x_069',
       #'x_070', 'x_071', 'x_072', 'x_073','x_081', 'x_082', 'x_083', 'x_084', 
       #'x_085', 'x_086', 'x_087','x_092', 'x_093', 'x_094', 'x_095', 'x_096', 'x_097', 'x_098', 'x_099',
       #'x_100', 'x_101', 'x_102', 'x_103', 'x_104', 'x_105', 'x_106', 'x_107',
       #'x_108', 'x_109', 'x_110', 'x_111', 'x_112', 'x_113', 'x_114', 'x_115',
       #'x_116', 'x_117', 'x_118', 'x_119', 'x_120','x_128', 'x_129', 'x_130'，'x_133','x_135','x_136']
#计算除身份财产信息之外的特征的变异系数，然后依据变异系数做特征选择
varlist=data.ix[:,41:].apply(lambda x:np.std(x)/np.mean(x),axis=0)
#去掉变异系数接近1或者小于1的特征['x_042', 'x_049', 'x_050', 'x_056', 'x_075', 'x_089', 'x_122', 'x_175',
      # 'x_196', 'x_197']
data.drop(varlist[varlist<1].index,axis=1,inplace=True)

#其余缺失属性值用中位数代替
imp1=Imputer(missing_values='NaN', strategy='median', axis=0)
for x in data.ix[:,41:].columns:
    data.ix[:,x]=imp1.fit_transform(data.ix[:,x].values.reshape(-1,1))
#对做了缺失值插补后的特征再利用xgboost做特征选择
xgb2 = XGBClassifier(booster="gbtree",
    learning_rate =0.04,
    n_estimators=500,
    max_depth=4,
    min_child_weight=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=4,
    scale_pos_weight=4,
    seed=27)
xgb2.fit(data.ix[:,41:],data.y,eval_metric="auc")
model2 = SelectFromModel(xgb2, prefit=True,threshold=0.007)
data_new=DataFrame(model2.transform(data.ix[:,41:]))
#将身份信息转化的身份信息和其余特征合并
X=pd.concat([poly_features,data_new],axis=1,ignore_index=True)
#加入特征缺失数量这个特征
X=pd.concat([X,miss_num],axis=1,ignore_index=True)
Y=data.y
   
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
#过采样
sm = SMOTE(k=5,random_state=42)
X_train,Y_train=sm.fit_sample(X_train,Y_train)
X_train=DataFrame(X_train)
Y_train=np.array(Y_train)
#线下测试集
X_test=DataFrame(X_test)
Y_test=np.array(Y_test)

#训练xgboost模型
#设置初试参数
xgb3 = XGBClassifier(booster="gbtree",learning_rate =0.1,
                     n_estimators=1000,
                     max_depth=5,
                     min_child_weight=2,
                     gamma=0,
                     reg_alpha=80,
                     reg_lambda=80,
                     subsample=0.8,
                     colsample_bytree=0.72,
                     objective= 'binary:logistic',
                     nthread=4,
                     scale_pos_weight=2,
                     seed=27)
xgb3.fit(X_train,Y_train,eval_metric="auc")
pred=xgb3.predict(X_test)
np.mean(f1_score(Y_test,pred,average=None))
#调整最大深度和最小子样比例
param_test1 = {
 'max_depth':list(range(3,6,1)),
 'min_child_weight':list(range(2,10,2))
}
gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=1000, max_depth=3,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=2, seed=27), 
 param_grid = param_test1, scoring='f1',n_jobs=-1,iid=False, cv=3)
gsearch1.fit(X_train,Y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
#调节gamma
param_test2 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=1000, max_depth=5,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=2,seed=27), 
 param_grid = param_test2, scoring='f1',n_jobs=-1,iid=False, cv=3)
gsearch2.fit(X_train,Y_train)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_ 
#调整取样比例参数   
param_test3 = {
 'colsample_bytree':[i/100.0 for i in range(70,80,2)],

}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=1000, max_depth=5,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.6,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=2,seed=27), 
 param_grid = param_test3, scoring='f1',n_jobs=-1,iid=False, cv=3)
gsearch3.fit(X_train,Y_train)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
#调整正则式参数
param_test4 = {
 'reg_lambda':list(range(80,90,2))
}
gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=1000, max_depth=5,
 min_child_weight=2, gamma=0, subsample=0.8, reg_alpha=80,colsample_bytree=0.72,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=2,seed=27), 
 param_grid = param_test4, scoring='f1',n_jobs=-1,iid=False, cv=3)
gsearch4.fit(X_train,Y_train)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
param_test5 = {
 'scale_pos_weight':np.linspace(1.5,2.5,5)
}
gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=1000, max_depth=5,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.72,reg_alpha=80,reg_lambda=80,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=2,seed=27), 
 param_grid = param_test5, scoring='f1',n_jobs=-1,iid=False, cv=5)
gsearch5.fit(X_train,Y_train)
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_
param_test6 = {
 'reg_lambda':np.linspace(0,1,10)
}
gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=500, max_depth=4,
 min_child_weight=8, gamma=0, subsample=0.87, colsample_bytree=0.6,reg_alpha=42,reg_lambda=36,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=3.9,seed=27), 
 param_grid = param_test6, scoring='f1',n_jobs=-1,iid=False, cv=3)
gsearch6.fit(X_train,Y_train)
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_
#调整学习率
param_test7 = {
 'learning_rate':np.linspace(0.03,0.1,5)
}
gsearch7 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=500, max_depth=4,
 min_child_weight=8, gamma=0, subsample=0.87, colsample_bytree=0.6,reg_alpha=42,reg_lambda=36,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=3.9,seed=27), 
 param_grid = param_test7, scoring='f1',n_jobs=-1,iid=False, cv=3)
gsearch7.fit(X_train,Y_train)
gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_
#调整树的数量
param_test8 = {
 'n_estimators':list(range(480,550,10))
}
gsearch8 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.04, n_estimators=500, max_depth=4,
 min_child_weight=8, gamma=0, subsample=0.87, colsample_bytree=0.6,reg_alpha=42,reg_lambda=36,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=4,seed=27), 
 param_grid = param_test8, scoring='f1',n_jobs=-1,iid=False, cv=3)
gsearch8.fit(X_new_train,Y_new_train)
gsearch8.grid_scores_, gsearch8.best_params_, gsearch8.best_score_

