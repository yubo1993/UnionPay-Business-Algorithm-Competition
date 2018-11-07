# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 10:37:30 2018

@author: YUBO
"""
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from collections import Counter
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import f1_score,auc
import reportgen
seed=0
#import missingno#缺失值可视化
model_sample=pd.read_csv("model_sample.csv")
Y=model_sample.y
model_sample.drop(["user_id","y"],inplace=True,axis=1)
verify_sample=pd.read_csv("verify_sample.csv")
user_id=verify_sample.user_id
verify_sample.drop("user_id",inplace=True,axis=1)
data1=pd.concat([model_sample,verify_sample],axis=0,ignore_index=True)

# 统计每列属性缺失值比例
list1=data1.select_dtypes(include=["int64"]).describe().T.assign(missing_pct=data1.apply(lambda x : (len(x)-x.count())/(float(len(x)))))
list2=data1.select_dtypes(include=["float64"]).describe().T.assign(missing_pct=data1.apply(lambda x : (len(x)-x.count())/(float(len(x)))))
#丢掉属性缺失比例超过0.89的部分
#缺失比例较大的属性名
droplist=['x_062', 'x_063', 'x_064', 'x_065', 'x_066', 'x_067', 'x_068', 'x_069',
       'x_070', 'x_071', 'x_072', 'x_073','x_081', 'x_082', 'x_083', 'x_084', 
       'x_085', 'x_086', 'x_087','x_092', 'x_093', 'x_094', 'x_095', 'x_096', 'x_097', 'x_098', 'x_099',
       'x_100', 'x_101', 'x_102', 'x_103', 'x_104', 'x_105', 'x_106', 'x_107',
       'x_108', 'x_109', 'x_110', 'x_111', 'x_112', 'x_113', 'x_114', 'x_115',
       'x_116', 'x_117', 'x_118', 'x_119', 'x_120','x_128', 'x_129', 'x_130']
data1.drop(droplist,inplace=True,axis=1)

#将性别和年龄做缺失值插补，年龄用均值代替，性别用-1代替，表示未知
from sklearn.preprocessing import Imputer
def sexage(data):   
    data.x_001.fillna(value=-1,inplace=True)
    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    data.x_002=Series(map(lambda x:int(x),imp.fit_transform(data.loc[:,"x_002"].values.reshape(-1,1))))
sexage(data1)

data1_SFinfo=data1.ix[:,:19]

data1_SFinfo.drop("x_012",inplace=True,axis=1)

data1_CKinfo=data1.ix[:,19:40]

data1_JYinfo=data1.ix[:,40:79]

data1_FKinfo=data1.ix[:,79:95]

data1_HKinfo=data1.ix[:,95:136]

data1_SQinfo=data1.ix[:,136:]

#对身份和财产信息做特征选择,基于xgboost的特征选择
#xgb1 = XGBClassifier(booster="gbtree",
    #learning_rate =0.04,
   #n_estimators=500,
    #max_depth=4,
    #min_child_weight=6,
    #subsample=0.8,
    #colsample_bytree=0.8,
    #objective= 'binary:logistic',
    #nthread=8,
    #scale_pos_weight=4,
    #seed=27)
#xgb1.fit(data1_SFinfo,Y,eval_metric="auc")
#根据特征重要性选择的特征
select_fecols1=["x_001","x_002","x_003","x_004","x_005","x_006","x_009",
                   "x_010","x_013","x_014","x_016","x_017","x_018"]

#对年龄分箱
dis1=reportgen.preprocessing.Discretization(method='chimerge',max_intervals=5,threshold=5)
dis1.fit(data1_SFinfo.ix[0:11016,"x_002"],Y)
agebins=dis1.transform(data1_SFinfo.ix[:,"x_002"])
data1_SFinfo["x_002"]=agebins
data1_SFinfo=data1_SFinfo.loc[:,select_fecols1]
poly = PolynomialFeatures(2)
poly_features=DataFrame(poly.fit_transform(data1_SFinfo.drop("x_002",axis=1)))
data1_SFinfo=pd.concat([data1_SFinfo.x_002,poly_features],axis=1)
data1_SFinfo=pd.get_dummies(data1_SFinfo)
#对持卡信息做特征选择
#xgb2 = XGBClassifier(booster="gbtree",
    #learning_rate =0.04,
   # n_estimators=500,
   # max_depth=4,
    #min_child_weight=6,
    #subsample=0.8,
    #colsample_bytree=0.8,
   # objective= 'binary:logistic',
   # nthread=8,
    #scale_pos_weight=4,
    #seed=27)
#xgb2.fit(data1_CKinfo,Y,eval_metric="auc")
select_fecols2=["x_020","x_021","x_024","x_025","x_026","x_027","x_029","x_030",
                   "x_031","x_032","x_033","x_034","x_035","x_036"]
data1_CKinfo=data1_CKinfo.loc[:,select_fecols2]
#由于连续变量比较少，不需要分箱处理
#对交易信息做特征选择
#missingno.matrix(data1_JYinfo)
#list3=data1_JYinfo.select_dtypes(include=["float64"]).describe().T.assign(
       # missing_pct=data1_JYinfo.apply(lambda x : (len(x)-x.count())/(float(len(x)))))
#首先基于xgboost做特征选择
#xgb3 = XGBClassifier(booster="gbtree",
    #learning_rate =0.04,
    #n_estimators=500,
    #max_depth=4,
   # min_child_weight=6,
    #subsample=0.8,
    #colsample_bytree=0.8,
    #objective= 'binary:logistic',
    #nthread=8,
    #scale_pos_weight=4,
    #seed=27)
#xgb3.fit(data1_JYinfo,data1.y,eval_metric="auc")

#根据特征重要性剔除的特征如下
droplist1=["x_042","x_049","x_056","x_075","x_089"]
data1_JYinfo.drop(droplist1,axis=1,inplace=True)
#信用卡交易金额出现负数，这样可以统计一个信用卡是否透支作为一个特征
def credit(x):
        if x<0:
            return 1#透支
        elif x>=0:
            return 0#未透支
        else:
            return -1#未知
       
data1_JYinfo["credit"]=data1_JYinfo["x_052"].map(credit)
#计算信用卡支付在交易中的占比
ratelist=[]
for x,y in zip(data1_JYinfo.x_045,data1_JYinfo.x_052):
    try:
        ratelist.append(np.abs(y/x))
    except:
        ratelist.append(np.nan)
data1_JYinfo["use_credit"]=ratelist   
#计算异地交易金额占比     
yidilist=[]
for x,y in zip(data1_JYinfo.x_045,data1_JYinfo.x_059):
    try:
        yidilist.append(np.abs(y/x))
    except:
        yidilist.append(np.nan)
data1_JYinfo["use_yidi"]=yidilist
#计算互联网交易金额占比     
netlist=[]
for x,y in zip(data1_JYinfo.x_045,data1_JYinfo.x_078):
    try:
        netlist.append(np.abs(y/x))
    except:
        netlist.append(np.nan)
data1_JYinfo["use_net"]=netlist
#计算金融交易金额占比     
JRlist=[]
for x,y in zip(data1_JYinfo.x_045,data1_JYinfo.x_125):
    try:
        JRlist.append(y/x)
    except:
        JRlist.append(np.nan)
data1_JYinfo["use_JR"]=JRlist
#由于变量之间相关性的限制，让缺失值插值变得很困难，很容易失去插值的意义，所以这里直接考虑分箱，把缺失值当做一类
credit=data1_JYinfo.ix[:,"credit"]
drop1=data1_JYinfo.ix[:,"x_122"]
dis2=reportgen.preprocessing.Discretization(method='chimerge',max_intervals=3,threshold=5)
dis2.fit(drop1.ix[0:11016],Y)
drop1=dis2.transform(drop1)
data1_JYinfo1=DataFrame()
for x in data1_JYinfo.drop(["x_122","credit"],axis=1).columns:
    dis3=reportgen.preprocessing.Discretization(method='chimerge',max_intervals=8,threshold=5)
    dis3.fit(data1_JYinfo.ix[:11016,x],Y)
    data1_JYinfo1=pd.concat([data1_JYinfo1,dis3.transform(data1_JYinfo.ix[:,x])],axis=1)
data1_JYinfo=pd.concat([data1_JYinfo1,drop1,credit],axis=1)
data1_JYinfo.replace(np.nan,"null",inplace=True)
data1_JYinfo=pd.get_dummies(data1_JYinfo)
#对放款信息做特征选择
#xgb5 = XGBClassifier(booster="gbtree",learning_rate =0.1,
                    # n_estimators=500,
                     #max_depth=6,
                     #min_child_weight=5,
                     #gamma=0,
                    # subsample=0.81,
                     #colsample_bytree=0.81,
                     #objective= 'binary:logistic',
                     #nthread=8,
                     #scale_pos_weight=3.9,
                     #seed=27)
#xgb5.fit(data1_FKinfo.ix[0:11016,:],Y,eval_metric="auc")

data1_FKinfo.drop(["x_132","x_134","x_137"],axis=1,inplace=True)
#分箱处理
data1_FKinfo1=DataFrame()
for col in data1_FKinfo.columns:
    dis4=reportgen.preprocessing.Discretization(method='chimerge',max_intervals=8,threshold=5)
    dis4.fit(data1_FKinfo.ix[:11016,col],Y)
    data1_FKinfo1=pd.concat([data1_FKinfo1,dis4.transform(data1_FKinfo.ix[:,col])],axis=1)
data1_FKinfo1.replace(np.nan,"null",inplace=True)
data1_FKinfo=pd.get_dummies(data1_FKinfo1)
   
    
#对还款信息做特征选择
#xgb6 = XGBClassifier(booster="gbtree",learning_rate =0.1,
                    # n_estimators=1000,
                     #max_depth=6,
                     #min_child_weight=5,
                     #gamma=0,
                     #reg_alpha=65,
                     #reg_lambda=10,
                     #subsample=0.81,
                     #colsample_bytree=0.81,
                     #objective= 'binary:logistic',
                     #nthread=8,
                     #scale_pos_weight=3.9,
                    # seed=27)
#xgb6.fit(data1_HKinfo.ix[0:11016,:],Y,eval_metric="auc")
                 
droplist2=["x_149","x_150","x_151","x_152","x_154","x_155","x_156",
           "x_157","x_158","x_162","x_163","x_164","x_165","x_169","x_171",
           "x_175","x_176","x_177","x_178"]
data1_HKinfo.drop(droplist2,inplace=True,axis=1)
data1_HKinfo1=DataFrame()
for col in data1_HKinfo.columns:
    dis5=reportgen.preprocessing.Discretization(method='chimerge',max_intervals=6,threshold=5)
    dis5.fit(data1_HKinfo.ix[:11016,col],Y)
    data1_HKinfo1=pd.concat([data1_HKinfo1,dis5.transform(data1_HKinfo.ix[:,col])],axis=1)
data1_HKinfo1.replace(np.nan,"null",inplace=True)
data1_HKinfo=pd.get_dummies(data1_HKinfo1)



#申请贷款信息特征选择

#xgb7= XGBClassifier(booster="gbtree",
    #learning_rate =0.04,
    #n_estimators=500,
    #max_depth=4,
    #min_child_weight=6,
    #subsample=0.8,
    #colsample_bytree=0.8,
    #objective= 'binary:logistic',
    #nthread=8,
    #scale_pos_weight=4,
    #seed=27)
#xgb7.fit(data1_SQinfo.ix[0:11016,:],Y,eval_metric="auc")
#最后全部作为特征留下
data1_SQinfo1=DataFrame()
for col in data1_SQinfo.columns:
    dis6=reportgen.preprocessing.Discretization(method='chimerge',max_intervals=3,threshold=5)
    dis6.fit(data1_SQinfo.ix[:11016,col],Y)
    data1_SQinfo1=pd.concat([data1_SQinfo1,dis6.transform(data1_SQinfo.ix[:,col])],axis=1)
data1_SQinfo1.replace(np.nan,"null",inplace=True)
data1_SQinfo=pd.get_dummies(data1_SQinfo1)


#合并数据集
data1_new=pd.concat([data1_SFinfo,data1_CKinfo,data1_JYinfo,data1_FKinfo,
                 data1_HKinfo,data1_SQinfo],axis=1)
data1_new.columns=list(range(0,1401))
data1_new_train=data1_new.ix[0:11016,:]

#data1_new_train.to_csv("data_train.csv")
data1_new_test=data1_new.ix[11017:,:]
X_train,X_test,Y_train,Y_test=train_test_split(data1_new_train,Y,test_size=0.3,random_state=42)

#训练xgboost模型
#设置初试参数
xgb_train1 = XGBClassifier(booster="gbtree",learning_rate =0.02,
                     n_estimators=800,
                     max_depth=3,
                     min_child_weight=4,
                     gamma=0,
                     reg_alpha=10,
                     reg_lambda=10,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     objective= 'binary:logistic',
                     nthread=8,
                     scale_pos_weight=4,
                     seed=27)
xgb_train1.fit(X_train,Y_train,eval_metric="auc")
from sklearn.feature_selection import SelectFromModel
model = SelectFromModel(xgb_train1,threshold=0.01,prefit=True)
X_new = model.transform(X_train)
X_new_test=model.transform(X_test)
xgb_train2 = XGBClassifier(booster="gbtree",learning_rate =0.01,
                     n_estimators=1000,
                     max_depth=5,
                     min_child_weight=4,
                     gamma=0,
                     reg_alpha=60,
                     reg_lambda=10,
                     subsample=0.81,
                     colsample_bytree=0.81,
                     objective= 'binary:logistic',
                     nthread=8,
                     scale_pos_weight=4,
                     seed=27)
xgb_train2.fit(X_new,Y_train,eval_metric="auc")
pred=xgb_train2.predict(X_new_test)
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(Y_test,pred)
metrics.auc(fpr,tpr)
np.mean(f1_score(Y_test,pred,average=None))
y_prediction=xgb_train.predict(data1_new_test)
predict_result=DataFrame({"user_id":user_id,"y_prediction":y_prediction})

