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
from sklearn.metrics import f1_score

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
#将性别和年龄做缺失值插补，年龄用均值代替，性别用-1代替，表示未知
from sklearn.preprocessing import Imputer
def sexage(data):   
    data.x_001.fillna(value=-1,inplace=True)
    imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    data.x_002=Series(map(lambda x:int(x),imp.fit_transform(data.loc[:,"x_002"].values.reshape(-1,1))))
sexage(data1)

data1_SFinfo=data1.ix[:,0:19]

data1_SFinfo.drop("x_012",inplace=True,axis=1)

data1_CKinfo=data1.ix[:,19:40]

data1_JYinfo=data1.ix[:,40:130]

data1_FKinfo=data1.ix[:,130:146]

data1_HKinfo=data1.ix[:,146:187]

data1_SQinfo=data1.ix[:,187:]

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
def cutage(x):
        if x<27:
            return "A"#A=[18,27)
        elif x<36:
            return "B"#B=[27,36)
        elif x<45:
            return "C"#C=[36,45)
        else:
            return "D"#D=[45,59)
data1_SFinfo["x_002"]=data1_SFinfo["x_002"].map(cutage)
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
#xgb2.fit(data1_CKinfo,data1.y,eval_metric="auc")
s=data1_CKinfo.drop(["x_027","x_033"],axis=1)
select_fecols2=["x_020","x_021","x_024","x_025","x_026","x_027","x_029","x_030",
                   "x_031","x_032","x_033","x_034","x_035","x_036"]
data1_CKinfo=data1_CKinfo.loc[:,select_fecols2]
#对交易信息做特征选择
#分类筛选特征
select_fecols3=["x_041","x_045","x_048","x_052","x_055","x_059","x_062",
               "x_064","x_065","x_067","x_068","x_070","x_071","x_073",
               "x_074","x_078","x_081","x_085","x_088","x_096","x_100",
               "x_102","x_104","x_105","x_108","x_109","x_111","x_112","x_114",
               "x_115","x_117","x_118","x_120","x_121","x_125","x_128","x_130"]
data1_JYinfo=data1_JYinfo.loc[:,select_fecols3]
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
#根据特征重要性选择的特征如下
select_fecols3=["x_041","x_045","x_048","x_052","x_055","x_059",
               "x_065","x_067","x_068","x_070",
               "x_074","x_078","x_085","x_088",
               "x_108","x_114","x_121","x_125","x_128","x_130"]
data1_JYinfo=data1_JYinfo.loc[:,select_fecols3]
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
def use_credit(x,y):
    try:
        return float(y/x)
    except:
        return np.nan
ratelist=[]
for x,y in zip(data1_JYinfo.x_045,data1_JYinfo.x_052):
    try:
        ratelist.append(y/x)
    except:
        ratelist.append(np.nan)
data1_JYinfo["use_credit"]=ratelist   
#计算异地交易金额占比     
def use_yidi(x,y):
    try:
        return float(y/x)
    except:
        return np.nan
yidilist=[]
for x,y in zip(data1_JYinfo.x_045,data1_JYinfo.x_059):
    try:
        yidilist.append(y/x)
    except:
        yidilist.append(np.nan)
data1_JYinfo["use_yidi"]=yidilist
#计算夜间交易金额占比     
def use_night(x,y):
    try:
        return float(y/x)
    except:
        return np.nan
nightlist=[]
for x,y in zip(data1_JYinfo.x_045,data1_JYinfo.x_067):
    try:
        nightlist.append(y/x)
    except:
        nightlist.append(np.nan)
data1_JYinfo["use_night"]=nightlist
#计算公共事业交易金额占比     
def use_public(x,y):
    try:
        return float(y/x)
    except:
        return np.nan
publiclist=[]
for x,y in zip(data1_JYinfo.x_045,data1_JYinfo.x_070):
    try:
        publiclist.append(y/x)
    except:
        publiclist.append(np.nan)
data1_JYinfo["use_public"]=publiclist
#计算互联网交易金额占比     
def use_net(x,y):
    try:
        return float(y/x)
    except:
        return np.nan
netlist=[]
for x,y in zip(data1_JYinfo.x_045,data1_JYinfo.x_078):
    try:
        netlist.append(y/x)
    except:
        netlist.append(np.nan)
data1_JYinfo["use_net"]=netlist
#计算大额交易金额占比     
def use_big(x,y):
    try:
        return float(y/x)
    except:
        return np.nan
biglist=[]
for x,y in zip(data1_JYinfo.x_045,data1_JYinfo.x_085):
    try:
        biglist.append(y/x)
    except:
        biglist.append(np.nan)
data1_JYinfo["use_big"]=biglist
#计算异地交易金额占比     
def use_trip(x,y):
    try:
        return float(y/x)
    except:
        return np.nan
triplist=[]
for x,y in zip(data1_JYinfo.x_045,data1_JYinfo.x_114):
    try:
        triplist.append(y/x)
    except:
        triplist.append(np.nan)
data1_JYinfo["use_trip"]=triplist
#计算汽车交易金额占比     
def use_car(x,y):
    try:
        return float(y/x)
    except:
        return np.nan
carlist=[]
for x,y in zip(data1_JYinfo.x_045,data1_JYinfo.x_108):
    try:
        carlist.append(y/x)
    except:
        carlist.append(np.nan)
data1_JYinfo["use_car"]=carlist
#计算金融交易金额占比     
def use_JR(x,y):
    try:
        return float(y/x)
    except:
        return np.nan
JRlist=[]
for x,y in zip(data1_JYinfo.x_045,data1_JYinfo.x_125):
    try:
        JRlist.append(y/x)
    except:
        JRlist.append(np.nan)
data1_JYinfo["use_JR"]=JRlist
#计算金融交易金额占比     
def use_JY(x,y):
    try:
        return float(y/x)
    except:
        return np.nan
JYlist=[]
for x,y in zip(data1_JYinfo.x_045,data1_JYinfo.x_130):
    try:
        JYlist.append(y/x)
    except:
        JYlist.append(np.nan)
data1_JYinfo["use_JY"]=JYlist
#对放款信息做特征选择,只选择各个期限内的放款金额和放款笔数
select_fecols4=["x_131","x_133","x_134","x_138","x_139","x_143","x_144"]
data1_FKinfo.loc[:,select_fecols4]
#对还款信息做特征选择
select_fecols5=["x_147","x_148","x_154","x_155","x_157","x_159","x_167","x_168",
                "x_170","x_172","x_180","x_181","x_183","x_185"]
data1_HKinfo=data1_HKinfo.loc[:,select_fecols5]
#申请贷款信息特征选择
select_fecols6=["x_190","x_191","x_194","x_195","x_198","x_199"]
data1_SQinfo=data1_SQinfo.loc[:,select_fecols6]
data1_new=pd.concat([data1_SFinfo,data1_CKinfo,data1_JYinfo,data1_FKinfo,
                 data1_HKinfo,data1_SQinfo],axis=1)
data1_new_train=data1_new.ix[0:11016,:]
data1_new_test=data1_new.ix[11017:,:]
X_train,X_test,Y_train,Y_test=train_test_split(data1_new_train,Y,test_size=0.3,random_state=42)

#训练xgboost模型
#设置初试参数
xgb_train = XGBClassifier(booster="gbtree",learning_rate =0.02,
                     n_estimators=1000,
                     max_depth=6,
                     min_child_weight=5,
                     gamma=0,
                     reg_alpha=65,
                     reg_lambda=10,
                     subsample=0.81,
                     colsample_bytree=0.81,
                     objective= 'binary:logistic',
                     nthread=8,
                     scale_pos_weight=3.9,
                     seed=27)
xgb_train.fit(X_train,Y_train,eval_metric="auc")
pred=xgb_train.predict(X_test)
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(Y_test,pred)
metrics.auc(fpr,tpr)
np.mean(f1_score(Y_test,pred,average=None))
y_prediction=xgb_train.predict(data1_new_test)
predict_result=DataFrame({"user_id":user_id,"y_prediction":y_prediction})

#合并数据集
data1_new=pd.concat([data1_SFinfo,data1_CKinfo,data1_JYinfo,data1_FKinfo,
                 data1_HKinfo,data1_SQinfo],axis=1)
data1_new_train=data1_new.ix[0:11016,:]
data1_new_test=data1_new.ix[11017:,:]

data1,data2,Y_train,Y_test=train_test_split(data1_new_train,Y,test_size=0.1,random_state=42)
#data2作为线上模拟测试数据

data1_a=data1.ix[Y_train==1,:]
data1_a["y"]=1
data1_b=data1.ix[Y_train==0,:]
data1_b["y"]=0
data1_b1,data1_b2=train_test_split(data1_b,test_size=0.5,random_state=42)
data1_b3,data1_b4=train_test_split(data1_b1,test_size=0.5,random_state=42)
data1_b1,data1_b2=train_test_split(data1_b2,test_size=0.5,random_state=42)
#将标签是0的分为四个与标签是0的结合在一起，分别训练四个模型，再将这四个模型融合
data_a=data1_a.append(data1_b1,ignore_index=True)
data_b=data1_a.append(data1_b2,ignore_index=True)
data_c=data1_a.append(data1_b3,ignore_index=True)
data_d=data1_a.append(data1_b4,ignore_index=True)
#根据data_a数据集训练xgboost模型
#设置初试参数
xgb_train1 = XGBClassifier(booster="gbtree",learning_rate =0.025,
                     n_estimators=550,
                     max_depth=6,
                     min_child_weight=5,
                     gamma=0,
                     reg_alpha=65,
                     reg_lambda=10,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     objective= 'binary:logistic',
                     nthread=8,
                     scale_pos_weight=0.58,
                     seed=27)
xgb_train1.fit(data_a.drop("y",axis=1),data_a.y,eval_metric="auc")
pred1=xgb_train1.predict(data2)
np.mean(f1_score(Y_test,pred1,average=None))
#根据data_b数据训练xgboost模型
xgb_train2 = XGBClassifier(booster="gbtree",learning_rate =0.02,
                     n_estimators=450,
                     max_depth=6,
                     min_child_weight=5,
                     gamma=0,
                     reg_alpha=65,
                     reg_lambda=10,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     objective= 'binary:logistic',
                     nthread=8,
                     scale_pos_weight=0.72,
                     seed=27)
xgb_train2.fit(data_b.drop("y",axis=1),data_b.y,eval_metric="auc")
pred2=xgb_train2.predict(data2)
np.mean(f1_score(Y_test,pred2,average=None))                    
#根据data_c数据训练xgboost模型
xgb_train3= XGBClassifier(booster="gbtree",learning_rate =0.02,
                     n_estimators=450,
                     max_depth=6,
                     min_child_weight=5,
                     gamma=0,
                     reg_alpha=65,
                     reg_lambda=10,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     objective= 'binary:logistic',
                     nthread=8,
                     scale_pos_weight=0.59,
                     seed=27)
xgb_train3.fit(data_c.drop("y",axis=1),data_c.y,eval_metric="auc")
pred3=xgb_train3.predict(data2)
np.mean(f1_score(Y_test,pred3,average=None))  
#根据data_d数据训练xgboost模型
xgb_train4 = XGBClassifier(booster="gbtree",learning_rate =0.06,
                     n_estimators=500,
                     max_depth=6,
                     min_child_weight=5,
                     gamma=0,
                     reg_alpha=65,
                     reg_lambda=10,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     objective= 'binary:logistic',
                     nthread=8,
                     scale_pos_weight=0.57,
                     seed=27)
xgb_train4.fit(data_d.drop("y",axis=1),data_d.y,eval_metric="auc")
pred4=xgb_train4.predict(data2)
np.mean(f1_score(Y_test,pred4,average=None))
#投票法融合
from sklearn.ensemble import VotingClassifier
clf=VotingClassifier([("clf1",xgb_train1),("clf2",xgb_train2),("clf3",xgb_train3),("clf4",xgb_train4)],weights=[1,1,1,1],flatten_transform=True)
                     
clf.fit(data_d.drop("y",axis=1),data_d.y)
pred5=clf.predict(data2)
np.mean(f1_score(Y_test,pred5,average=None))

                      

