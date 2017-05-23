
# coding: utf-8

# In[2]:

import xgboost as xgb
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt  
import operator
# In[3]:
PATH_ROOT = r'G:\python\data'
PATH_DATA_TRAIN_ALL = PATH_ROOT + r'\data_train_all.csv'
PATH_DATA_TEST = PATH_ROOT + r'\data_test_all.csv'
TIME_TRAIN = time.strftime('%m-%d-%H-%M',time.localtime(time.time()))
PATH_SUBMIT_MODEL = PATH_ROOT + '\\' + TIME_TRAIN + r'.raw.txt'
PATH_SUBMIT = PATH_ROOT + '\\' + TIME_TRAIN + r'.csv'

print(PATH_SUBMIT)

# In[4]:

# -----------------导入数据，dtrain表示训练数据，dval表示验证数据
# -----------------train_label表示训练标签，val_label表示测试标签
data_train_all = pd.read_csv(PATH_DATA_TRAIN_ALL)

#------------------数据预处理-训练数据
data_train_day = [23,24,25,26,27,28,29] 
data_train = data_train_all[ (data_train_all['clickTime'] // 10000).isin(data_train_day)]

#分解点击时间： 小时 分钟
data_train['clickTimeHour'] = data_train['clickTime'] // 100 % 100
data_train['clickTimeMinute'] = data_train['clickTime'] % 100

#分解家乡： 省份 城市
data_train['hometown_province'] = data_train['hometown'] // 100 % 100
data_train['hometown_city'] = data_train['hometown'] % 100

#分解现居地： 省份 城市
data_train['residence_province'] = data_train['residence'] // 100 % 100
data_train['residence_city'] = data_train['residence'] % 100


#------------------数据预处理-验证数据
data_val_day = [30] 
data_val = data_train_all[ (data_train_all['clickTime'] // 10000).isin(data_val_day)]

#分解点击时间： 小时 分钟
data_val['clickTimeHour'] = data_val['clickTime'] // 100 % 100
data_val['clickTimeMinute'] = data_val['clickTime'] % 100

#分解家乡： 省份 城市
data_val['hometown_province'] = data_val['hometown'] // 100 % 100
data_val['hometown_city'] = data_val['hometown'] % 100

#分解现居地： 省份 城市
data_val['residence_province'] = data_val['residence'] // 100 % 100
data_val['residence_city'] = data_val['residence'] % 100


#指定训练数据、验证数据 及 相应的标签
train_label = data_train['label'].values
data_train.drop(['instanceID', 'label', 'conversionTime', 'clickTime'], axis=1, inplace=True)
train_data = data_train.values

val_label = data_val['label'].values
data_val.drop(['instanceID', 'label', 'conversionTime',  'clickTime' ], axis=1, inplace=True)
val_data = data_val.values



# In[5]:

# ------------------获得训练用的xgb数据格式-----------------------------
xgb_dtrain = xgb.DMatrix(train_data, label=train_label, missing=0.0)
xgb_dval = xgb.DMatrix(val_data, label=val_label, missing=0.0)


# In[63]:

# ------------------ 设置参数 ------------------------------------
param = {'bst:max_depth': 5, 
         'bst:subsample': 0.8,
         'bst:min_child_weight': 1,
         'bst:colsample_bytree': 0.2,
         'bst:eta': 0.2, 
         'bst:gamma': 0.2,
         'bst"min_child_leaf': 1,
         'bst:scale_pos_weight': 0,
         #'booster': 'gbtree',
         'silent': 0,
         'lambda': 10,
         'objective': 'binary:logistic',
         'eval_metric': 'logloss',
         'seed':55 }
plst = list(param.items())


# In[64]:

evallist = [(xgb_dtrain, 'train'),(xgb_dval, 'eval')]


# In[65]:

num_round =100000
print(plst)
bst = xgb.train(plst, xgb_dtrain, num_round, evallist, early_stopping_rounds=10)

# In[66]:

#特征选择
importance = bst.get_fscore()  
importance = sorted(importance.items(), key=operator.itemgetter(1))  
df = pd.DataFrame(importance, columns=['feature', 'fscore'])

df['feature'].map()
plt.figure()  
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))  
plt.title('XGBoost Feature Importance')  
plt.xlabel('relative importance')  
plt.show() 

# In[66]:

# ------保存模型-----------------
bst.dump_model(PATH_SUBMIT_MODEL)
# bst.dump_model('dump.raw.txt', 'featmap.txt')


# In[67]:

# --------------读入数据并调用模型进行预测-----------------
data_test = pd.read_csv(PATH_DATA_TEST)
#分解点击时间
data_test['clickTimeHour'] = data_test['clickTime'] // 100 % 100
data_test['clickTimeMinute'] = data_test['clickTime'] % 100

#分解家乡
data_test['hometown_province'] = data_test['hometown'] // 100 % 100
data_test['hometown_city'] = data_test['hometown'] % 100

#分解现居地： 省份 城市
data_test['residence_province'] = data_test['residence'] // 100 % 100
data_test['residence_city'] = data_test['residence'] % 100


data_test_ = data_test.drop(['instanceID', 'label', 'clickTime'], axis=1)

xgb_dtest = xgb.DMatrix(data_test_.values, missing=0.0)
ypred = bst.predict(xgb_dtest, ntree_limit=bst.best_iteration)


# In[68]:

# ----------------保存预测结果-------------------------
data_test['pred'] = ypred
data_test[['instanceID','pred']].to_csv(PATH_SUBMIT,index=None)


# In[ ]:



