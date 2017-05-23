
# coding: utf-8

# In[2]:

import xgboost as xgb
import pandas as pd
import numpy as np
import time

# In[3]:
PATH_ROOT = r'G:\python\data'
PATH_DATA_TRAIN_ALL = PATH_ROOT + r'\data_train_all.csv'
PATH_DATA_TEST = PATH_ROOT + r'\data_test_all.csv'
TIME_TRAIN = time.strftime('%m-%d-%H-%M',time.localtime(time.time()))
PATH_SUBMIT = PATH_ROOT + '\\' + TIME_TRAIN + r'.csv'

# In[4]:

# -----------------导入数据，dtrain表示训练数据，dval表示验证数据
# -----------------train_label表示训练标签，val_label表示测试标签
data_train_all = pd.read_csv(PATH_DATA_TRAIN_ALL)
#数据预处理-训练数据
data_train_day = [17,18,19,20,21,22, 24,25,26,27,28,29] 

data_train = data_train_all[ (data_train_all['clickTime'] // 10000).isin(data_train_day)]
data_train['clickTime'] = data_train['clickTime'] // 100 % 100 * 60  + data_train['clickTime'] % 100

#数据预处理-验证数据
data_val_day = [23,30] 
data_val = data_train_all[ (data_train_all['clickTime'] // 10000).isin(data_val_day)]
data_val['clickTime'] = data_val['clickTime'] // 100 % 100 * 60  + data_val['clickTime'] % 100

#指定训练数据、验证数据 及 相应的标签
train_label = data_train['label'].values
data_train.drop(['instanceID', 'label', 'conversionTime'], axis=1, inplace=True)
train_data = data_train.values

val_label = data_val['label'].values
data_val.drop(['instanceID', 'label', 'conversionTime'], axis=1, inplace=True)
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

# ------保存模型-----------------
bst.dump_model('dump05211755.raw.txt')
# bst.dump_model('dump.raw.txt', 'featmap.txt')


# In[67]:

# --------------读入数据并调用模型进行预测-----------------
data_test = pd.read_csv(PATH_DATA_TEST)
data_test['clickTime'] = data_test['clickTime'] // 100 % 100 * 60  + data_test['clickTime'] % 100
data_test_ = data_test.drop(['instanceID', 'label'], axis=1)

xgb_dtest = xgb.DMatrix(data_test_.values, missing=0.0)
ypred = bst.predict(xgb_dtest, ntree_limit=bst.best_iteration)


# In[68]:

# ----------------保存预测结果-------------------------
data_test['pred'] = ypred
data_test[['instanceID','pred']].to_csv(PATH_SUBMIT,index=None)


# In[ ]:



