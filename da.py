# -*- coding: utf-8 -*-
"""
Created on Sun May 14 10:59:55 2017

@author: lulu
"""

import pandas as pd
import numpy as np




class DAClass(object):
    
    #path_root = r'G:\python\data\FeatureEng'
    path_root = r'F:\Tencent\python\data'
    path_root_lulu = path_root + '\\lulu'
    # 编号, label, 点击时间, 回流时间, 素材ID, 用户ID, 曝光位置ID, 联网方式, 运营商,
    # 年龄, 性别, 学历, 婚恋情况, 育儿情况, 家乡, 现住地, 
    # 广告ID, 推广计划ID, 账户ID, appID, app平台, app类型
    # 位置节点结合, 广告位类型
    
    def testTempX(self):
        self.tempx =  pd.DataFrame({'userID':['zhanglulu','wangziyang'], 
                                    'appID':['10101', '20202'], 
                                    'time':[6,8] })
        
    def readTrainAll(self):
        path_traincsv = self.path_root + '\\data_train_all.csv'
        train_names = ['instanceID','label','clickTime','conversionTime','creativeID','userID','positionID','connectionType','telecomsOperator',
                       'age','gender','education','marriageStatus','haveBaby','hometown','residence',
                       'adID','camgaignID','advertiserID','appID','appPlatform','appCategory',
                       'sitesetID','positionType']

        #names=train_names, 文件里已经有列名了
        self.traincsv = pd.read_csv(path_traincsv, index_col='instanceID')

        #result = pd.Series(np.zeros(338490,int), index = testcsv.index)

        
    def featureEngCont(self, ftype):
        #posdata:所有成功转化的样本(正样本)  93,262个(一共3,749,528,占到2.5%) 
        self.posdata = da.traincsv[da.traincsv['label'] == ftype] 

        # 1.点击时间分布  [0:2]:天   [2:4]:小时  [4:6]：分钟
        self.clicktime_day_count =  self.posdata.clickTime.map(lambda x : str(x)[4:6]).value_counts()
        self.clicktime_day_count.to_csv(self.path_root_lulu  + '\\1_clicktime_count.csv')
        #print('************* clicktime_count: *************')
        #print(self.clicktime_day_count)
        #print('************* describe **************')
        #print(self.clicktime_day_count.describe())
        #print('cv: ' + str(self.clicktime_day_count.std() / self.clicktime_day_count.mean()) + '\n')
        
        
        # 2. 回流时间？？？
        
        # 3.素材ID
        self.creativeID_count =  self.posdata.creativeID.value_counts()
        self.creativeID_count.to_csv(self.path_root_lulu  + '\\3_creativeID_count.csv')
        
        # 4.用户ID
        self.userID_count =  self.posdata.userID.value_counts()
        self.userID_count.to_csv(self.path_root_lulu  + '\\4_userID_count.csv')
        
        # 5.曝光位置ID
        self.positionID_count =  self.posdata.positionID.value_counts()
        self.positionID_count.to_csv(self.path_root_lulu  + '\\5_positionID_count.csv')
        
        # 6.联网
        self.connectionType_count =  self.posdata.connectionType.value_counts()
        
        # 7.运营商
        self.telecomsOperator_count =  self.posdata.telecomsOperator.value_counts()
        
        # 4-1 user.年龄
        self.age_count =  self.posdata.age.value_counts()
        self.age_count.to_csv(self.path_root_lulu  + '\\4-1age_count.csv')
        
        # 4-2 user.性别
        self.gender_count =  self.posdata.gender.value_counts()
        
        # 4-3 user.学历
        self.education_count =  self.posdata.education.value_counts()
        
        # 4-4 user.婚恋
        self.marriageStatus_count =  self.posdata.marriageStatus.value_counts()
        
        # 4-5 user.育儿 
        self.haveBaby_count =  self.posdata.haveBaby.value_counts()
        
        # 4-6 user.家乡
        self.hometown_count =  self.posdata.hometown.value_counts()
        self.hometown_count.to_csv(self.path_root_lulu  + '\\4-6hometown_count.csv')
        
        # 4-7 user.现居地
        self.residence_count =  self.posdata.residence.value_counts()
        self.residence_count.to_csv(self.path_root_lulu  + '\\4-7residence_count.csv')
        
        # 3-1 广告ID
        self.adID_count_count =  self.posdata.adID.value_counts()
        self.adID_count_count.to_csv(da.path_root_lulu  + '\\3-1adID_count.csv')
        #self.adID_count_count.describe()
        
        # 3-2 推广计划ID
        self.camgaignID_count =  self.posdata.camgaignID.value_counts()
        self.camgaignID_count.to_csv(da.path_root_lulu  + '\\3-2camgaignID_count.csv')
        #self.camgaignID_count.describe()
        #self.camgaignID_count.std() / self.camgaignID_count.mean()
        
        # 3-3 账户ID
        self.advertiserID_count =  self.posdata.advertiserID.value_counts()
        self.advertiserID_count.to_csv(da.path_root_lulu  + '\\3-3advertiserID_count.csv')
        #self.advertiserID_count.describe()
        #self.advertiserID_count.std() / self.advertiserID_count.mean()
        
        # 3-4 appID
        self.appID_count =  self.posdata.appID.value_counts()
        self.appID_count.to_csv(da.path_root_lulu  + '\\3-4appID_count.csv')
        #self.appID_count.describe()
        #self.appID_count.std() / self.appID_count.mean()
        
        # 3-5 app平台
        self.appPlatform_count =  self.posdata.appPlatform.value_counts()
        #self.appPlatform_count.describe()
        #self.appPlatform_count.std() / self.appPlatform_count.mean()
        
        # 3-6 app类型
        self.appCategory_count =  self.posdata.appCategory.value_counts()
        #self.appCategory_count.describe()
        #self.appCategory_count.std() / self.appCategory_count.mean()
        
        
        # 5-1 站点
        self.sitesetID_count =  self.posdata.sitesetID.value_counts()
        
        # 5-2 位置类型
        self.positionType_count =  self.posdata.positionType.value_counts()
        
    
    #-------------------- 计算连续值的CV --------------------
    def featureEng2(self):
        path_root = self.path_root
        #path_root = da.path_root
        
        
        feature = '3-4appID'           
        self.posdata = pd.read_csv(path_root+'\\'+feature+'_count_1.csv', header=None, names=[feature,'POS'], index_col=[feature]) 
        self.negdata = pd.read_csv(path_root+'\\'+feature+'_count_0.csv', header=None, names=[feature,'NEG'], index_col=[feature]) 
        
        self.data = pd.merge(self.posdata, self.negdata, how='outer', left_index=True, right_index=True).fillna(0)
        self.data['转化率'] =  self.data['POS'] / (self.data['POS'] + self.data['NEG'])
        self.data = self.data.sort_index(by=['转化率','POS'], ascending=False)
        
        self.data.to_csv(path_root+'\\'+feature+'_count.csv', columns=['POS', 'NEG', '转化率'])
        
        #da.data.describe()
        #da.data.std() / da.data.mean()
        
        
        
    #-------------------- 计算离散值的CV -------------------- 
    def featureEngUnCont(self):   
        self.posdata = da.traincsv[da.traincsv['label'] == 1]
        self.negdata = da.traincsv[da.traincsv['label'] == 0]
        
        feature_uncont = ['connectionType','telecomsOperator', 
                          'gender','education','marriageStatus','haveBaby',
                          'appPlatform','appCategory',
                          'sitesetID','positionType']
        for feature in feature_uncont:
            print('\n--------------------- ' + feature + ' ---------------------\n')
            pos = self.posdata[feature].value_counts()
            pos.name = 'POS'
            neg = self.negdata[feature].value_counts()
            neg.name = 'NEG'
        
            self.result = pd.concat([pos,neg], axis=1).fillna(0)  #全连接
            self.result['转化率'] =  self.result['POS'] / (self.result['POS'] + self.result['NEG'])
        
            print(self.result)
            print('\n' + feature + '.describe(): ')
            print(self.result['转化率'].describe())
            print('\n变异系数CV：')
            print(self.result['转化率'].std() / self.result['转化率'].mean())
        
    #-------------------- 计算家乡中 【省份，城市】的CV --------------------    
    def featureEngHome(self):
          
        self.traincsv['province'] = self.traincsv['hometown'] // 100 % 100
        self.traincsv['city'] = self.traincsv['hometown'] % 100
        
        self.x = self.traincsv.groupby(['province','label']).size()
        self.y = self.traincsv.groupby(['city','label']).size()
        
        self.xx = self.x.unstack()
        self.yy = self.y.unstack()        
                        
        self.result1 = self.xx[1] / (self.xx[0] + self.xx[1])
        
 
    #-------------------- 查看奇怪ID -------------------- 
    def featureOdd(self):
        self.creativeID_count = pd.read_csv(self.path_root_lulu+'\\3_creativeID_count.csv')
        self.odd = self.creativeID_count[(self.creativeID_count['trans'] > 0.3) 
                                            & (self.creativeID_count['POS'] + self.creativeID_count['NEG'] < 20)]
        
        self.x = list(da.odd['3_creativeID'])
        
        self.y = self.traincsv[self.traincsv['creativeID'].isin(self.x)]
        
        
    def test(self):
        self.traincsv_only = pd.read_csv(self.path_root+'\\pre\\train.csv')
        self.actions = pd.read_csv(self.path_root+'\\pre\\user_app_actions.csv')
        
        #x = da.traincsv_only[(da.traincsv_only['label'] == 1) & (da.traincsv_only['clickTime'] // 10000 == 30)]
        #y = da.actions[da.actions['installTime'] //10000 == 30]

        t1 = da.actions.drop_duplicates(['userID','appID'])
        
        
if __name__ == '__main__':
    da = DAClass()
    da.testTempX()
    da.readTrainAll()
    #da.featureEng(0)
    #da.featureEng2()
    #da.featureEngUnCont()
    #da.featureEngHome()
    #da.featureOdd()
    da.test()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    