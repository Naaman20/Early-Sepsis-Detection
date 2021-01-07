#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import os
from Function import fillNANValue,loadfilesnew,result,plot_feature_importances,Grid_result,pca
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.ensemble import BalancedBaggingClassifier


# In[2]:


# loading train & test data (part 1)
first_train_file_path1 = os.path.join("H:\\Project\\AI Projects\\Sepcis detection\\dataset\\training","p000001.psv")
first_train_file_df1 = pd.read_csv(first_train_file_path1,sep='|')
initial_df_train1 = fillNANValue(first_train_file_df1)
path1 = "H:\\Project\\AI Projects\\Sepcis detection\\dataset\\training"
file_in_dir = os.listdir(path1)
first_test_file_path1 = os.path.join("H:\\Project\\AI Projects\\Sepcis detection\\dataset\\training",file_in_dir[-1])
first_test_file_df1 = pd.read_csv(first_test_file_path1,sep='|')
initial_df_test1 = fillNANValue(first_test_file_df1)
file_in_dir = file_in_dir[1:-1]
file_in_dir = random.sample(file_in_dir, len(file_in_dir))
train_files1 = file_in_dir[:int(len(file_in_dir)*0.125)]
test_files1 = file_in_dir[int(len(file_in_dir)*0.125):]

train_data1 = loadfilesnew(path1, train_files1, initial_df_train1)
test_data1 = loadfilesnew(path1, test_files1, initial_df_test1)

print(train_data1.shape,test_data1.shape)


# In[5]:


# saving data to train & test
train_data1.to_csv('Test_data.csv',header = True, index = False, mode = 'w')
test_data1.to_csv('Train_data.csv',header = True, index = False, mode = 'w')


# In[16]:


# plot1
count1 = (train_data1["SepsisLabel"] == 1).sum()
count2 = (train_data1["SepsisLabel"] == 0).sum()
count = [count2 ,count1]
class_name = ['Not_Sepsis' , 'Sepsis']
print(count1, count2, 'sepsis = ', (count1/(count1+count2))*100)
sns.barplot(x = class_name , y = count , linewidth = 2.5 , errcolor = '.2' , edgecolor = '.2')


# In[17]:


# plot 2
count3 = (test_data1["SepsisLabel"] == 1).sum()
count4 = (test_data1["SepsisLabel"] == 0).sum()
count = [count4, count3]
class_name = ['Not_Sepsis' , 'Sepsis']
print(count3, count4, 'sepsis = ', (count3/(count3+count4))*100)
sns.barplot(x = class_name , y = count , linewidth = 2.5 , errcolor = '.2' , edgecolor = '.2')


# In[20]:


# loading train & test data (part 2)
first_train_file_path2 = os.path.join("H:\\Project\\AI Projects\\Sepcis detection\\dataset\\training_setB","p100001.psv")
first_train_file_df2 = pd.read_csv(first_train_file_path2,sep='|')
initial_df_train2 = fillNANValue(first_train_file_df2)
path2 = "H:\\Project\\AI Projects\\Sepcis detection\\dataset\\training_setB"
file_in_dir2 = os.listdir(path2)
first_test_file_path2 = os.path.join("H:\\Project\\AI Projects\\Sepcis detection\\dataset\\training_setB",file_in_dir2[-1])
first_test_file_df2 = pd.read_csv(first_test_file_path2,sep='|')
initial_df_test2 = fillNANValue(first_test_file_df2)
file_in_dir2 = file_in_dir2[1:-1]
file_in_dir2 = random.sample(file_in_dir2, len(file_in_dir2))
train_files2 = file_in_dir2[:int(len(file_in_dir2)*0.875)]
test_files2 = file_in_dir2[int(len(file_in_dir2)*0.875):]

train_data2 = loadfilesnew(path2, train_files2, initial_df_train2)
test_data2 = loadfilesnew(path2, test_files2, initial_df_test2)

print(train_data2.shape,test_data2.shape)


# In[23]:


# save data to  train & test file
test_data2.to_csv('Test_data.csv',header = False, index = False, mode = 'a')
train_data2.to_csv('Train_data.csv',header = False, index = False, mode = 'a')


# In[22]:


# plot 3
count1 = (train_data2["SepsisLabel"] == 1).sum()
count2 = (train_data2["SepsisLabel"] == 0).sum()
count = [count2 ,count1]
class_name = ['Not_Sepsis' , 'Sepsis']
print(count1, count2, 'sepsis = ', (count1/(count1+count2))*100)
sns.barplot(x = class_name , y = count , linewidth = 2.5 , errcolor = '.2' , edgecolor = '.2')


# In[21]:


# plot 4
count3 = (test_data2["SepsisLabel"] == 1).sum()
count4 = (test_data2["SepsisLabel"] == 0).sum()
count = [count4, count3]
class_name = ['Not_Sepsis' , 'Sepsis']
print(count3, count4, 'sepsis = ', (count3/(count3+count4))*100)
sns.barplot(x = class_name , y = count , linewidth = 2.5 , errcolor = '.2' , edgecolor = '.2')


# # List of different dataset

# 1. All raw data
# 2. Raw data with selected feature
# 3. All data with some feature changed to categorical
# 4. Categorical data with some selected feature

# In[19]:


# load with non-Nan valued data
path = 'H:\\Project\\AI Projects\\Sepsis Detection New' 
train_data = pd.read_csv(os.path.join(path, 'Train_data.csv'))
test_data = pd.read_csv(os.path.join(path, 'Test_data.csv'))
test_data.dropna(inplace = True)
train_data.dropna(inplace = True)
train_X, train_y = train_data.iloc[:,0:-1], train_data.iloc[:,-1]
test_X, test_y = test_data.iloc[:,0:-1], test_data.iloc[:,-1]


# In[6]:


# plot 4
count1 = (train_data["SepsisLabel"] == 1).sum()
count2 = (train_data["SepsisLabel"] == 0).sum()
count3 = (test_data["SepsisLabel"] == 1).sum()
count4 = (test_data["SepsisLabel"] == 0).sum()
count = [count1, count2, count3, count4]
class_name = ['Sepsis _train', 'Not_Sepsis_train', 'Sepsis_test', 'Not_Sepsis_test']
print(count1, count2, 'sepsis = ', (count1/(count1+count2))*100, ' , ', count3, count4, 'sepsis = ', (count3/(count3+count4))*100)
sns.barplot(x = class_name , y = count , linewidth = 2.5 , errcolor = '.2' , edgecolor = '.2')


# ##                                   simple raw data

# In[7]:


# simple all raw data
std = StandardScaler()
train_X_scaled = std.fit_transform(train_X)
test_X_scaled = std.transform(test_X)


# In[33]:

#                                               Decision Tree

# In[37]:


# decision tree classifier
from sklearn.tree import DecisionTreeClassifier
clf_DTC = DecisionTreeClassifier()
clf_DTC = clf_DTC.fit(train_X_scaled,train_y)
result(clf_DTC,train_X_scaled,train_y,test_X_scaled,test_y)

# OUTPUT
# Accuracy on test set 0.9573122570188846
# Precision on test set 0.061751497005988025
# Recall on test set 0.09068425391591096
# F1 score on test set 0.07347211399309808


#                                         
#                                         
#                                         Logistic Regression

# In[38]:


# Logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr = lr.fit(train_X_scaled,train_y)
result(lr,train_X_scaled,train_y,test_X_scaled,test_y)

# OUTPUT
# Accuracy on test set 0.9813462307794886
# Precision on test set 0.5192307692307693
# Recall on test set 0.00741962077493817
# F1 score on test set 0.014630181522622595


#                                                 
#                                                 
#                                                 
#                                                 Random Forest

# In[6]:


# random forest classifier
from sklearn.ensemble import RandomForestClassifier
clf_RF = RandomForestClassifier()
clf_RF = clf_RF.fit(train_X_scaled,train_y)
result(clf_RF,train_X_scaled,train_y,test_X_scaled,test_y)

# OUTPUT
# Accuracy on test set 0.9812692974447875
# Precision on test set 0.24
# Recall on test set 0.0016488046166529267
# F1 score on test set 0.0032751091703056767


#                                                Naive Bayes

# In[5]:


# naive bayes
from sklearn.naive_bayes import GaussianNB
clf_NB = GaussianNB()
clf_NB = clf_NB.fit(train_X_scaled,train_y)
result(clf_NB, train_X_scaled, train_y, test_X_scaled, test_y)

# OUTPUT
# Accuracy on test set 0.9008380604593433
# Precision on test set 0.04457083164064767
# Recall on test set 0.2110469909315746
# F1 score on test set 0.07359846669861046


#                                                Gradient Boosting 

# In[8]:


# gradiient boostng decision tree
from sklearn.ensemble import GradientBoostingClassifier
clf_GBC = GradientBoostingClassifier()
clf_GBC = clf_GBC.fit(train_X_scaled,train_y)
result(clf_GBC, train_X_scaled, train_y, test_X_scaled, test_y)

# OUTPUT
# Accuracy on test set 0.9809872085508837
# Precision on test set 0.211864406779661
# Recall on test set 0.006870019236053861
# F1 score on test set 0.013308490817141336


# ### Conclusion of raw data
# on the basis of above training these model are godd for this problem 
# 1. Decision Tree 2. Random Forest 3. Naive Bayes 
# 1. Neural Network 2. SVM 

# ## Raw data with selected feature manually

# In[9]:


train_data.columns


# In[15]:


# Raw data with selected feature manually
features = ['HR', 'Temp', 'DBP', 'SBP', 'Resp', 'Platelets', 'Calcium', 'Glucose', 'Age', 'Gender', 'HospAdmTime', 'ICULOS','pH', 'SepsisLabel']
train_data_with_selected_feature_manually = train_data[features]
test_data_with_selected_feature_manually = test_data[features]
train_X = train_data_with_selected_feature_manually.iloc[:,0:-1]
train_y = train_data_with_selected_feature_manually.iloc[:,-1]
test_X = test_data_with_selected_feature_manually.iloc[:,0:-1]
test_y = test_data_with_selected_feature_manually.iloc[:,-1]

# scaling
std = StandardScaler()
train_X_scaled = std.fit_transform(train_X)
test_X_scaled = std.transform(test_X)


# ###                                                    Decision Tree

# In[11]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer , f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from Function import Grid_result

params = {'max_depth' : [6,10,25,50] ,
          'min_samples_split' : [5,10,15] ,
          'min_samples_leaf' : [4,8,12], 
          'class_weight' : ['balanced' , {0:1 , 1:9} , {0:1 , 1:10} , {0:5 , 1:5} , {0:1 , 1:90} , {0:1 , 1:99}]
          }
recall = []
precision = []
f1_scores = []
grid_params = []
for max_depth in [6,10,25,50]:
    for min_samples_leaf in [4,8,12]:
        for class_weight in ['balanced' , {0:1 , 1:9} , {0:1 , 1:10} , {0:5 , 1:5} , {0:1 , 1:90} , {0:1 , 1:99}]:
            dtc = DecisionTreeClassifier(max_depth = max_depth, min_samples_leaf = min_samples_leaf, class_weight = class_weight)
            dtc = dtc.fit(train_X_scaled, train_y)
            pred_test_y = dtc.predict(test_X)
            re = recall_score(test_y,pred_test_y)
            pre = precision_score(test_y,pred_test_y)
            f1 = f1_score(test_y,pred_test_y)
            grid_param = '['+','.join([str(max_depth),str(min_samples_leaf),str(class_weight)])+']'
            grid_params.append(grid_param) 
            print("at {} --> recall : {} , precision : {} , f1-score : {}".format(grid_param,re,pre,f1))
            recall.append(re)
            precision.append(pre)
            f1_scores.append(f1)


# In[20]:


plt.figure(figsize= (40,10))
params = len(grid_params)
plt.plot(range(params),recall , label = 'recall')
plt.plot(range(params),precision , label = 'precision')
plt.plot(range(params),f1_scores , label = 'f1-score')
plt.xticks(np.arange(params), grid_params) 
plt.gca().margins(x = 0)
plt.gcf().canvas.draw()
t1 = plt.gca().get_xticklabels()
maxszie = max([t.get_window_extent().width for t in t1])
m = 0.2
s = maxszie/plt.gcf().dpi*params+2*m
margin = m/plt.gcf().get_size_inches()[0]
plt.gcf().subplots_adjust(left = margin, right = 1.-margin)
plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
plt.legend()
plt.show()

# OUTPUT 
# best paramenter on Recall = (10,8,{0:1,1:9})
# best parameter on Precision =  (10,8,balanced)
# best parameter on F1_Score = (10,8,balanced)
# best parameters on average of all or overall good model = (10,8,balanced)


# In[6]:


# decision tree classifier on best hyperparameter
from sklearn.tree import DecisionTreeClassifier
clf_DTC = DecisionTreeClassifier(max_depth = 10, min_samples_leaf = 8, class_weight = 'balanced')
clf_DTC = clf_DTC.fit(train_X_scaled,train_y)
result(clf_DTC,train_X_scaled,train_y,test_X_scaled,test_y)

# OUTPUT
# Accuracy on test set 0.7753187604501113
# Precision on test set 0.044435875334573334
# Recall on test set 0.5383347073371806
# F1 score on test set 0.08209533787323206


# ### Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer , f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from Function import Grid_result

recall = []
precision = []
f1_scores = []
grid_params = []

for n_estimators in [100,200,300]:
    for max_depth in [6,10,25,50]:
        for min_samples_leaf in [4,8,12]:
            for class_weight in ['balanced' , {0:1 , 1:9} , {0:1 , 1:10} , {0:5 , 1:5} , {0:1 , 1:90} , {0:1 , 1:99}, None]:
                rf = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, min_samples_leaf = min_samples_leaf, class_weight = class_weight)
                rf = rf.fit(train_X_scaled, train_y)
                pred_test_y = rf.predict(test_X)
                re = recall_score(test_y,pred_test_y)
                pre = precision_score(test_y,pred_test_y)
                f1 = f1_score(test_y,pred_test_y)
                grid_param = '[' + ','.join([str(n_estimators), str(max_depth), str(min_samples_leaf), str(class_weight)]) + ']'
                grid_params.append(grid_param) 
                print("at {} --> recall : {} , precision : {} , f1-score : {}".format(grid_param,re,pre,f1))
                recall.append(re)
                precision.append(pre)
                f1_scores.append(f1)

plt.figure(figsize= (40,10))
params = len(grid_params)
plt.plot(range(params),recall , label = 'recall')
plt.plot(range(params),precision , label = 'precision')
plt.plot(range(params),f1_scores , label = 'f1-score')
plt.xticks(np.arange(params), grid_params) 
plt.gca().margins(x = 0)
plt.gcf().canvas.draw()
t1 = plt.gca().get_xticklabels()
maxszie = max([t.get_window_extent().width for t in t1])
m = 0.2
s = maxszie/plt.gcf().dpi*params+2*m
margin = m/plt.gcf().get_size_inches()[0]
plt.gcf().subplots_adjust(left = margin, right = 1.-margin)
plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
plt.legend()
plt.show()

# OUTPUT 
# best paramenter on Recall = [100,6,4,{0: 1, 1: 9}]
# best parameter on Precision =  [100,10,8,balanced]
# best parameter on F1_Score = [100,25,4,{0: 1, 1: 10}]
# best parameters on average of all or overall good model = [100,25,4,{0: 1, 1: 10}]


# In[ ]:


# random Forest on best parameter
# random forest classifier
from sklearn.ensemble import RandomForestClassifier
clf_RF = RandomForestClassifier(n_estimators = 100, max_depth = 25, min_samples_leaf = 4, class_weight = {0: 1, 1: 10})
clf_RF = clf_RF.fit(train_X_scaled,train_y)
result(clf_RF,train_X_scaled,train_y,test_X_scaled,test_y)

# OUTPUT
# Accuracy on test set 0.9812692974447875
# Precision on test set 0.24
# Recall on test set 0.0016488046166529267
# F1 score on test set 0.0032751091703056767


# ### Naive Bayes

# In[5]:


# naive bayes
from sklearn.naive_bayes import GaussianNB
clf_NB = GaussianNB()
clf_NB = clf_NB.fit(train_X_scaled,train_y)
result(clf_NB, train_X_scaled, train_y, test_X_scaled, test_y)

# OUTPUT
# Accuracy on test set 0.9266363720290911
# Precision on test set 0.06762344928241304
# Recall on test set 0.2291838417147568
# F1 score on test set 0.10443275732531931


# ## data distribution of this data

# In[16]:


train_pca_X, test_pca_X = pca(train_X_scaled, train_y, test_X_scaled, 2, ['feature1', 'feature2'])


# In[19]:


plt.scatter(train_pca_X[0::5000,0],train_pca_X[0::5000,1], c = train_y[0::5000])
plt.title('train data distribution')


# In[20]:


plt.scatter(test_pca_X[0::1000,0],test_pca_X[0::1000,1], c = test_y[0::1000])
plt.title('test data distribution')


# ## Data with categorical version

# In[20]:


# Data with categorical version
# changing data to categorical

# HR = (normal(60-90), down, high)
# DBP = (normal(70-80), lower, higher)
# SBP = (normal(100-120), lower, higher)
# Age = (children(< = 14), teenage(15-21),adult(21-45), old(>45))
# Temp = (normal(35-38), lower, higher)
# Resp = (normal(14-18), lower, higher)
# pH = (normal(7.35 - 7.45), lower, higher)
# Platelets = (normal(240 - 300), lower, higher)
bin_age = [0,14,21,45,120]
label_age = ['children', 'teen', 'adult', 'old']
bin_HR = [0,60,90,200]
label_HR = ['down', 'normal', 'higher']
bin_SBP = [0,100,121,300]
label_SBP = ['down', 'normal', 'higher']
bin_DBP = [0,70,81,300]
label_DBP = ['down', 'normal', 'higher']
bin_Temp = [0,35,39,70]
label_Temp = ['down', 'normal', 'higher']
bin_Resp = [0,14,19,40]
label_Resp = ['down', 'normal', 'higher']
bin_pH = [0,7.35,7.46,10]
label_pH = ['down', 'normal', 'higher']
bin_Platelets = [0,240,301,1000]
label_Platelets = ['down', 'normal', 'higher']
train_data['Age'] = pd.cut(train_data['Age'], bins = bin_age, labels = label_age, right = False)
test_data['Age'] = pd.cut(test_data['Age'], bins = bin_age, labels = label_age, right = False)
train_data['HR'] = pd.cut(train_data['HR'], bins = bin_HR, labels = label_HR, right = False)
test_data['HR'] = pd.cut(test_data['HR'], bins = bin_HR, labels = label_HR, right = False)
train_data['SBP'] = pd.cut(train_data['SBP'], bins = bin_SBP, labels = label_SBP, right = False)
test_data['SBP'] = pd.cut(test_data['SBP'], bins = bin_SBP, labels = label_SBP, right = False)
train_data['DBP'] = pd.cut(train_data['DBP'], bins = bin_DBP, labels = label_DBP, right = False)
test_data['DBP'] = pd.cut(test_data['DBP'], bins = bin_DBP, labels = label_DBP, right = False)
train_data['Temp'] = pd.cut(train_data['Temp'], bins = bin_Temp, labels = label_Temp, right = False)
test_data['Temp'] = pd.cut(test_data['Temp'], bins = bin_Temp, labels = label_Temp, right = False)
train_data['Resp'] = pd.cut(train_data['Resp'], bins = bin_Resp, labels = label_Resp, right = False)
test_data['Resp'] = pd.cut(test_data['Resp'], bins = bin_Resp, labels = label_Resp, right = False)
train_data['pH'] = pd.cut(train_data['pH'], bins = bin_pH, labels = label_pH, right = False)
test_data['pH'] = pd.cut(test_data['pH'], bins = bin_pH, labels = label_pH, right = False)
train_data['Platelets'] = pd.cut(train_data['Platelets'], bins = bin_Platelets, labels = label_Platelets, right = False)
test_data['Platelets'] = pd.cut(test_data['Platelets'], bins = bin_Platelets, labels = label_Platelets, right = False)


# In[21]:


train_data = pd.get_dummies(train_data)
test_data = pd.get_dummies(test_data)
col_seq = [i for i in train_data.columns if i != 'SepsisLabel']
col_seq.append('SepsisLabel')
train_data = train_data.reindex(columns = col_seq)
test_data = test_data.reindex(columns = col_seq)


# #### on all data

# In[22]:


# scaling
train_X, train_y = train_data.iloc[:,0:-1], train_data.iloc[:,-1]
test_X, test_y = test_data.iloc[:,0:-1], test_data.iloc[:,-1]
std = StandardScaler()
train_X_scaled = std.fit_transform(train_X)
test_X_scaled = std.transform(test_X)


#                                                             Decision Tree

# In[28]:


# decision tree classifier on best hyperparameter
from sklearn.tree import DecisionTreeClassifier
clf_DTC = DecisionTreeClassifier(max_depth = 10, min_samples_leaf = 8, class_weight = 'balanced')
clf_DTC = clf_DTC.fit(train_X_scaled,train_y)
result(clf_DTC,train_X_scaled,train_y,test_X_scaled,test_y)

# OUTPUT
# Accuracy on test set 0.7696000492373342
# Precision on test set 0.04613118142439367
# Recall on test set 0.57653201428964
# F1 score on test set 0.0854269310639684


#                                                             Random Forest

# In[30]:


# random Forest on best parameter
# random forest classifier
from sklearn.ensemble import RandomForestClassifier
clf_RF = RandomForestClassifier(n_estimators = 100, max_depth = 25, min_samples_leaf = 4, class_weight = {0: 1, 1: 10})
clf_RF = clf_RF.fit(train_X_scaled,train_y)
result(clf_RF,train_X_scaled,train_y,test_X_scaled,test_y)

# OUTPUT
# Accuracy on test set 0.9707037861458451
# Precision on test set 0.10574362875618105
# Recall on test set 0.07639461390491893
# F1 score on test set 0.0887045309508615


#                                                             Naive Bayes

# In[31]:


# naive bayes
from sklearn.naive_bayes import GaussianNB
clf_NB = GaussianNB()
clf_NB = clf_NB.fit(train_X_scaled,train_y)
result(clf_NB, train_X_scaled, train_y, test_X_scaled, test_y)

# OUTPUT
# Accuracy on test set 0.8977607270713018
# Precision on test set 0.042506597787635464
# Recall on test set 0.2080241824677109
# F1 score on test set 0.0705893323386796


# ### categorical data with seleted feature

# In[4]:


# categorical data with selected feature manually
features = ['HR', 'Temp', 'DBP', 'SBP', 'Resp', 'Platelets', 'Calcium', 'Glucose', 'Age', 'Gender', 'HospAdmTime', 'ICULOS','pH', 'SepsisLabel']
train_data = train_data[features]
test_data = test_data[features]

train_data = pd.get_dummies(train_data)
test_data = pd.get_dummies(test_data)
col_seq = [i for i in train_data.columns if i != 'SepsisLabel']
col_seq.append('SepsisLabel')
train_data = train_data.reindex(columns = col_seq)
test_data = test_data.reindex(columns = col_seq)

train_X = train_data.iloc[:,0:-1]
train_y = train_data.iloc[:,-1]
test_X = test_data.iloc[:,0:-1]
test_y = test_data.iloc[:,-1]
print(train_X.shape,test_X.shape)
# scaling
std = StandardScaler()
train_X_scaled = std.fit_transform(train_X)
test_X_scaled = std.transform(test_X)


#                                                         Decision Tree

# In[5]:


# decision tree classifier on best hyperparameter
from sklearn.tree import DecisionTreeClassifier
clf_DTC = DecisionTreeClassifier(max_depth = 10, min_samples_leaf = 8, class_weight = 'balanced')
clf_DTC = clf_DTC.fit(train_X_scaled,train_y)
result(clf_DTC,train_X_scaled,train_y,test_X_scaled,test_y)

# OUTPUT
# Accuracy on test set 0.7755803337880949
# Precision on test set 0.04588983722351769
# Recall on test set 0.557021159659247
# F1 score on test set 0.08479397615561599


#                                                         Random Forest

# In[6]:


# random Forest on best parameter
# random forest classifier
from sklearn.ensemble import RandomForestClassifier
clf_RF = RandomForestClassifier(n_estimators = 100, max_depth = 25, min_samples_leaf = 4, class_weight = {0: 1, 1: 10})
clf_RF = clf_RF.fit(train_X_scaled,train_y)
result(clf_RF,train_X_scaled,train_y,test_X_scaled,test_y)

# OUTPUT
# Accuracy on test set 0.9656826038343574
# Precision on test set 0.11696787148594377
# Recall on test set 0.12805715856004396
# F1 score on test set 0.12226157680703134


# In[7]:


# naive bayes
from sklearn.naive_bayes import GaussianNB
clf_NB = GaussianNB()
clf_NB = clf_NB.fit(train_X_scaled,train_y)
result(clf_NB, train_X_scaled, train_y, test_X_scaled, test_y)

# OUTPUT
# Accuracy on test set 0.9267389498086924
# Precision on test set 0.058628410315946594
# Recall on test set 0.19428414399560318
# F1 score on test set 0.09007516881131353


# ## Using Data balancing Technique

# ### On Raw Data

# #### Balance Bagging Classifier

# In[8]:


# decision tree classifier on best hyperparameter
from sklearn.tree import DecisionTreeClassifier
clf_DTC = DecisionTreeClassifier(max_depth = 10, min_samples_leaf = 8, class_weight = 'balanced')
bb_clf = BalancedBaggingClassifier(base_estimator = clf_DTC  , sampling_strategy = 'auto' ,replacement = False ,random_state = 0).fit(train_X_scaled,train_y)
result(bb_clf,train_X_scaled,train_y,test_X_scaled,test_y)

# OUTPUT
# Accuracy on test set 0.8029532142747238
# Precision on test set 0.05453660535888109
# Recall on test set 0.5850508381423468
# F1 score on test set 0.09977271129648291


# In[12]:


# Random Forest classifier on best hyperparameter
from sklearn.ensemble import RandomForestClassifier
clf_RF = RandomForestClassifier(n_estimators = 100, max_depth = 25, min_samples_leaf = 4, class_weight = {0: 1, 1: 10})
bb_clf = BalancedBaggingClassifier(base_estimator = clf_RF  , sampling_strategy = 'auto' ,replacement = False ,random_state = 0).fit(train_X_scaled,train_y)
result(bb_clf,train_X_scaled,train_y,test_X_scaled,test_y)

# OUTPUT
# Accuracy on test set 0.41542974960764
# Precision on test set 0.027646731452545058
# Recall on test set 0.8873316845287167
# F1 score on test set 0.05362273112244051


# In[13]:


# naive bayes
from sklearn.naive_bayes import GaussianNB
clf_NB = GaussianNB()
bb_clf = BalancedBaggingClassifier(base_estimator = clf_NB  , sampling_strategy = 'auto' ,replacement = False ,random_state = 0).fit(train_X_scaled,train_y)
result(bb_clf,train_X_scaled,train_y,test_X_scaled,test_y)

# OUTPUT
# Accuracy on test set 0.7755803337880949
# Precision on test set 0.04588983722351769
# Recall on test set 0.557021159659247
# F1 score on test set 0.08479397615561599


# ### on selected feature data

# #### Balance Bagging Classifier

# In[16]:


# decision tree classifier on best hyperparameter
from sklearn.tree import DecisionTreeClassifier
clf_DTC = DecisionTreeClassifier(max_depth = 10, min_samples_leaf = 8, class_weight = 'balanced')
bb_clf = BalancedBaggingClassifier(base_estimator = clf_DTC  , sampling_strategy = 'auto' ,replacement = False ,random_state = 0).fit(train_X_scaled,train_y)
result(bb_clf,train_X_scaled,train_y,test_X_scaled,test_y)

# OUTPUT
# Accuracy on test set 0.8076717921363874
# Precision on test set 0.05310949213388238
# Recall on test set 0.5528991481176148
# F1 score on test set 0.09691014618404258


# In[17]:


# Random Forest classifier on best hyperparameter
from sklearn.ensemble import RandomForestClassifier
clf_RF = RandomForestClassifier(n_estimators = 100, max_depth = 25, min_samples_leaf = 4, class_weight = {0: 1, 1: 10})
bb_clf = BalancedBaggingClassifier(base_estimator = clf_RF  , sampling_strategy = 'auto' ,replacement = False ,random_state = 0).fit(train_X_scaled,train_y)
result(bb_clf,train_X_scaled,train_y,test_X_scaled,test_y)

# OUTPUT
# Accuracy on test set 0.5178844358735011
# Precision on test set 0.03100638397259563
# Recall on test set 0.8208298983237153
# F1 score on test set 0.05975553643947427


# In[18]:


# naive bayes
from sklearn.naive_bayes import GaussianNB
clf_NB = GaussianNB()
bb_clf = BalancedBaggingClassifier(base_estimator = clf_NB  , sampling_strategy = 'auto' ,replacement = False ,random_state = 0).fit(train_X_scaled,train_y)
result(bb_clf,train_X_scaled,train_y,test_X_scaled,test_y)

# OUTPUT
# Accuracy on test set 0.853559961841066
# Precision on test set 0.050160702033151566
# Recall on test set 0.3816982687551525
# F1 score on test set 0.08866900734120652


# ### All data with some feature changed to categorical

# #### Balance Bagging Classifier

# In[23]:


# decision tree classifier on best hyperparameter
from sklearn.tree import DecisionTreeClassifier
clf_DTC = DecisionTreeClassifier(max_depth = 10, min_samples_leaf = 8, class_weight = 'balanced')
bb_clf = BalancedBaggingClassifier(base_estimator = clf_DTC  , sampling_strategy = 'auto' ,replacement = False ,random_state = 0).fit(train_X_scaled,train_y)
result(bb_clf,train_X_scaled,train_y,test_X_scaled,test_y)

# OUTPUT
# Accuracy on test set 0.7932750007693333
# Precision on test set 0.052612313624051346
# Recall on test set 0.592470458917285
# F1 score on test set 0.09664261060558518


# In[24]:


# Random Forest classifier on best hyperparameter
from sklearn.ensemble import RandomForestClassifier
clf_RF = RandomForestClassifier(n_estimators = 100, max_depth = 25, min_samples_leaf = 4, class_weight = {0: 1, 1: 10})
bb_clf = BalancedBaggingClassifier(base_estimator = clf_RF  , sampling_strategy = 'auto' ,replacement = False ,random_state = 0).fit(train_X_scaled,train_y)
result(bb_clf,train_X_scaled,train_y,test_X_scaled,test_y)

# OUTPUT
# Accuracy on test set 0.1931898612122642
# Precision on test set 0.02179595698068114
# Recall on test set 0.9623522945864248
# F1 score on test set 0.04262648270657472

