#!/usr/bin/env python
# coding: utf-8

# In[73]:


get_ipython().system('pip install imbalanced-learn')
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import model_selection, naive_bayes, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter


# In[61]:


df = pd.read_csv('Dataset_1_Training.csv',encoding='latin-1')


# In[62]:


df = df.T
df


# In[63]:


df = df.iloc[1:]
# df = df.iloc[:, :-2]
df_train = df.iloc[:,-2:]
df_train.columns = ['CO:_1', 'CO:_2']
df_train.to_csv('Train_Y.csv')


# In[64]:


df_train


# In[65]:


#df = df.iloc[1:]
df = df.iloc[:, :-2]
df


# In[66]:


X1 = df.to_numpy() 


# In[67]:


X1


# In[68]:


df.to_csv('Train_X.csv')
df_one = pd.read_csv('Train_Y.csv',encoding='latin-1')


# In[69]:


df_one = df_one.iloc[:,-2:]
df_one


# In[70]:


Y_one = df_one['CO:_1']
Y_two = df_one['CO:_2']


# In[71]:


df_one.apply(pd.value_counts)


# In[74]:


sns.countplot(df_one['CO:_1'])


# In[75]:


sns.countplot(df_one['CO:_2'])


# In[77]:


sm = SMOTE(random_state=42)
X_1, Y_1 = sm.fit_resample(X1, Y_one)


# In[78]:


print(Counter(Y_1))


# In[79]:


Train_X1, Test_X1, Train_Y1, Test_Y1  = model_selection.train_test_split(X_1, Y_1, test_size=0.3,random_state=0)


# In[21]:


# model_params = {
#     'svm': {
#         'model': svm.SVC(gamma='auto'),
#         'params' : {
#             'C': [1,5,30,50],
#             'kernel': ['rbf','linear'],
#             'random_state' : [1,5,10]
#         }  
#     },
#     'random_forest': {
#         'model': RandomForestClassifier(),
#         'params' : {
#             'n_estimators': [10,15,20,30,50],
#             'random_state':[1,10,30,4,50,100],
#             'max_depth':[2,3,5]
#         }
#     },
#     'logistic_regression' : {
#         'model': LogisticRegression(solver='liblinear',multi_class='auto'),
#         'params': {
#             'C': [1,5,10,20,50,100]
#         }
#     }
# }
# scores = []

# for model_name, mp in model_params.items():
#     clf1 =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
#     clf1.fit(Train_X1, Train_Y1)
#     scores.append({
#         'model': model_name,
#         'best_score': clf1.best_score_,
#         'best_params': clf1.best_params_
#     })
    
# df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
# df

# 0	svm	0.903704	{'C': 5, 'kernel': 'rbf', 'random_state': 1}
# 1	random_forest	0.896296	{'max_depth': 3, 'n_estimators': 30, 'random_state': 1}
# 2	logistic_regression	0.888889	{'C': 1}


# In[80]:


sm = SMOTE(random_state=42)
X_2, Y_2 = sm.fit_resample(X1, Y_two)


# In[81]:


print(Counter(Y_2))


# In[82]:


Train_X2, Test_X2, Train_Y2, Test_Y2  = model_selection.train_test_split(X_2, Y_2, test_size=0.3,random_state=0)


# In[83]:


# model_params = {
#     'svm': {
#         'model': svm.SVC(gamma='auto'),
#         'params' : {
#             'C': [1,5,30,50],
#             'kernel': ['rbf','linear'],
#             'random_state' : [1,5,10]
#         }  
#     },
#     'random_forest': {
#         'model': RandomForestClassifier(),
#         'params' : {
#             'n_estimators': [10,15,20,30,50],
#             'random_state':[1,10,30,50,100],
#             'max_depth':[2,3,5]
#         }
#     },
#     'logistic_regression' : {
#         'model': LogisticRegression(solver='liblinear',multi_class='auto'),
#         'params': {
#             'C': [1,5,10,20,50,100]
#         }
#     }
# }
# scores = []

# for model_name, mp in model_params.items():
#     clf1 =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
#     clf1.fit(Train_X2, Train_Y2)
#     scores.append({
#         'model': model_name,
#         'best_score': clf1.best_score_,
#         'best_params': clf1.best_params_
#     })
    
# df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
# df

# 0	svm	0.786147	{'C': 1, 'kernel': 'rbf', 'random_state': 1}
# 1	random_forest	0.787296	{'max_depth': 5, 'n_estimators': 30, 'random_state': 10}
# 2	logistic_regression	0.776190	{'C': 100}


# In[84]:


df2 = pd.read_csv('Dataset_2_Training.csv',encoding='latin-1')


# In[85]:


df2 = df2.T
df2


# In[86]:


df2 = df2.iloc[1:]
df2_train = df2.iloc[:,-4:]
df2_train.columns = ['CO:_3', 'CO:_4','CO:_5', 'CO:_6']
df2_train.to_csv('Train_Y2.csv')
df2_train


# In[87]:


df2 = df2.iloc[:, :-4]
df2


# In[88]:


X2 = df2.to_numpy() 
X2


# In[89]:


df2.to_csv('Train_X2.csv')
df_two = pd.read_csv('Train_Y2.csv',encoding='latin-1')
df_two = df_two.iloc[:,-4:]
df_two


# In[90]:


Y_three = df_two['CO:_3']
Y_four = df_two['CO:_4']
Y_five = df_two['CO:_5']
Y_six = df_two['CO:_6']


# In[91]:


df_two.apply(pd.value_counts)


# In[93]:


sns.countplot(df_two['CO:_3'])


# In[94]:


sns.countplot(df_two['CO:_4'])


# In[95]:


sns.countplot(df_two['CO:_5'])


# In[96]:


sns.countplot(df_two['CO:_6'])


# In[97]:


sm = SMOTE(random_state=42)
X_3, Y_3 = sm.fit_resample(X2, Y_three)


# In[98]:


print(Counter(Y_3))


# In[99]:


Train_X3, Test_X3, Train_Y3, Test_Y3  = model_selection.train_test_split(X_3, Y_3, test_size=0.3,random_state=0)


# In[36]:


# model_params = {
#     'svm': {
#         'model': svm.SVC(gamma='auto'),
#         'params' : {
#             'C': [1,5,30,50],
#             'kernel': ['rbf','linear'],
#             'random_state' : [1,5,10]
#         }  
#     },
#     'random_forest': {
#         'model': RandomForestClassifier(),
#         'params' : {
#             'n_estimators': [10,15,20,30,50],
#             'random_state':[1,10,30,50,100],
#             'max_depth':[2,3,5]
#         }
#     },
#     'logistic_regression' : {
#         'model': LogisticRegression(solver='liblinear',multi_class='auto'),
#         'params': {
#             'C': [1,5,10,20,50,100]
#         }
#     }
# }
# scores = []

# for model_name, mp in model_params.items():
#     clf1 =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
#     clf1.fit(Train_X3, Train_Y3)
#     scores.append({
#         'model': model_name,
#         'best_score': clf1.best_score_,
#         'best_params': clf1.best_params_
#     })
    
# df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
# df


# 0	svm	0.921987	{'C': 1, 'kernel': 'linear', 'random_state': 1}
# 1	random_forest	0.846753	{'max_depth': 5, 'n_estimators': 30, 'random_state': 100}
# 2	logistic_regression	0.919210	{'C': 1}


# In[100]:


sm = SMOTE(random_state=42)
X_4, Y_4 = sm.fit_resample(X2, Y_four)


# In[101]:


print(Counter(Y_4))


# In[102]:


Train_X4, Test_X4, Train_Y4, Test_Y4  = model_selection.train_test_split(X_4, Y_4, test_size=0.3,random_state=0)


# In[103]:



# model_params = {
#     'svm': {
#         'model': svm.SVC(gamma='auto'),
#         'params' : {
#             'C': [1,5,30,50],
#             'kernel': ['rbf','linear'],
#             'random_state' : [1,5,10]
#         }  
#     },
#     'random_forest': {
#         'model': RandomForestClassifier(),
#         'params' : {
#             'n_estimators': [10,15,20,30,50],
#             'random_state':[1,10,30,50,100],
#             'max_depth':[2,3,5]
#         }
#     },
#     'logistic_regression' : {
#         'model': LogisticRegression(solver='liblinear',multi_class='auto'),
#         'params': {
#             'C': [1,5,10,20,50,100]
#         }
#     }
# }
# scores = []

# for model_name, mp in model_params.items():
#     clf1 =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
#     clf1.fit(Train_X4, Train_Y4)
#     scores.append({
#         'model': model_name,
#         'best_score': clf1.best_score_,
#         'best_params': clf1.best_params_
#     })
    
# df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
# df


# 0	svm	0.977685	{'C': 5, 'kernel': 'rbf', 'random_state': 1}
# 1	random_forest	0.943086	{'max_depth': 5, 'n_estimators': 50, 'random_s...
# 2	logistic_regression	0.972747	{'C': 1}


# In[104]:


sm = SMOTE(random_state=42)
X_5, Y_5 = sm.fit_resample(X2, Y_five)


# In[105]:


print(Counter(Y_5))


# In[106]:


Train_X5, Test_X5, Train_Y5, Test_Y5  = model_selection.train_test_split(X_5, Y_5, test_size=0.3,random_state=0)


# In[44]:


# model_params = {
#     'svm': {
#         'model': svm.SVC(gamma='auto'),
#         'params' : {
#             'C': [1,5,30,50],
#             'kernel': ['rbf','linear'],
#             'random_state' : [1,5,10]
#         }  
#     },
#     'random_forest': {
#         'model': RandomForestClassifier(),
#         'params' : {
#             'n_estimators': [10,15,20,30,50],
#             'random_state':[1,10,30,50,100],
#             'max_depth':[2,3,5]
#         }
#     },
#     'logistic_regression' : {
#         'model': LogisticRegression(solver='liblinear',multi_class='auto'),
#         'params': {
#             'C': [1,5,10,20,50,100]
#         }
#     }
# }
# scores = []

# for model_name, mp in model_params.items():
#     clf1 =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
#     clf1.fit(Train_X5, Train_Y5)
#     scores.append({
#         'model': model_name,
#         'best_score': clf1.best_score_,
#         'best_params': clf1.best_params_
#     })
    
# df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
# df


# 0	svm	0.885455	{'C': 1, 'kernel': 'linear', 'random_state': 1}
# 1	random_forest	0.878047	{'max_depth': 2, 'n_estimators': 50, 'random_s...
# 2	logistic_regression	0.874411	{'C': 50}


# In[107]:


sm = SMOTE(random_state=42)
X_6, Y_6 = sm.fit_resample(X2, Y_six)


# In[108]:


print(Counter(Y_6))


# In[109]:


Train_X6, Test_X6, Train_Y6, Test_Y6  = model_selection.train_test_split(X_6, Y_6, test_size=0.3,random_state=0)


# In[110]:


# model_params = {
#     'svm': {
#         'model': svm.SVC(gamma='auto'),
#         'params' : {
#             'C': [1,5,30,50],
#             'kernel': ['rbf','linear'],
#             'random_state' : [1,5,10]
#         }  
#     },
#     'random_forest': {
#         'model': RandomForestClassifier(),
#         'params' : {
#             'n_estimators': [10,15,20,30,50],
#             'random_state':[1,10,30,50,100],
#             'max_depth':[2,3,5]
#         }
#     },
#     'logistic_regression' : {
#         'model': LogisticRegression(solver='liblinear',multi_class='auto'),
#         'params': {
#             'C': [1,5,10,20,50,100]
#         }
#     }
# }
# scores = []

# for model_name, mp in model_params.items():
#     clf6 =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
#     clf6.fit(Train_X6, Train_Y6)
#     scores.append({
#         'model': model_name,
#         'best_score': clf6.best_score_,
#         'best_params': clf6.best_params_
#     })
    
# df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
# df

# 0	svm	0.689286	{'C': 1, 'kernel': 'linear', 'random_state': 1}
# 1	random_forest	0.657143	{'max_depth': 3, 'n_estimators': 10, 'random_s...
# 2	logistic_regression	0.678571	{'C': 10}


# In[122]:


rdf1 = svm.SVC(C=10,kernel='rbf')
rdf1 = rdf1.fit(X1,Y_one)

rdf2 = RandomForestClassifier(n_estimators=5,random_state=40)
rdf2 = rdf2.fit(X1,Y_two)

rdf3 = RandomForestClassifier(n_estimators=50,random_state=20)
rdf3 = rdf3.fit(X2,Y_three)

rdf4 = RandomForestClassifier(n_estimators=15,random_state=10)
rdf4 = rdf4.fit(X2,Y_four)

rdf5 = RandomForestClassifier(n_estimators=100,random_state=1)
rdf5 = rdf5.fit(X2,Y_five)

rdf6 = AdaBoostClassifier(n_estimators=12,learning_rate=0.5)
rdf6 = rdf6.fit(X2,Y_six)


# In[123]:


df_test1 = pd.read_csv('Dataset_1_Testing.csv',encoding='latin-1')
df_test2 = pd.read_csv('Dataset_2_Testing.csv',encoding='latin-1')


# In[124]:


df_test1 = df_test1.T
df_test2 = df_test2.T
df_test1 = df_test1.iloc[1:]
df_test2 = df_test2.iloc[1:]
X_test1 = df_test1.to_numpy()
X_test2 = df_test2.to_numpy() 


# In[125]:


Y_test_CO1=rdf1.predict(X_test1)


# In[126]:


Y_test_CO2=rdf2.predict(X_test1)


# In[127]:


Y_test_CO3=rdf3.predict(X_test2)
Y_test_CO4=rdf4.predict(X_test2)
Y_test_CO5=rdf5.predict(X_test2)
Y_test_CO6=rdf6.predict(X_test2)


# In[128]:


final_array = np.concatenate((Y_test_CO1,Y_test_CO2,Y_test_CO3,Y_test_CO4,Y_test_CO5,Y_test_CO6),axis = 0)
df = pd.DataFrame()
df['Predicted']=final_array
df.insert(0,'Id', df.index)

df.to_csv('CS21M023_CS21M039.csv',index=False)


# In[130]:



# These, models gave highest accuracy/score. All clinical descriptors when trained on AdaBoost, whose 
# accuracy was 0.45244 on kaggle


# rdf1 = AdaBoostClassifier(n_estimators=12,learning_rate=0.5,random_state=1)
# rdf1 = rdf1.fit(X1,Y_one)

# rdf2 = AdaBoostClassifier(n_estimators=12,learning_rate=0.5,random_state=1)
# rdf2 = rdf2.fit(X1,Y_two)

# rdf3 = AdaBoostClassifier(n_estimators=12,learning_rate=0.5,random_state=1)
# rdf3 = rdf3.fit(X2,Y_three)

# rdf4 = AdaBoostClassifier(n_estimators=12,learning_rate=0.5,random_state=1)
# rdf4 = rdf4.fit(X2,Y_four)

# rdf5 = AdaBoostClassifier(n_estimators=12,learning_rate=0.5,random_state=1)
# rdf5 = rdf5.fit(X2,Y_five)

# rdf6 = AdaBoostClassifier(n_estimators=12,learning_rate=0.5,random_state=1)
# rdf6 = rdf6.fit(X2,Y_six)


# In[ ]:




