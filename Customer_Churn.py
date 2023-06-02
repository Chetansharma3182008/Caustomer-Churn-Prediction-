#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mutual_info_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[2]:


df = pd.read_csv(r"D:\Python\Telco-Customer-Churn.csv")


# In[3]:


df


# In[4]:


df.info()


# In[5]:


for column in df.columns:
    print('Column: {} - Unique Values: {}'.format(column, df[column].unique()))


# In[6]:


df['TotalCharges']=pd.to_numeric(df['TotalCharges'], errors='coerce')


# In[7]:


df[df['TotalCharges'].isnull()]


# In[8]:


df.dropna(inplace=True)


# In[9]:


df.drop(columns='customerID',inplace=True)


# In[10]:


df.PaymentMethod.unique()


# In[11]:


df['PaymentMethod']=df['PaymentMethod'].str.replace('(automatic)','',regex=False)


# In[12]:


df.PaymentMethod.unique()


# In[13]:


fig=plt.figure(figsize=(10,6))
ax=fig.add_subplot(111)

prop_response=df['Churn'].value_counts(normalize=True)

prop_response.plot(kind='bar',ax=ax,color=['springgreen','salmon'])

ax.set_title('Proportion of observations of the response variable',fontsize=18,loc='left')
ax.set_xlabel('churn',fontsize=14)
ax.set_ylabel('proportion of observations',fontsize=14)
ax.tick_params(rotation='auto')

spine_names=('top','right','bottom','left')
for spine_name in spine_names:
    ax.spines[spine_name].set_visible(False)


# In[14]:


def percentage_stacked_plot(columns_to_plot, super_title):
    
    number_of_columns = 2
    number_of_rows= math.ceil(len(columns_to_plot)/2)
    
    fig=plt.figure(figsize=(12,5 * number_of_rows))
    fig.suptitle(super_title,fontsize=22, y=.95)
    
    for index,column in enumerate(columns_to_plot, 1):
        ax=fig.add_subplot(number_of_rows , number_of_columns , index)
    
        prop_by_independent = pd.crosstab(df[column],df['Churn']).apply(lambda x: x/x.sum()*100 , axis=1)
        prop_by_independent.plot(kind='bar',ax=ax ,stacked =True , rot=0, color=['springgreen','salmon'])
    
        ax.legend(loc="upper right",bbox_to_anchor=(0.62,0.5,0.5,0.5),title='Churn',fancybox=True)
        ax.set_title('Proportion of observation by '+column, fontsize=16 ,loc='left')
        ax.tick_params(rotation = 'auto')
    
        spine_names=('top','right','bottom','left')
    
        for spine_name in spine_names:
            ax.spines[spine_name].set_visible(False)


# In[15]:


demographic_columns = ['gender','SeniorCitizen','Partner','Dependents']

percentage_stacked_plot(demographic_columns,'Demograpic Information')


# In[16]:


account_columns = ['Contract' , 'PaperlessBilling', 'PaymentMethod']
percentage_stacked_plot(account_columns, 'Customer Account Information')


# In[17]:


def histogram_plots(columns_to_plot, super_title):
    number_of_columns = 2
    number_of_rows= math.ceil(len(columns_to_plot)/2)
    
    fig=plt.figure(figsize=(12,5 * number_of_rows))
    fig.suptitle(super_title , fontsize=22 , y=.95)
    
    for index , column in enumerate(columns_to_plot,1):
        
        ax=fig.add_subplot(number_of_rows, number_of_columns, index)
        
        df[df['Churn']=='No'][column].plot(kind='hist',ax=ax,density=True,alpha=0.5,color='springgreen',label='No')
        df[df['Churn']=='Yes'][column].plot(kind='hist',ax=ax,density=True,alpha=0.5,color='salmon',label='Yes')
        
        ax.legend(loc="upper right", bbox_to_anchor=(0.5,0.5,0.5,0.5), title='Churn',fancybox=True)
        ax.tick_params(rotation='auto')
        
        spine_names=('top','right','bottom','left')
        for spine_name in spine_names:
            ax.spines[spine_name].set_visible(False)
            
account_columns_numeric =['tenure' , 'MonthlyCharges', 'TotalCharges']
histogram_plots(account_columns_numeric, 'Customer Account Information')
    


# In[18]:


services_columns = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

percentage_stacked_plot(services_columns, 'Services Information')


# In[19]:


def compute_mutual_information(categorical_serie):
    return mutual_info_score(categorical_serie, df.Churn)

categorical_variables = df.select_dtypes(include=object).drop('Churn' ,axis=1)
feature_importance = categorical_variables.apply(compute_mutual_information).sort_values(ascending=False)

print(feature_importance)


# In[20]:


df_transformed = df.copy()

label_encoding_columns = ['gender','Partner','Dependents','PaperlessBilling','PhoneService','Churn']

for column in label_encoding_columns:
    if column == 'gender':
        df_transformed[column]=df_transformed[column].map({'Female' : 1, 'Male' : 0})
    else:
        df_transformed[column]=df_transformed[column].map({'Yes' : 1 , 'No' : 0})
        


# In[21]:


one_hot_encoding_columns = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
df_transformed = pd.get_dummies(df_transformed , columns = one_hot_encoding_columns)


# In[22]:


min_max_columns =['tenure','MonthlyCharges','TotalCharges']

for column  in min_max_columns:
    min_column =df_transformed[column].min()
    max_column = df_transformed[column].max()
    
    df_transformed[column] = (df_transformed[column] - min_column) / (max_column - min_column)


# In[23]:


X= df_transformed.drop(columns='Churn')
y=df_transformed.loc[:,'Churn']

print(X.columns)
print(y.name)


# In[24]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=40,shuffle=True)


# In[25]:


def create_models(seed=2):
    
    models =[]
    models.append(('dummy_classifier',DummyClassifier(random_state=seed , strategy ='most_frequent')))
    models.append(('k_nearest_neighbors',KNeighborsClassifier()))
    models.append(('logistic_regression', LogisticRegression(random_state=seed)))
    models.append(('support_vector_machines' ,SVC(random_state=seed)))
    models.append(('random_forest', RandomForestClassifier(random_state=seed)))
    models.append(('gradient_boosting', GradientBoostingClassifier(random_state=seed)))
    
    return models
models=create_models()


# In[26]:


results = []
names = []
scoring = 'accuracy'

for name, model in models :
    model.fit(X_train,y_train).predict(X_test)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test,predictions)
    results.append(accuracy)
    names.append(name)
    print('Classifier : {}, Accuracy: {}'.format(name,accuracy))


# In[27]:


from sklearn.model_selection import RandomizedSearchCV


# In[28]:


grid_parameters = {'n_estimators': [80, 90, 100, 110, 115, 120],'max_depth': [3, 4, 5, 6],'max_features': [None, 'auto', 'sqrt', 'log2'], 'min_samples_split': [2, 3, 4, 5]}



random_search = RandomizedSearchCV(estimator=GradientBoostingClassifier(),param_distributions=grid_parameters,cv=5, n_iter=150, n_jobs=-1)


random_search.fit(X_train, y_train)


print(random_search.best_params_)


# In[29]:


from sklearn.metrics import confusion_matrix


# In[30]:


random_search_predictions = random_search.predict(X_test)
confusion_matrix = confusion_matrix(y_test, random_search_predictions)

confusion_matrix


# In[31]:


from sklearn.metrics import classification_report


# In[32]:


print(classification_report(y_test, random_search_predictions))


# In[33]:


accuracy_score(y_test, random_search_predictions)

