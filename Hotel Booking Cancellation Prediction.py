#!/usr/bin/env python
# coding: utf-8

# ## Importing Relevant Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# ## Loading dataset using pandas library

# In[2]:


dataset = pd.read_csv('booking.csv')


# In[3]:


dataset


# ## Data Preprocessing

# In[4]:


dataset.describe(include = 'all')


# In[5]:


data = dataset.copy()
data.head()


# In[6]:


data = data.drop('Booking_ID', axis = 1)


# In[7]:


data.shape


# In[8]:


data.dtypes


# In[9]:


data.isnull().sum()


# In[10]:


data['booking status'] = data['booking status'].map({'Not_Canceled': 0, 'Canceled': 1})


# In[11]:


data.head()


# In[12]:


data.corr()['booking status']


# In[13]:


avarage_corr = data.corr()['booking status'].mean()
avarage_corr


# In[14]:


data.columns


# In[15]:


dropped_columns = []

for i in data[['number of adults', 'number of children', 'number of weekend nights',
       'number of week nights', 'car parking space',
               'lead time', 'repeated', 'P-C',
       'P-not-C', 'average price', 'special requests',
       'booking status']]:
    
    if abs(data.corr()['booking status'][i]) < avarage_corr:
        dropped_columns.append(i)
    
data.drop(dropped_columns, axis=1, inplace=True) 


# In[16]:


data


# In[17]:


data['date of reservation'] = pd.to_datetime(data['date of reservation'], errors='coerce')


# In[18]:


data.dtypes


# In[19]:


data.columns


# In[20]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data[[
    'lead time',
    'average price', 
    'special requests'
]]

vif=pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['Features'] = variables.columns
vif


# In[21]:


for i in data[[
    
    'lead time',
    'average price', 
    'special requests'
]]:
    
    sns.boxplot(x=data[i], data=data)
    plt.show()


# In[22]:


q1=data.quantile(0.25)
q3=data.quantile(0.75)
IQR=q3-q1
Lower=q1-1.5*IQR
Upper=q3+1.5*IQR


# In[23]:


for i in data[['lead time', 'average price', 'special requests']]:
    
    data[i] = np.where(data[i] > Upper[i], Upper[i],data[i])
    data[i] = np.where(data[i] < Lower[i], Lower[i],data[i])


# In[24]:


for i in data[['lead time', 'average price', 'special requests']]:
    
    sns.boxplot(x=data[i], data=data)
    plt.show()


# In[25]:


data = data.reset_index()


# In[26]:


data.head(5)


# ## Data Conversion using One-Hot Encoder 

# In[27]:


data = pd.get_dummies(data, drop_first = True)
data


# In[28]:


data.columns


# In[29]:


data = data[['index', 'lead time', 'average price', 'special requests',
       'date of reservation', 'type of meal_Meal Plan 2',
       'type of meal_Meal Plan 3', 'type of meal_Not Selected',
       'room type_Room_Type 2', 'room type_Room_Type 3',
       'room type_Room_Type 4', 'room type_Room_Type 5',
       'room type_Room_Type 6', 'room type_Room_Type 7',
       'market segment type_Complementary', 'market segment type_Corporate',
       'market segment type_Offline', 'market segment type_Online', 'booking status']]


# In[30]:


data.head(5)


# ## Data Scaling

# In[31]:


from sklearn.preprocessing import StandardScaler


# In[32]:


X = data.drop(['date of reservation', 'booking status'], axis=1)


# In[33]:


scaled = StandardScaler().fit_transform(X)


# In[34]:


scaled = pd.DataFrame(scaled, columns=X.columns)


# In[35]:


scaled.head(5)


# In[36]:


scaled['date of reservation'] = data['date of reservation']
scaled['booking status'] = data['booking status']


# In[37]:


scaled.head(5)


# In[38]:


data = scaled


# In[39]:


data.columns


# In[40]:


data = data[['index', 'lead time', 'average price', 'special requests',
       'date of reservation',
       'type of meal_Meal Plan 2', 'type of meal_Meal Plan 3',
       'type of meal_Not Selected', 'room type_Room_Type 2',
       'room type_Room_Type 3', 'room type_Room_Type 4',
       'room type_Room_Type 5', 'room type_Room_Type 6',
       'room type_Room_Type 7', 'market segment type_Complementary',
       'market segment type_Corporate', 'market segment type_Offline',
       'market segment type_Online', 'booking status']]


# In[41]:


data.head(5)


# ## Data Modeling

# In[42]:


X = data.drop(['date of reservation', 'booking status'], axis=1)
y = data['booking status']


# In[43]:


from sklearn.model_selection import train_test_split


# In[44]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[45]:


from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def evaluate(model, X_test, y_test):
    
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:,1]
    
    roc_score_test = roc_auc_score(y_test, y_prob_test)
    gini_score_test = roc_score_test*2-1
    
    y_pred_train = model.predict(X_train)
    y_prob_train = model.predict_proba(X_train)[:,1]
    
    roc_score_train = roc_auc_score(y_train, y_prob_train)
    gini_score_train = roc_score_train*2-1
    
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred_test)
    
    accuracy_score_test = metrics.accuracy_score(y_test, y_pred_test)
    accuracy_score_train = metrics.accuracy_score(y_train, y_pred_train)
    
    print('Model Performance:')

    print('Gini Score for Test:', gini_score_test*100)
    
    print('Gini Score for Train:', gini_score_train*100)
    
    print('Accuracy Score for Test:', accuracy_score_test*100)
    
    print('Accuracy Score for Train:', accuracy_score_train*100)
    
    print('Confusion Matrix:', confusion_matrix)


# ### Random Forest Classifier

# In[46]:


from sklearn.ensemble import RandomForestClassifier


# In[47]:


rfc_base=RandomForestClassifier()
rfc_base.fit(X_train, y_train)


# In[48]:


result_rfc_base=evaluate(rfc_base, X_test, y_test)


# In[49]:


y_prob = rfc_base.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# ## Applying SelectFromModel

# In[50]:


from sklearn.feature_selection import SelectFromModel


# In[51]:


sfm = SelectFromModel(rfc_base)
sfm.fit(X_train, y_train)


# In[52]:


selected_feature= X.columns[(sfm.get_support())]
selected_feature


# In[53]:


feature_scores = pd.Series(rfc_base.feature_importances_, index=X.columns).sort_values(ascending=False)

feature_scores


# In[54]:


X_train=X_train[['index', 'lead time', 'average price', 'special requests']]
X_test=X_test[['index', 'lead time', 'average price', 'special requests']]


# In[55]:


X_train.head()


# In[56]:


X_test.head()


# In[57]:


rfc_importance=RandomForestClassifier()
rfc_importance.fit(X_train, y_train)


# In[58]:


result_rfc_importance = evaluate(rfc_importance, X_test, y_test)


# In[59]:


y_prob = rfc_importance.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# ## Hyperparameter Tuning

# In[60]:


from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

min_samples_split = [2, 5, 10]

min_samples_leaf = [1, 2, 4]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[61]:


rfc_randomized = RandomizedSearchCV(estimator = rfc_importance, param_distributions = random_grid, 
                                    n_iter = 10, 
                                    cv = 5, 
                                    verbose=1, 
                                    random_state=42, 
                                    n_jobs = -1)

rfc_randomized.fit(X_train, y_train)


# In[62]:


result_rfc_randomized=evaluate(rfc_randomized, X_test, y_test)


# In[64]:


y_prob = rfc_randomized.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# ## Univariate Analysis

# In[65]:


variables= []
train_Gini=[]
test_Gini=[]

for i in X_train.columns:
    X_train_single=X_train[[i]]
    X_test_single=X_test[[i]]
    
    rfc_randomized.fit(X_train_single, y_train)
    y_prob_train_single=rfc_randomized.predict_proba(X_train_single)[:, 1]
    
    
    roc_prob_train=roc_auc_score(y_train, y_prob_train_single)
    gini_prob_train=2*roc_prob_train-1
    
    
    rfc_randomized.fit(X_test_single, y_test)
    y_prob_test_single=rfc_randomized.predict_proba(X_test_single)[:, 1]
    
    
    roc_prob_test=roc_auc_score(y_test, y_prob_test_single)
    gini_prob_test=2*roc_prob_test-1
    
    
    variables.append(i)
    train_Gini.append(gini_prob_train)
    test_Gini.append(gini_prob_test)
    

df = pd.DataFrame({'Variable': variables, 'Train Gini': train_Gini, 'Test Gini': test_Gini})

df= df.sort_values(by='Test Gini', ascending=False)

df   

