#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv("data.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS'].value_counts()


# In[6]:


housing.describe()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


#for plotting histogram
#import matplotlib.pyplot as plt
#housing.hist(bins=50, figsize=(20,15))
#plt.show()


# # Train_Test Splitting(Manually)

# In[9]:


import numpy as np

#def split_train_test(data,test_ratio):
    #np.random.seed(42)
    #shuffled =np.random.permutation(len(data))
    #test_set_size=int(len(data)*test_ratio)
    #test_indices=shuffled[:test_set_size]
    #train_indices=shuffled[test_set_size:]
    #return data.iloc[train_indices], data.iloc[test_indices]


# In[10]:


#train_set,test_set=split_train_test(housing,0.2)


# In[11]:


#print(f"Rows in train Sets: {len(train_set)} \nRows in Test Sets : {len(test_set)}\n")


# # Train Test Split With sklearn

# In[12]:


from sklearn.model_selection import train_test_split
train_set,test_set = train_test_split(housing,test_size=0.2,random_state=42)
print(f"Rows in train Sets: {len(train_set)} \nRows in Test Sets : {len(test_set)}\n")


# In[13]:


from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing, housing["CHAS"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[14]:


strat_test_set


# In[15]:


strat_train_set


# In[16]:


housing=strat_train_set.copy()


# # Looking For Correlation

# In[17]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[18]:


from pandas.plotting import scatter_matrix
attributes = ["MEDV","RM","ZN","LSTAT","PTRATIO"]
scatter_matrix(housing[attributes],figsize=(12,8))


# In[19]:


housing.plot(kind='scatter',x="RM",y="MEDV",alpha=1)


# # Trying Out Atrributes Combination

# In[20]:


housing["TPM"]=housing["TAX"]/housing["RM"]
print(housing["TPM"])


# In[21]:


housing.head()


# In[22]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[23]:


housing.plot(kind='scatter',x="TPM",y="MEDV",alpha=1)


# In[24]:


housing = strat_train_set.drop("MEDV",axis=1)
housing_lable = strat_train_set["MEDV"].copy()


# # MISSING ATTRIBUTES

# In[25]:


#1. get rid of the missing data points
#2.get rid of the whole attribute
#3. set the value to some value(0,mean,median)


# In[26]:


a = housing.dropna(subset=["RM"])                          #option 1
a.shape


# In[27]:


housing.drop('RM', axis=1).shape                           #option 2 no RM coulom


# In[28]:


median = housing["RM"].median()                            #option 3 getting median


# In[29]:


housing["RM"].fillna(median)


# In[30]:


housing.shape


# In[31]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(housing)


# In[32]:


imputer.statistics_


# In[33]:


x = imputer.transform(housing)


# In[34]:


housing_tr=pd.DataFrame(x,columns=housing.columns)


# In[35]:


housing_tr.describe()


# # Creating Pipeline 

# In[36]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("stdscaled",StandardScaler()),
])


# In[37]:


housing_num_tr = my_pipeline.fit_transform(housing)
housing_num_tr


# # Desired model selection

# In[38]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model= DecisionTreeRegressor()
model= RandomForestRegressor()
#model = LinearRegression()
model.fit(housing_num_tr,housing_lable)


# In[39]:


few_data = housing.iloc[:5]


# In[40]:


few_lable = housing_lable.iloc[:5]


# In[41]:


prepared_data = my_pipeline.transform(few_data)


# In[42]:


model.predict(prepared_data)


# In[43]:


list(few_lable)


# In[44]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_lable,housing_predictions)
rmse=np.sqrt(mse)
rmse


# Using Cross validation

# In[45]:


from sklearn.model_selection import cross_val_score
scrs = cross_val_score(model,housing_tr,housing_lable,scoring='neg_mean_squared_error',cv=10)
rmse_scrs=np.sqrt(-scrs)


# In[46]:


rmse_scrs


# In[47]:


def print_scores(scrs):
    print("Scores : " ,scrs)
    print("Mean : ",scrs.mean())
    print("Standard Deviation : ", scrs.std())


# In[48]:


print_scores(rmse_scrs)


# # Saving The model 

# In[49]:


from joblib import dump,load
dump(model,'Dragon.joblib')


# # Testing the model on test data

# In[50]:


X_test = strat_test_set.drop("MEDV",axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test,final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse
#print(final_predictions,list(Y_test))


# In[ ]:




