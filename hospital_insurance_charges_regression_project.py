#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor


# In[2]:


df=pd.read_csv(r"C:\Users\HP\Downloads\insurance.csv")


# In[3]:


df.head()


# In[4]:


#data set shape
df.shape


# In[5]:


#data set information
df.info()


# In[6]:


#checking null values
df.isnull().sum()


# In[7]:


#checking duplicate values
df.duplicated().sum()


# In[8]:


#dropping the duplicate values
df.drop_duplicates(inplace=True)


# In[9]:


df.duplicated().sum()


# In[10]:


df.describe()


# In[11]:


#unique values in categorical columns
for i in df.select_dtypes(include='object'):
    print(f'{i} has {df[i].nunique()} unique values: {df[i].unique()}')


# In[12]:


#categorical_columns
cat_cols=df.select_dtypes(include='object').columns
cat_cols


# In[13]:


num_cols=df.select_dtypes(include=['float64','int64']).columns
num_cols


# In[14]:


#plotting box plot for outliers
for i in num_cols:
    sns.boxplot(x=df[i])
    plt.show()


# In[45]:


#bar plot for bmi vs charges
sns.scatterplot(data=df,x='bmi',y='charges')
plt.show()


# # From the above plot we can say that there is positive correlation but not strong, There are a lot of variations; even people with medium BMI have high charges, suggesting other factors influence cost

# In[17]:


ax=sns.barplot(data=df,x='smoker',y='charges')
ax.bar_label(ax.containers[0])
plt.show()


# # From the above bar plot we can say that Smokers pay significantly higher charges compared to non-smokers. Smoking status is a key predictor in medical insurance cost.

# In[18]:


#line plot for age vs charges
sns.lineplot(data=df,x='age',y='charges')
plt.show()


# # Positive Trend: As age increases, insurance charges also increase. This makes sense — older people are generally at higher medical risk

# In[19]:


#bar plot for children vs charges
sns.barplot(data=df,x='children',y='charges')
plt.show()


# # Families with 2 or 3 children seem to have the highest average charges.

# In[20]:


#bar plot for region vs charges
sns.barplot(data=df,x='region',y='charges')
plt.show()


# # Southeast region has the highest average charges, likely due to a higher proportion of smokers, older individuals, or higher BMI.
# 
# 

# In[21]:


#need to encode the categorical columns
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
for i in cat_cols:
    df[i]=encoder.fit_transform(df[i])
df.head()    


# In[22]:


#identifying dependent and independent features
x=df.drop('charges',axis=1)
y=df['charges']


# In[23]:


#splitting the data into train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[24]:


#scaling the values
scaler=MinMaxScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)


# In[25]:


x_train_scaled


# In[26]:


x_test_scaled


# In[27]:


#model building
model=LinearRegression()
model.fit(x_train_scaled,y_train)


# In[28]:


y_pred=model.predict(x_test_scaled)


# In[29]:


mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
rmse=np.sqrt(mse)


# In[30]:


mse


# In[31]:


r2


# In[32]:


rmse


# In[ ]:


np.mean(df['charges'])


# In[36]:


y_train_pred=model.predict(x_train_scaled)
r2_train_score=r2_score(y_train,y_train_pred)
r2_train_score


# In[37]:


#comparing training and testing accuracy
print('training accuracy: ',r2_train_score)
print('testing accuracy: ',r2)


# In[44]:


#Since training and testing R2 scores are close, there is no sign of overfitting


# In[39]:


model2=RandomForestRegressor(n_estimators=100,random_state=42)
model2.fit(x_train_scaled,y_train)
y_pred2=model2.predict(x_test_scaled)
mse=mean_squared_error(y_test,y_pred2)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred2)


# In[40]:


print('mse: ',mse)
print('rmse: ',rmse)
print('r2: ',r2)


# In[41]:


import xgboost as xgb
from xgboost import XGBRegressor
model3=XGBRegressor(n_estimators=100,learning_rate=0.1, max_depth=5,random_state=42)
model3.fit(x_train_scaled,y_train)
y_pred3=model3.predict(x_test_scaled)
mse=mean_squared_error(y_test,y_pred3)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred3)


# In[42]:


print('mse: ',mse)
print('rmse: ',rmse)
print('r2: ',r2)


# In[43]:


print("Model Comparison:")
print(f"Linear Regression R2: {r2_score(y_test, y_pred):.4f}")
print(f"Random Forest R2:     {r2_score(y_test, y_pred2):.4f}")
print(f"XGBoost R2:           {r2_score(y_test, y_pred3):.4f}")


# In[46]:


results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
    'R2 Score': [
        r2_score(y_test, y_pred),
        r2_score(y_test, y_pred2),
        r2_score(y_test, y_pred3)
    ],
    'RMSE': [
        np.sqrt(mean_squared_error(y_test, y_pred)),
        np.sqrt(mean_squared_error(y_test, y_pred2)),
        np.sqrt(mean_squared_error(y_test, y_pred3))
    ]
})
print(results)


# In[ ]:




