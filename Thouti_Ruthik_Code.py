#!/usr/bin/env python
# coding: utf-8

# # Importing the required libraries

# In[1]:


import pandas as pd 
import numpy as np
import datetime as dt
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
plt.style.use('seaborn-darkgrid')
  
# To ignore warnings 
import warnings
warnings.filterwarnings("ignore")


# # Loading the dataset

# In[2]:


dfX = pd.read_csv('stocknet_trn_data.csv')
dfY = pd.read_csv('stocknet_trn_data_targets.csv', names = ['label', 'close'])
dfT = pd.read_csv('stocknet_tst_data.csv')


# In[3]:


dfX.head()


# In[4]:


dfY.head()


# In[5]:


dfT.head()


# In[6]:


dfX[dfX.isna().any(axis=1)]


# In[7]:


dfY[dfY.isna().any(axis=1)]


# In[8]:


dfT[dfT.isna().any(axis=1)]


# In[9]:


dfX.dropna(how='any', inplace=True)


# In[10]:


dfY.dropna(how='any', inplace=True)


# In[11]:


dfX.isna().sum()


# In[12]:


dfY.isna().sum()


# In[13]:


dfX.head()


# In[14]:


dfY.head()


# In[15]:


df = pd.concat([dfX, dfY], axis=1, join='inner')


# In[16]:


df.head()


# In[17]:


df.isna().sum()


# In[18]:


X = df[['Open', 'High', 'Low', 'Volume']]


# In[19]:


X.head()


# In[20]:


y = df.iloc[:, 5:6]


# In[21]:


y.head()


# In[22]:


X.shape


# In[23]:


y.shape


# In[26]:


plt.plot('Open', 'High', data=df)


# In[27]:


plt.plot('High', 'Low', data=df)


# In[28]:


plt.plot('High', 'Volume', data=df)


# In[29]:


plt.plot('Open', 'close', data=df)


# 
# # PHASE_2

# # Linear Regression

# In[30]:


X['Open'].plot(figsize=(24,6))


# In[31]:


dfX.info()


# In[33]:


dfY.info()


# In[34]:


from sklearn.preprocessing import MinMaxScaler
sc_X = MinMaxScaler()
sc_y = MinMaxScaler()

X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 0)


# In[36]:


X_train.shape


# In[37]:


X_test.shape


# In[38]:


y_train.shape


# In[39]:


y_test.shape


# In[40]:


# Fitting the model
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,y_train)


# In[41]:


#Return the coefficient of determination of the prediction.
LR.score(X_test,y_test,sample_weight=None)


# In[42]:


pred=LR.predict(X_test) #Predict using the linear model.
pred


# In[43]:


#checking predicted y and labeled y using a scatter plot.
plt.scatter(y_test,pred)


# In[44]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math as math
print('Model Coefficients:', LR.coef_)
print('Model intercept:', LR.intercept_)
print('Mean Squared Error:', mean_squared_error(y_test, pred))
print('Mean Absolute Error:', mean_absolute_error(y_test, pred))
print('Root Mean Squared Error:', math.sqrt(mean_squared_error(y_test, pred)))
print('R_squared Value:', r2_score(y_test, pred))


# In[ ]:





# In[ ]:





# # Decision Tree Regressor

# In[48]:


X


# In[49]:


y


# In[50]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,random_state=0)


# In[51]:


X_train.shape


# In[52]:


y_train.shape


# In[53]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

param_grid={'criterion':['squared_error','friedman_mse','absolute_error','poisson']}
grid=GridSearchCV(DecisionTreeRegressor(max_depth=10,min_samples_split=8),param_grid)
grid.fit(X_train,y_train)


# In[54]:


print(grid.best_params_)
print(grid.score(X_test,y_test))


# In[55]:


from sklearn.tree import DecisionTreeRegressor
grid = DecisionTreeRegressor(criterion="poisson",max_depth=10,min_samples_split=8,random_state=0)
grid.fit(X_train,y_train)
pred_dt=grid.predict(X_test)


# In[56]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math as math
print('Mean Squared Error:', mean_squared_error(y_test, pred_dt))
print('Mean Absolute Error:', mean_absolute_error(y_test, pred_dt))
print('Root Mean Squared Error:', math.sqrt(mean_squared_error(y_test, pred_dt)))
print('R_squared Value:', r2_score(y_test, pred_dt))


# In[ ]:





# In[ ]:





# # Random Forest Regressor

# In[70]:


X


# In[71]:


y


# In[72]:


X_train.shape


# In[73]:


y_train.shape


# In[74]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

param_grid_rf={'criterion':['squared_error','friedman_mse','poisson']}
grid_rf=GridSearchCV(RandomForestRegressor(n_estimators = 90,max_depth=10,min_samples_split=5),param_grid_rf)
grid_rf.fit(X_train,y_train)


# In[75]:


print(grid_rf.best_params_)
print(grid_rf.score(X_test,y_test))


# In[76]:


from sklearn.ensemble import RandomForestRegressor
regressor_rf = RandomForestRegressor(criterion='friedman_mse',n_estimators = 90,max_depth=10,min_samples_split=5)
regressor_rf.fit(X_train, y_train) 
pred_rf = regressor_rf.predict(X_test)


# In[77]:


pred_rf


# In[78]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math as math
print('Mean Squared Error:', mean_squared_error(y_test, pred_rf))
print('Mean Absolute Error:', mean_absolute_error(y_test, pred_rf))
print('Root Mean Squared Error:', math.sqrt(mean_squared_error(y_test, pred_rf)))
print('R_squared Value:', r2_score(y_test, pred_rf))


# In[ ]:





# In[ ]:





# # Support Vector Regression

# In[79]:


X


# In[80]:


y


# In[81]:


X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)
#performing the train test slpit


# In[82]:


from sklearn.model_selection import GridSearchCV

param_grid={'C':[0.1,1,10,50,100],'kernel':['rbf','linear'],'gamma':[0.005,0.01,0.1,1]}
grid_svr=GridSearchCV(SVR(),param_grid)
grid_svr.fit(X_train,y_train)


# In[83]:


print(grid_svr.best_params_)
print(grid_svr.score(X_test,y_test))


# In[121]:


#svr_lin  = SVR(kernel='linear', C=100)
svr_rbf = SVR(kernel='rbf',C = 1, gamma=1)
svr_rbf.fit(X_train, y_train)
pred_svr = svr_rbf.predict(X_test)


# In[122]:


pred_svr


# In[123]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math as math
print('Mean Squared Error:', mean_squared_error(y_test, pred_svr))
print('Mean Absolute Error:', mean_absolute_error(y_test, pred_svr))
print('Root Mean Squared Error:', math.sqrt(mean_squared_error(y_test, pred_svr)))
print('R_squared Value:', r2_score(y_test, pred_svr))


# In[ ]:





# In[ ]:





# # KNN

# In[96]:


X


# In[97]:


y


# In[98]:


rmse_val = [] #to store rmse values for different k
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, y_train)  #fit the model
    pred=model.predict(X_test) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    print('RMSE value for k= ' , K , 'is:', error)


# In[99]:


#plotting the rmse values against k values
curve = pd.DataFrame(rmse_val) #elbow curve 
curve.plot()


# In[100]:


#varifying the same using gridsearch

from sklearn.model_selection import GridSearchCV
params = {'n_neighbors':[2,3,4,5,6,7,8,9]}

knn = neighbors.KNeighborsRegressor()

model = GridSearchCV(knn, params, cv=5)
model.fit(X_train,y_train)


# In[101]:


print(model.best_params_)
print(model.score(X_test,y_test))


# In[102]:


#At k = 3, approx, we can see we have the lowest rmse


# In[103]:


knn = neighbors.KNeighborsRegressor(n_neighbors = 3)
knn.fit(X_train, y_train)  #fit the model
pred_knn=model.predict(X_test) #make prediction on test set


# In[104]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math as math
print('Mean Squared Error:', mean_squared_error(y_test, pred_knn))
print('Mean Absolute Error:', mean_absolute_error(y_test, pred_knn))
print('Root Mean Squared Error:', math.sqrt(mean_squared_error(y_test, pred_knn)))
print('R_squared Value:', r2_score(y_test, pred_knn))


# In[ ]:





# In[ ]:





# # Errors of all models

# In[ ]:


#After comparing all the models we found less Error in Linear Regression,so we can use linear regression to predict the stock price.
# The following are the errors of models that we have used
#Errors of linear regression 
Mean Squared Error: 6.105198492075248e-08
Mean Absolute Error: 2.0368883495576596e-05
Root Mean Squared Error: 0.0002470869986882201
R_squared Value: 0.999990296851084
    
#Errors of Decision tree regressor
Mean Squared Error: 1.474497034680812e-07
Mean Absolute Error: 5.038266703567706e-05
Root Mean Squared Error: 0.0003839918013032065
R_squared Value: 0.9999765654395639

#Errors of Random forest regressor
Mean Squared Error: 1.0254544685305473e-07
Mean Absolute Error: 3.001035930635447e-05
Root Mean Squared Error: 0.00032022718006605047
R_squared Value: 0.9999837021885077
    
#Errors of Support vector regressor
Mean Squared Error: 0.009457716707986598
Mean Absolute Error: 0.09680778112363161
Root Mean Squared Error: 0.0972507928398869
R_squared Value: -0.5031392303088582 
    
#Errors of KNeighborsRegressor
Mean Squared Error: 9.721230450810448e-08
Mean Absolute Error: 3.166236517712933e-05
Root Mean Squared Error: 0.00031178887810200107
R_squared Value: 0.999984549798531


# In[93]:


#Applying the best model Linear regression


# In[141]:


dfT = pd.read_csv('stocknet_tst_data.csv')


# In[142]:


dfT


# In[143]:


Final_predictions=LR.predict(dfT) #Predict using the linear model.
Final_predictions


# In[138]:


D = dfY[['label']]
d = D.iloc[:10860,:]
d


# In[144]:


c = pd.DataFrame(Final_predictions, columns =['close'])
c


# In[145]:


DF = pd.concat([d,c], axis=1, join='inner')
DF


# In[166]:


DF.to_csv('FINAL_PRED.csv') 


# In[ ]:





# In[ ]:




