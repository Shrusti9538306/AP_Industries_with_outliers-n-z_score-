#!/usr/bin/env python
# coding: utf-8

# In[134]:


import pandas as pd
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("AP Industries_Web_Scrapping.csv")
df.head()


# In[135]:


df['cap_Latitude'] = 17.6868
df['cap_Longitude'] = 83.2185


from haversine import haversine

loc1 = list(zip(df.Latitude, df.Longitude))
loc2 = list(zip(df.cap_Latitude, df.cap_Longitude))
df['distance_from_cap'] = ''
for ind in df.index: 
     df['distance_from_cap'].values[ind] = haversine(loc1[ind], loc2[ind])
#df.head()

list(df)


# In[136]:


df['Category'] = df['Category'].replace('Large.png"               ', 'Large')
df['Category'] = df['Category'].fillna("NA")
print(df['Category'].unique())

df.groupby('Category').size()

# Replacing with NA values with micro (selected based on mode)
df['Category'] = df['Category'].replace('NA', 'Micro')
print(df['Category'].unique())

print(df['District Name'].unique())

df.groupby('Sector Name').size()

# Pulp industry consists of packaging industry and other service providing shops, so replacing with service
df.loc[df['Activity Name'] == 'Pulp', 'Sector Name'] = "SERVICE"

# Plastic industry consists manufacturing of plastic products, so replacing with Engineering.
df.loc[df['Activity Name'] ==  'Plastics', 'Sector Name'] = "ENGINEERING"

print(df['Activity Name'].unique())

df['Activity Name'] = df['Activity Name'].replace([' Total Workers ', ' Industry as per pollution Index Category '], 'AUTOMOBILE SERVICING')
print((df['Activity Name'].unique()))

df['Activity Name'] = df['Activity Name'].replace('NA', 'AUTOMOBILE SERVICING')
print(df['Activity Name'].unique())

df.groupby('Activity Name').size()

print(df['Pollution Index Category'].unique())
df.groupby('Pollution Index Category').size()

df['Pollution Index Category'] = df['Pollution Index Category'].replace(' Total Workers ', 'NA')
df['Pollution Index Category'] = (df['Pollution Index Category'].fillna("Green"))
print(df['Pollution Index Category'].unique())

df.groupby('Pollution Index Category').size()
df['Pollution Index Category'] = df['Pollution Index Category'].replace('NA', 'Green')
print(df['Pollution Index Category'].unique())

df['Total Workers'].describe().round(1)

# replacing with nan values with median (2)
df['Total Workers'] = df['Total Workers'].fillna(2)
print(df['Total Workers'])

# Removing outliers
#df.loc[df['Total Workers']> 9, 'Total Workers'] = 9
#df['Total Workers'].describe().round(1)

# Removing outliers
df.loc[df['distance_from_cap']> 1070, 'distance_from_cap'] = 1070
df['distance_from_cap'].describe().round(1)


# In[4]:


df.isnull().sum()


# In[137]:


df['distance_from_cap']= pd.to_numeric(df['distance_from_cap'])
df['distance_from_cap'].describe().round(2)

df.info()


# In[138]:


from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['Category']  = LE.fit_transform(df['Category'])
#df['Category'].dtype

df['District Name']  = LE.fit_transform(df['District Name'])                                    

df['Sector Name']  = LE.fit_transform(df['Sector Name'])

df['Activity Name']  = LE.fit_transform(df['Activity Name'])

df['Pollution Index Category'] = LE.fit_transform(df['Pollution Index Category']).astype(float)
#df['Pollution Index Category'].dtype

df.head()


# In[139]:


df1 = df.drop(['Unnamed: 0', 'Industry Name', 'Latitude', 'Longitude', 'cap_Latitude', 'cap_Longitude', 'Activity Name'], axis=1)
df1.head()


# In[9]:


#plt.figure(figsize=(70,70))
sns.heatmap(df1.corr(), annot = True, cmap = 'coolwarm' ,fmt = '.0%')


# In[140]:


x = df1.drop([ 'Total Workers'], axis=1)

y = df1['Total Workers']
x.head()


# In[141]:


from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(x, y, 
                                                    test_size=0.25, 
                                                    random_state=50)
X1_train.shape, X1_test.shape, y1_train.shape, y1_test.shape


# In[69]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

reg = LinearRegression()
reg = reg.fit(X1_train, y1_train)

lr_y_pred_train = reg.predict(X1_train)
lr_y_pred1_test = reg.predict(X1_test)

lr_mse_train = mean_squared_error(y1_train, lr_y_pred_train)
print("MSE", lr_mse_train)

lr_mse_test = mean_squared_error(y1_test, lr_y_pred1_test)
print("MSE", lr_mse_test)

lr_r2_train = r2_score(y1_train, lr_y_pred_train)
print("r2_score", lr_r2_train)

lr_r2_test = r2_score(y1_test, lr_y_pred1_test)
print("r2_score", lr_r2_test)


# In[71]:


rid_r = Ridge(alpha = 1, random_state = 1).fit(X1_train, y1_train)

y_pred_train_rid = rid_r.predict(X1_train)
y_pred_test_rid = rid_r.predict(X1_test)

rid_mse_train = mean_squared_error(y1_train, y_pred_train_rid)
print("Mse of RR on Test data",rid_mse_train.round(3))

rid_mse_test = mean_squared_error(y1_test, y_pred_test_rid)
print("Mse of RR on Test data",rid_mse_test.round(3))

rid_r2_train = r2_score(y1_train, y_pred_train_rid)
print("R2_score of RR on Test data", rid_r2_train.round(3))

rid_r2_test = r2_score(y1_test, y_pred_test_rid)
print("R2_score of RR on Test data", rid_r2_test.round(3))


# In[10]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

dt = DecisionTreeRegressor()

dt.fit(X1_train, y1_train)

dt.tree_.node_count
dt.tree_.max_depth

print(f"Decision tree has {dt.tree_.node_count} nodes with maximum depth covered up to {dt.tree_.max_depth}")


# In[11]:


max_depth = [int(x) for x in np.linspace(start = 1, stop = 60, num = 60)]

min_samples_split = [2, 3, 4, 5, 6, 7, 8, 10, ]

min_samples_leaf = [1, 2, 5, 10]

random_state = [1, 5, 10, 20, 50]

tree_para = {
            'criterion':['mse'],
            'max_features' : ['auto'],
            'max_depth' : max_depth,
            'min_samples_split' : min_samples_split,
            'min_samples_leaf' : min_samples_leaf,
            'random_state' : random_state
            }


clf = GridSearchCV(dt, tree_para, cv=10)
clf.fit(X1_train, y1_train)

print(clf.best_score_)
print(clf.best_params_)


# In[25]:


dt1 = DecisionTreeRegressor(
                                criterion = 'mse',
                                min_samples_split = 2,
                                min_samples_leaf = 10,
                                max_features = 'auto',
                                max_depth = 5,
                                random_state = 1
                                )

dt1.fit(X1_train, y1_train)

dt_y_pred_train = dt1.predict(X1_train)
dt_y_pred_dt_test = dt1.predict(X1_test)

dt_mse_train = mean_squared_error(y1_train, dt_y_pred_train)
print("Mse of DT on Train data", dt_mse_train.round(3))

dt_mse_test = mean_squared_error(y1_test, dt_y_pred_dt_test)
print("Mse of DT on Test data", dt_mse_test.round(3))

dt_r2_train = r2_score(y1_train, dt_y_pred_train)
print("R2_score of DT on Train data", dt_r2_train.round(3))

dt_r2_test = r2_score(y1_test, dt_y_pred_dt_test)
print("R2_score of DT on Test data", dt_r2_test.round(3))

print(f"Decision tree has {dt.tree_.node_count} nodes with maximum depth covered up to {dt.tree_.max_depth}")


# In[28]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

max_features = ['auto']

max_depth = [int(x) for x in np.linspace(start = 1, stop = 15, num = 15)]

min_samples_split = [2, 3, 4, 5, 6, 7, 8, 10, ]

min_samples_leaf = [1, 2, 5, 10]

random_state = [1, 5, 10, 20, 50]

from sklearn.model_selection import RandomizedSearchCV

random_grid = {'n_estimators' : n_estimators,
               'max_features' : max_features,
               'max_depth' : max_depth,
               'min_samples_split' : min_samples_split,
               'min_samples_leaf' : min_samples_leaf,
               'random_state' : random_state
               }

print(random_grid)

rf = RandomizedSearchCV(estimator= RandomForestRegressor(), param_distributions = random_grid, scoring= 'neg_mean_squared_error', n_iter = 10, cv = 5, verbose=5, n_jobs=1)
rf

rf.fit(X1_train, y1_train)

print("Best score", rf.best_score_)
print("Best parameters", rf.best_params_)


# In[32]:


from sklearn.ensemble import RandomForestRegressor
rf1 = RandomForestRegressor(
                             n_estimators = 300, 
                             min_samples_split = 6,
                             min_samples_leaf = 1,
                             max_features = 'auto',
                             max_depth = 12,
                             random_state = 20
                            )

rfr = rf1.fit(X1_train, y1_train)

rf_Y_pred_train = rfr.predict(X1_train)

rf_Y_pred_test = rfr.predict(X1_test)

rf_mse_train = mean_squared_error(y1_train, rf_Y_pred_train)
print("Mse of RFR on Trained data", rf_mse_train.round(3))

rf_mse_test = mean_squared_error(y1_test, rf_Y_pred_test)
print("Mse of RFR on Test data", rf_mse_test.round(3))

rf_r2_train = r2_score(y1_train, rf_Y_pred_train)
print("R2_score of RFR on Trained data", rf_r2_train.round(3))

rf_r2_test = r2_score(y1_test, rf_Y_pred_test)
print("R2_score of RFR on Test data", rf_r2_test.round(3))


# In[36]:


from sklearn.ensemble import BaggingRegressor
bag = BaggingRegressor()

base_estimator = [dt1]

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

max_samples = [int(x) for x in np.linspace(start = 1, stop = 10, num = 10)]

max_features = ['auto']

random_state = [1, 5, 10, 20, 50]

from sklearn.model_selection import RandomizedSearchCV

random_grid = {
                'base_estimator': base_estimator,
                'n_estimators' : n_estimators,
                'max_samples' : max_samples,
                'random_state' : random_state
               }

print(random_grid)

bag = RandomizedSearchCV(estimator= BaggingRegressor(), param_distributions = random_grid, scoring= 'neg_mean_squared_error', n_iter = 10, cv = 5, verbose=5, n_jobs=1)
bag

bag.fit(X1_train, y1_train)

print("Best score", bag.best_score_)
print("Best parameters", bag.best_params_)


# In[37]:


from sklearn.ensemble import BaggingRegressor
bag = BaggingRegressor(
                        base_estimator = dt1, 
                        max_samples=8, 
                        n_estimators=1100, 
                        random_state = 10
                      )

bag.fit(X1_train, y1_train)

y_pred_train_bag = bag.predict(X1_train)
y_pred_test_bag = bag.predict(X1_test)

bag_mse_train = mean_squared_error(y1_train, y_pred_train_bag)
print("MSE of bagging on Training data", bag_mse_train)

bag_mse_test = mean_squared_error(y1_test, y_pred_test_bag)
print("MSE of bagging on Test data", bag_mse_test)

bag_r2_train = r2_score(y1_train, y_pred_train_bag)
print("R2_score of bagging on Train data", bag_r2_train.round(3))

bag_r2_test = r2_score(y1_test, y_pred_test_bag)
print("R2_score of bagging on Test data", bag_r2_test.round(3))


# In[43]:


from sklearn.linear_model import SGDRegressor

alpha = np.arange(0.001, 0.015,0.001)
sgd_rmse_train = []
sgd_rmse_test = []
sgd_r2_train = []
sgd_r2_test = []

for x in alpha:
    sgd = SGDRegressor(learning_rate='constant', eta0=x)
    sgd.fit(X1_train,y1_train)
    sgd_y_pred_train = sgd.predict(X1_train)
    sgd_y_pred_test = sgd.predict(X1_test)
    sgd_rmse_train.append(mean_squared_error(y1_train, sgd_y_pred_train))
    sgd_rmse_test.append(mean_squared_error(y1_test, sgd_y_pred_test))
    sgd_r2_train.append(r2_score(y1_train, sgd_y_pred_train))
    sgd_r2_test.append(r2_score(y1_test, sgd_y_pred_test))

print(sgd_rmse_train)
print(sgd_rmse_test)
print(sgd_r2_train)
print(sgd_r2_test)

err_train = pd.DataFrame(sgd_rmse_train)

err_test = pd.DataFrame(sgd_rmse_test).round(3)

acc_train = pd.DataFrame(sgd_r2_train).round(3)

acc_test = pd.DataFrame(sgd_r2_test).round(3)


# In[44]:


plt.figure(figsize=(15,6))
plt.plot(alpha, err_test)


# In[45]:


plt.plot(alpha, acc_test)


# In[68]:


sgd = SGDRegressor(alpha = 0.01, learning_rate='constant', eta0=0.004)
sgd.fit(X1_train, y1_train)

sgd_y_pred_train1 = sgd.predict(X1_train)

sgd_y_pred_test1 = sgd.predict(X1_test)

sgd_mse_train1 = mean_squared_error(y1_train, sgd_y_pred_train1)
print("MSE of XGB on Training data", sgd_mse_train1)

sgd_mse_test1 = mean_squared_error(y1_test, sgd_y_pred_test1)
print("MSE of XGB on Test data", sgd_mse_test1)

sgd_r2_train1 = r2_score(y1_train, sgd_y_pred_train1)
print("R2_score of XGB on Train data", sgd_r2_train1.round(3))


sgd_r2_test1 = r2_score(y1_test, sgd_y_pred_test1)
print("R2_score of XGB on Test data", sgd_r2_test1.round(3))


# In[53]:


import xgboost as xg
xgreg = xg.XGBRegressor()
xgreg.fit(X1_train, y1_train)

xgreg_y_pred_train = xgreg.predict(X1_train)

xgreg_y_pred_test = xgreg.predict(X1_test)

xgreg_mse_train = mean_squared_error(y1_train, xgreg_y_pred_train)
print("MSE of XGB on Training data", xgreg_mse_train)

xgreg_mse_test = mean_squared_error(y1_test, xgreg_y_pred_test)
print("MSE of XGB on Test data", xgreg_mse_test)

xgreg_r2_train = r2_score(y1_train, xgreg_y_pred_train)
print("R2_score of XGB on Train data", xgreg_r2_train.round(3))

xgreg_r2_test = r2_score(y1_test, xgreg_y_pred_test)
print("R2_score of XGB on Test data", xgreg_r2_test.round(3))


# In[64]:


from scipy.stats import uniform, randint

n_iter = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

#random_state = [1, 5, 10, 20, 50]

params = {
    "colsample_bytree": uniform(0.7, 0.3),
    "gamma": uniform(0, 0.5),
    "learning_rate": uniform(0.03, 0.3),
    "max_depth": randint(2, 6), 
    "n_estimators": randint(100, 150),
    "subsample": uniform(0.6, 0.4)
    
}

gridcv = RandomizedSearchCV(xgreg, param_distributions=params, n_iter = 100, random_state= 1, cv=3, verbose=1, n_jobs=1, return_train_score=True)

gridcv.fit(X1_train, y1_train)

print("Best score", gridcv.best_score_)
print("Best parameters", gridcv.best_params_)


# In[67]:


import xgboost as xg
xgreg1 = xg.XGBRegressor(
                        colsample_bytree = 0.7,
                        gamma = 0.3,
                        learning_rate = 0.1, 
                        max_depth = 4,
                        n_estimators = 118,
                        subsample = 0.97,
                        random_state=1,
                        n_iter=100,
                        verbose=1, 
                        n_jobs=1,
                        return_train_score=True
                        )

xgreg1.fit(X1_train, y1_train)

xgreg_y_pred_train1 = xgreg.predict(X1_train)

xgreg_y_pred_test1 = xgreg.predict(X1_test)

xgreg_mse_train1 = mean_squared_error(y1_train, xgreg_y_pred_train1)
print("MSE of XGB on Training data", xgreg_mse_train1)

xgreg_mse_test1 = mean_squared_error(y1_test, xgreg_y_pred_test1)
print("MSE of XGB on Test data", xgreg_mse_test1)

xgreg_r2_train1 = r2_score(y1_train, xgreg_y_pred_train1)
print("R2_score of XGB on Train data", xgreg_r2_train1.round(3))

xgreg_r2_test1 = r2_score(y1_test, xgreg_y_pred_test1)
print("R2_score of XGB on Test data", xgreg_r2_test1.round(3))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[107]:


plt.hist(df1['Total Workers'], bins = 20, rwidth = 0.5)


# In[108]:


plt.boxplot(df1['Total Workers'])


# In[142]:


df1['zscore'] = (df1['Total Workers'] - df1['Total Workers'].mean()) / df1['Total Workers'].std()


# In[143]:


df1.head()


# In[118]:


plt.boxplot(df1['zscore'])


# In[119]:


df1.describe()


# In[144]:


upper_zscore = df1.zscore.mean() + 3 * df1.zscore.std()
upper_zscore


# In[145]:


df1[df1.zscore > upper_zscore].shape


# In[146]:


upper_worker = df1['Total Workers'].mean() + 3 * df1['Total Workers'].std()
upper_worker


# In[147]:


df1[df1['Total Workers']> upper_worker].shape


# In[148]:


df1.loc[df1.zscore> 3, 'zscore'] = 3
df1.zscore.describe().round(1)


# In[128]:


plt.boxplot(df1['zscore'])


# In[129]:


plt.hist(df1['zscore'], bins = 20, rwidth = 0.5)


# In[149]:


z = df1.zscore
z


# In[150]:


sns.heatmap(df1.corr(), annot = True, cmap = 'coolwarm' ,fmt = '.0%')


# In[151]:


X2_train, X2_test, z_train, z_test = train_test_split(x, z, 
                                                    test_size=0.25, 
                                                    random_state=50)
X2_train.shape, X2_test.shape, z_train.shape, z_test.shape


# In[154]:


reg = LinearRegression()
reg = reg.fit(X1_train, z_train)

lr_z_pred_train = reg.predict(X1_train)
lr_z_pred1_test = reg.predict(X1_test)

lr_mse_train_z = mean_squared_error(z_train, lr_z_pred_train)
print("MSE of LR on Training data", lr_mse_train_z.round(3))

lr_mse_test_z = mean_squared_error(z_test, lr_z_pred1_test)
print("MSE of LR on Test data", lr_mse_test_z.round(3))

lr_r2_train_z = r2_score(z_train, lr_z_pred_train)
print("R2_score of LR on Train data", lr_r2_train_z.round(3))

lr_r2_test_z = r2_score(z_test, lr_z_pred1_test)
print("R2_score of LR on Test data", lr_r2_test_z.round(3))


# In[155]:


rid_z = Ridge(alpha = 1, random_state = 1).fit(X1_train, z_train)

z_pred_train_rid = rid_z.predict(X1_train)
z_pred_test_rid = rid_z.predict(X1_test)

rid_mse_train_z = mean_squared_error(z_train, z_pred_train_rid)
print("Mse of RR on Test data",rid_mse_train_z.round(3))

rid_mse_test_z = mean_squared_error(z_test, z_pred_test_rid)
print("Mse of RR on Test data",rid_mse_test_z.round(3))

rid_r2_train_z = r2_score(z_train, z_pred_train_rid)
print("R2_score of RR on Test data", rid_r2_train_z.round(3))

rid_r2_test_z = r2_score(z_test, z_pred_test_rid)
print("R2_score of RR on Test data", rid_r2_test_z.round(3))


# In[156]:


dt_z = DecisionTreeRegressor()

dt_z.fit(X1_train, z_train)

dt_z.tree_.node_count
dt_z.tree_.max_depth

print(f"Decision tree has {dt_z.tree_.node_count} nodes with maximum depth covered up to {dt_z.tree_.max_depth}")


# In[ ]:


max_depth = [int(x) for x in np.linspace(start = 1, stop = 60, num = 60)]

min_samples_split = [2, 3, 4, 5, 6, 7, 8, 10, ]

min_samples_leaf = [1, 2, 5, 10]

random_state = [1, 5, 10, 20, 50]

tree_para_z = {
            'criterion':['mse'],
            'max_features' : ['auto'],
            'max_depth' : max_depth,
            'min_samples_split' : min_samples_split,
            'min_samples_leaf' : min_samples_leaf,
            'random_state' : random_state
            }


clf_z = GridSearchCV(dt, tree_para_z, cv=10)
clf_z.fit(X1_train, z_train)

print(clf_z.best_score_)
print(clf_z.best_params_)


# In[ ]:


dt1_z = DecisionTreeRegressor(
                                criterion = 'mse',
                                min_samples_split = 2,
                                min_samples_leaf = 10,
                                max_features = 'auto',
                                max_depth = 5,
                                random_state = 1
                                )

dt1_z.fit(X1_train, z_train)

dt_z_pred_train = dt1_z.predict(X1_train)
dt_z_pred_test = dt1_z.predict(X1_test)

dt_mse_train_z = mean_squared_error(z_train, dt_z_pred_train)
print("Mse of DT on Train data", dt_mse_train_z.round(3))

dt_mse_test_z = mean_squared_error(z_test, dt_z_pred_test)
print("Mse of DT on Test data", dt_mse_test_z.round(3))

dt_r2_train_z = r2_score(z_train, dt_z_pred_train)
print("R2_score of DT on Train data", dt_r2_train_z.round(3))

dt_r2_test_z = r2_score(z_test, dt_z_pred_test)
print("R2_score of DT on Test data", dt_r2_test_z.round(3))

print(f"Decision tree has {dt1_z.tree_.node_count} nodes with maximum depth covered up to {dt1_z.tree_.max_depth}")


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf_z = RandomForestRegressor()

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

max_features = ['auto']

max_depth = [int(x) for x in np.linspace(start = 1, stop = 15, num = 15)]

min_samples_split = [2, 3, 4, 5, 6, 7, 8, 10, ]

min_samples_leaf = [1, 2, 5, 10]

random_state = [1, 5, 10, 20, 50]

from sklearn.model_selection import RandomizedSearchCV

random_grid_z = {'n_estimators' : n_estimators,
               'max_features' : max_features,
               'max_depth' : max_depth,
               'min_samples_split' : min_samples_split,
               'min_samples_leaf' : min_samples_leaf,
               'random_state' : random_state
               }

print(random_grid_z)

rf_z = RandomizedSearchCV(estimator= RandomForestRegressor(), param_distributions = random_grid_z, scoring= 'neg_mean_squared_error', n_iter = 10, cv = 5, verbose=5, n_jobs=1)
rf_z

rf_z.fit(X1_train, z_train)

print("Best score", rf_z.best_score_)
print("Best parameters", rf_z.best_params_)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf1_z = RandomForestRegressor(
                             n_estimators = 300, 
                             min_samples_split = 6,
                             min_samples_leaf = 1,
                             max_features = 'auto',
                             max_depth = 12,
                             random_state = 20
                            )

rfr_z = rf1_z.fit(X1_train, z_train)

rf_z_pred_train = rfr_z.predict(X1_train)

rf_z_pred_test = rfr_z.predict(X1_test)

rf_mse_train_z = mean_squared_error(z_train, rf_z_pred_train)
print("Mse of RFR on Trained data", rf_mse_train_z.round(3))

rf_mse_test_z = mean_squared_error(z_test, rf_z_pred_test)
print("Mse of RFR on Test data", rf_mse_test_z.round(3))

rf_r2_train_z = r2_score(z_train, rf_z_pred_train)
print("R2_score of RFR on Trained data", rf_r2_train_z.round(3))

rf_r2_test_z = r2_score(z_test, rf_z_pred_test)
print("R2_score of RFR on Test data", rf_r2_test_z.round(3))


# In[ ]:


from sklearn.ensemble import BaggingRegressor
bag_z = BaggingRegressor()

base_estimator = [dt1_z]

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

max_samples = [int(x) for x in np.linspace(start = 1, stop = 10, num = 10)]

max_features = ['auto']

random_state = [1, 5, 10, 20, 50]

from sklearn.model_selection import RandomizedSearchCV

random_grid_z = {
                'base_estimator': base_estimator,
                'n_estimators' : n_estimators,
                'max_samples' : max_samples,
                'random_state' : random_state
               }

print(random_grid_z)

bag_z = RandomizedSearchCV(estimator= BaggingRegressor(), param_distributions = random_grid_z, scoring= 'neg_mean_squared_error', n_iter = 10, cv = 5, verbose=5, n_jobs=1)
bag_z

bag_z.fit(X1_train, z_train)

print("Best score", bag_z.best_score_)
print("Best parameters", bag_z.best_params_)


# In[ ]:


bag1 = BaggingRegressor(
                        base_estimator = dt1, 
                        max_samples=8, 
                        n_estimators=1100, 
                        random_state = 10
                      )

bag_z = bag1.fit(X1_train, z_train)

z_pred_train_bag = bag_z.predict(X1_train)
z_pred_test_bag = bag_z.predict(X1_test)

bag_mse_train_z = mean_squared_error(z_train, z_pred_train_bag)
print("MSE of bagging on Training data", bag_mse_train_z)

bag_mse_test_z = mean_squared_error(z_test, z_pred_test_bag)
print("MSE of bagging on Test data", bag_mse_test_z)

bag_r2_train_z = r2_score(z_train, z_pred_train_bag)
print("R2_score of bagging on Train data", bag_r2_train_z.round(3))

bag_r2_test_z = r2_score(z_test, z_pred_test_bag)
print("R2_score of bagging on Test data", bag_r2_test_z.round(3))


# In[ ]:


alpha = np.arange(0.001, 0.015,0.001)
sgd_mse_train_z = []
sgd_mse_test_z = []
sgd_r2_train_z = []
sgd_r2_test_z = []

for x in alpha:
    sgd_z = SGDRegressor(learning_rate='constant', eta0=x)
    sgd_z.fit(X1_train,z_train)
    sgd_z_pred_train = sgd_z.predict(X1_train)
    sgd_z_pred_test = sgd_z.predict(X1_test)
    sgd_mse_train_z.append(mean_squared_error(z_train, sgd_z_pred_train))
    sgd_mse_test_z.append(mean_squared_error(z_test, sgd_z_pred_test))
    sgd_r2_train_z.append(r2_score(z_train, sgd_z_pred_train))
    sgd_r2_test_z.append(r2_score(z_test, sgd_z_pred_test))

print(sgd_mse_train_z)
print(sgd_mse_test_z)
print(sgd_r2_train_z)
print(sgd_r2_test_z)

err_train_z = pd.DataFrame(sgd_mse_train_z)

err_test_z = pd.DataFrame(sgd_mse_test_z).round(3)

acc_train_z = pd.DataFrame(sgd_r2_train_z).round(3)

acc_test_z = pd.DataFrame(sgd_r2_test_z).round(3)


# In[ ]:


plt.figure(figsize=(15,6))
plt.plot(alpha, err_test_z)


# In[ ]:


plt.plot(alpha, acc_test_z)


# In[ ]:


sgd1_z = SGDRegressor(alpha = 0.01, learning_rate='constant', eta0=0.004)
sgd1_z.fit(X1_train, z_train)

sgd_z_pred_train1 = sgd1_z.predict(X1_train)

sgd_z_pred_test1 = sgd1_z.predict(X1_test)

sgd_mse_train1_z = mean_squared_error(z_train, sgd_z_pred_train1)
print("MSE of XGB on Training data", sgd_mse_train1_z.round(3))

sgd_mse_test1_z = mean_squared_error(z_test, sgd_z_pred_test1)
print("MSE of XGB on Test data", sgd_mse_test1_z.round(3))

sgd_r2_train1_z = r2_score(z_train, sgd_z_pred_train1)
print("R2_score of XGB on Train data", sgd_r2_train1_z.round(3))


sgd_r2_test1_z = r2_score(z_test, sgd_z_pred_test1)
print("R2_score of XGB on Test data", sgd_r2_test1_z.round(3))


# In[ ]:


xgb_z = xg.XGBRegressor()
xgb_z.fit(X1_train, z_train)

xgb_z_pred_train = xgb_z.predict(X1_train)

xgb_z_pred_test = xgb_z.predict(X1_test)

xgb_mse_train_z = mean_squared_error(z_train, xgb_z_pred_train)
print("MSE of XGB on Training data", xgb_mse_train_z.round(3))

xgb_mse_test_z = mean_squared_error(z_test, xgb_z_pred_test)
print("MSE of XGB on Test data", xgb_mse_test_z.round(3))

xgb_r2_train_z = r2_score(z_train, xgb_z_pred_train)
print("R2_score of XGB on Train data", xgb_r2_train_z.round(3))

xgb_r2_test = r2_score(z_test, xgb_z_pred_test)
print("R2_score of XGB on Test data", xgb_r2_test_z.round(3))


# In[ ]:




