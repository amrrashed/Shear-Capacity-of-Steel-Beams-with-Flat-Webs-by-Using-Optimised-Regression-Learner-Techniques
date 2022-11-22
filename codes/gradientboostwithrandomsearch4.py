
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 23:04:32 2022

@author: amr_r
"""
import numpy as np
import pandas as pd  # To read data
#import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
# Hide warnings
import warnings
warnings.filterwarnings("ignore")
# Setting up max columns displayed to 100
pd.options.display.max_columns = 100
data = pd.read_csv('G:/new researches/civil/DATASET/DB2.csv')  # load data set
data.dropna(inplace=True)
a=data.describe()
data.info
X = data.iloc[:,0:7]# 0:4 for DB1  0:7 for DB2
#X = preprocessing.normalize(X1)
#X=preprocessing.normalize(X) 
y = data.iloc[:,7]# 4:8 FOR DB1 7 for DB2
# Call train_test_split on the data and capture the results

# A parameter grid for XGBoost
params = {
 'learning_rate' : [0.05,0.10,0.15,0.20,0.25,0.30],
 'max_depth' : [ 3, 4, 5, 6, 8, 10, 12, 15],
 'min_samples_split' : [ 1, 3, 5, 7 ],
}

#reg=GradientBoostingRegressor(random_state=0)
reg = GradientBoostingRegressor()

# run randomized search
n_iter_search = 500
random_search = RandomizedSearchCV(reg, param_distributions=params,
                                   n_iter=n_iter_search, cv=5,n_jobs=-1,verbose=3, scoring='neg_mean_squared_error')

start = time.time()
result= random_search.fit(X, y)
print("RandomizedSearchCV took %.2f seconds for %d candidates" " parameter settings." % ((time.time() - start), n_iter_search))

best_regressor = random_search.best_estimator_

from sklearn.metrics import mean_absolute_error,r2_score ,mean_squared_error
# Get predictions
y_pred = best_regressor.predict(X)
# Calculate MAE
rmse_pred = mean_absolute_error(y, y_pred) 
r2score_pred = r2_score(y, y_pred) 
mse = mean_squared_error(y,y_pred) 

print("Root Mean Absolute Error:" , np.sqrt(rmse_pred))
print("R2 score:" , r2score_pred)
print("The mean squared error (MSE) : {:.4f}".format(mse))
print('Best Hyperparameters: %s' % result.best_params_)

# Plot outputs
# plt.scatter(X_test, y_test, color="black")
# plt.plot(X_test, y_pred, color="blue", linewidth=3)
# plt.xticks(())
# plt.yticks(())

#plt.show()
###########
 # 'n_estimators':[500],
 #    'min_child_weight':[4,5], 
 #    'gamma':[i/10.0 for i in range(3,6)],  
 #    'subsample':[i/10.0 for i in range(6,11)],
 #    'colsample_bytree':[i/10.0 for i in range(6,11)], 
 #    'max_depth': [2,3,4,6,7],
 #    'objective': ['reg:squarederror', 'reg:tweedie'],
 #    'booster': ['gbtree', 'gblinear'],
 #    'eval_metric': ['rmse'],
 #    'eta': [i/10.0 for i in range(3,6)],




