import numpy as np
import pandas as pd
import sklearn as sc
import math
from scipy.stats import uniform as sp_rand
from sklearn import preprocessing,linear_model,ensemble
from sklearn.preprocessing import Imputer
from sklearn.neural_network import MLPClassifier
from sklearn.grid_search import RandomizedSearchCV


train_data = pd.read_csv('encoded_train.csv')
Y = train_data['SalePrice']
X = train_data.drop(columns=['SalePrice', 'Unnamed: 0'])
imp = Imputer(missing_values="NaN",strategy="mean",axis=0)
imp.fit(X)
X = pd.DataFrame(imp.transform(X))

param_grid = {'alpha': sp_rand()}
model = linear_model.LinearRegression()
rsearch = RandomizedSearchCV(estimator=model,param_distributions=param_grid, n_iter=100)
rsearch.fit(X,Y)
alpha = rsearch.best_estimator_.alpha
model = linear_model.Ridge(alpha=alpha,solver="auto")
model.fit(X,Y)
train_predictions = model.predict(X)
errors = sc.metrics.mean_squared_error(Y,train_predictions)
print(f'Cost function = {errors}')
