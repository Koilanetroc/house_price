import numpy as np
import pandas as pd
import sklearn as sc
import math
from sklearn import preprocessing,linear_model,ensemble
from sklearn import model_selection
from sklearn.preprocessing import Imputer
from sklearn.neural_network import MLPClassifier

imp = Imputer(missing_values="NaN",strategy="mean",axis=0)

# prepare data
full_data = pd.read_csv('encoded_train.csv')
train_data, test_data = model_selection.train_test_split(full_data, test_size=0.3, train_size=0.7, shuffle=False)

train_X = train_data.drop(columns=['SalePrice', 'Unnamed: 0'])
imp.fit(train_X)
train_X = pd.DataFrame(imp.transform(train_X))
train_Y = train_data['SalePrice']

# model = sc.neural_network.MLPRegressor(activation="identity",solver='lbfgs',learning_rate='adaptive')
# model = linear_model.Ridge(alpha=.1)
model = ensemble.RandomForestClassifier()
# model = linear_model.LinearRegression()
model.fit(train_X, train_Y)
train_predictions = model.predict(train_X)
train_errors = sc.metrics.mean_squared_error(train_Y,train_predictions)
print(f'Cost function for train data = {train_errors}')

test_X = test_data.drop(columns=['SalePrice', 'Unnamed: 0'])
imp.fit(test_X)
test_X = pd.DataFrame(imp.transform(test_X))
test_Y = test_data['SalePrice']
model = ensemble.RandomForestClassifier(n_estimators=24)
model.fit(test_X, test_Y)
test_predictions = model.predict(test_X)
test_errors = sc.metrics.mean_squared_error(test_Y, test_predictions)
print(f'Cost function for test data = {test_errors}')

# print('Start writing...')
# f = open('random_forest_submission.csv', 'w+')
# f.write('Id,SalePrice\n')
# counter = 1461
# for y in predictions:
#     f.write(f'{counter},{int(y)}\n')
#     counter += 1
# print('Finished writing')
