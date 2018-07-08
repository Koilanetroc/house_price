import numpy as np
import pandas as pd
import sklearn as sc
import math
from sklearn import preprocessing,linear_model
from sklearn.neural_network import MLPClassifier

data = pd.read_csv('test.csv').drop(columns=['Id'])
encoded_columns = ['MSSubClass','MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition']
for column in data.columns:
    print(f'Current column is {column}')
    if column in encoded_columns:
        for i in range(len(data[column])):
            if data[column][i] != data[column][i]:
                data[column][i] = 'NA'
        le = preprocessing.LabelEncoder()
        le.fit(data[column])
        data[column] = le.transform(data[column])

print('Start writing...')
data.to_csv('encoded_test.csv', sep=',')
print('Finished writing')
