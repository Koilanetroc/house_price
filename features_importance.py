print(__doc__)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.ensemble import ExtraTreesClassifier

# Build a classification task using 3 informative features
train_data = pd.read_csv('encoded_train.csv')
Y = train_data['SalePrice']
X = train_data.drop(columns=['SalePrice', 'Unnamed: 0'])
imp = Imputer(missing_values="NaN",strategy="mean",axis=0)
imp.fit(X)
X = pd.DataFrame(imp.transform(X))
columns = train_data.drop(columns=['SalePrice', 'Unnamed: 0']).columns

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, Y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, columns[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(X.shape[1]), importances[indices],
#        color="r", yerr=std[indices], align="center")
# plt.xticks(range(X.shape[1]), indices)
# plt.xlim([-1, X.shape[1]])
# plt.show()
