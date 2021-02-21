# %%
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Read Dataset file
df = pd.read_csv('pre-processed-dataset.csv')


independent_variables = ["mass_margin", "mass_density"]
dependent_variables = ["gravity"]

X = df[independent_variables]
y = df[dependent_variables]


#  Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=.8, random_state=10)


# Normalização
'''
scaler = StandardScaler()
scaler.fit(X_train)
X = scaler.transform(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
'''

#  Logistics Regression Config 1
lr = LogisticRegression()
# l1 - lasso regularization ; l2 - ridge regularization
# C - inverse of regularization strength
lr.fit(X_train, y_train)

# make predictions on the testing set
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)
y_pred = lr.predict(X)

# full data
# print summary
print("\n")
print('Train MAE : ', metrics.mean_absolute_error(y_train, y_train_pred))
print("\n")
print('Train RMSE ', np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))
print("\n")
print('Test MAE: ', metrics.mean_absolute_error(y_test, y_test_pred))
print("\n")
print('Test RMSE : ', np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))
print("\n")
print("Accuracy on full : %s" % lr.score(X, y))
print("\n")
print("Accuracy on test: " + str(lr.score(X_test, y_test)))
print("\n")
print("Accuracy on train: " + str(lr.score(X_train, y_train)))
print("\n")

# determine the false positive and false negative rates
fpr, tpr, _ = metrics.roc_curve(y, lr.predict_proba(X)[:, 1])

# calculate AUC
roc_auc = metrics.auc(fpr, tpr)
print('ROC AUC : %.2f' % roc_auc)


# plot a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve ( area = %.2f ' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate ')
plt.title('ROC curve ')
plt.legend(loc='lower right')
plt.show()

# %%
