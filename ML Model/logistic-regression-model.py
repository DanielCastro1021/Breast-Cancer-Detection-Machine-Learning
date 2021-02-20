# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Read Dataset file
df = pd.read_csv('dataset.csv')

# Convert Missing values To -1
df['evaluation_bi_rads'] = np.where(
    df.evaluation_bi_rads == '?', -1, df.evaluation_bi_rads)
df['mass_shape'] = np.where(df.mass_shape == '?', -1, df.mass_shape)
df['mass_margin'] = np.where(df.mass_margin == '?', -1, df.mass_margin)
df['mass_density'] = np.where(df.mass_density == '?', -1, df.mass_density)
df['age'] = np.where(df.age == '?', -1, df.age)


# Convert Columns from object type to int type
for col_name in df.columns:
    if(df[col_name].dtype == 'object'):
        df[col_name] = df[col_name].astype('int')


# Replace Nan With Mode for Categorical Variables and Mean for Age

# evaluation_bi_rads Range is 1 to 5
df.loc[df['evaluation_bi_rads'] < 1,
       'evaluation_bi_rads'] = df['evaluation_bi_rads'].mode()

df.loc[df['evaluation_bi_rads'] > 5,
       'evaluation_bi_rads'] = df['evaluation_bi_rads'].mode()

df['evaluation_bi_rads'].fillna(
    df['evaluation_bi_rads'].mode()[0], inplace=True)

# mass_shape Range is 1 to 4
df.loc[df['mass_shape'] < 1,
       'mass_shape'] = df['mass_shape'].mode()

df.loc[df['mass_shape'] > 4,
       'mass_shape'] = df['mass_shape'].mode()

df['mass_shape'].fillna(df['mass_shape'].mode()[0], inplace=True)

# mass_margin Range is 1 to 5
df.loc[df['mass_margin'] < 1,
       'mass_margin'] = df['mass_margin'].mode()

df.loc[df['mass_margin'] > 5,
       'mass_margin'] = df['mass_margin'].mode()

df['mass_margin'].fillna(df['mass_margin'].mode()[0], inplace=True)


# mass_density Range is 1 to 4
df.loc[df['mass_density'] < 1,
       'mass_density'] = df['mass_density'].mode()

df.loc[df['mass_density'] > 4,
       'mass_density'] = df['mass_density'].mode()

df['mass_density'].fillna(df['mass_density'].mode()[0], inplace=True)

# age
df['age'] = np.where(df.age == -1, df['age'].mean(), df.age)


independent_variables = ["evaluation_bi_rads", "age",
                         "mass_shape", "mass_margin", "mass_density"]
dependent_variables = ["gravity"]

X = df[independent_variables]
y = df[dependent_variables]


thresh = 10  # threshold for VIF

for i in np.arange(0, len(independent_variables)):
    vif = [variance_inflation_factor(X[independent_variables].values, ix)
           for ix in range(X[independent_variables].shape[1])]
    maxloc = vif.index(max(vif))
    if max(vif) > thresh:
        del independent_variables[maxloc]
    else:
        break

X = df[independent_variables]


#  Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=.8, random_state=1)

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
