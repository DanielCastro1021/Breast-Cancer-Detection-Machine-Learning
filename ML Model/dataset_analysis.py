# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist
from sklearn.preprocessing import StandardScaler

# Read Dataset file
df = pd.read_csv('dataset.csv')

# Config Seaborn Graphs
sns.set_theme(style="ticks", color_codes=True)

# Check For Null Values and which atribuite has Null values.
print("\n")
print(df.isnull().any())
print("\n")

# Convert Missing values To -1
df['evaluation_bi_rads'] = np.where(
    df.evaluation_bi_rads == '?', -1, df.evaluation_bi_rads)
df['mass_shape'] = np.where(df.mass_shape == '?', -1, df.mass_shape)
df['mass_margin'] = np.where(df.mass_margin == '?', -1, df.mass_margin)
df['mass_density'] = np.where(df.mass_density == '?', -1, df.mass_density)
df['age'] = np.where(df.age == '?', -1, df.age)


# Convert Columns from object type to int type
print(df.dtypes)
print("\n")

for col_name in df.columns:
    if(df[col_name].dtype == 'object'):
        df[col_name] = df[col_name].astype('int')

print(df.dtypes)
print("\n")


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


# See Count of Values Submited per Atribuite
print("\n\n\n\nGravity Count Per Value Submited")
print(df['gravity'].value_counts())


print("\n\n\n\nAge Count Per Value Submited")
print(df['age'].value_counts())


print("\n\n\n\nEvaluation Bi Rads Count Per Value Submited")
print(df['evaluation_bi_rads'].value_counts())


print("\n\n\n\nMass Shape Count Per Value Submited")
print(df['mass_shape'] .value_counts())


print("\n\n\n\nMasse Margin Count Per Value Submited")
print(df['mass_margin'].value_counts())

print("\n\n\n\nMasse Desnsity Count Per Value Submited")
print(df['mass_density'].value_counts())


# See Data in Plots
for col_name in df.columns:
    if col_name != "age":
        sns.catplot(x=col_name, data=df, kind='count')
    if col_name != "gravity":
        sns.catplot(x=col_name, data=df, kind='box')


# See correlation matrix
corr = df.corr()
print(corr)
sm.graphics.plot_corr(corr, xnames=list(corr.columns))
plt.show()


independent_variables = ["evaluation_bi_rads", "age",
                         "mass_shape", "mass_margin", "mass_density"]
dependent_variables = ["gravity"]

X = df[independent_variables]
y = df[dependent_variables]

print("Done defining X and y")


thresh = 10  # threshold for VIF

for i in np.arange(0, len(independent_variables)):
    vif = [variance_inflation_factor(X[independent_variables].values, ix)
           for ix in range(X[independent_variables].shape[1])]
    maxloc = vif.index(max(vif))
    if max(vif) > thresh:
        print('vif : ', vif)
        print('dropping ',
              X[independent_variables]. columns[maxloc], ' at index ', maxloc)

        del independent_variables[maxloc]
    else:
        break

X = df[independent_variables]
print('Final variables : ', independent_variables)


################################################################################################################################################
################################################################################################################################################
################################################################################################################################################


# %% Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=.8, random_state=1)


################################################################################################################################################
################################################################################################################################################
################################################################################################################################################

# %% Logistics Regression Config 1
lr1 = LogisticRegression()
# l1 - lasso regularization ; l2 - ridge regularization
# C - inverse of regularization strength
lr1.fit(X_train, y_train)

# make predictions on the testing set
y_train_pred = lr1.predict(X_train)
y_test_pred = lr1.predict(X_test)
y_pred = lr1.predict(X)

# full data
# print summary
print(lr1.score(X, y_pred))
print('Train MAE : ', metrics.mean_absolute_error(y_train, y_train_pred))
print('Train RMSE ', np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))
print('Test MAE: ', metrics.mean_absolute_error(y_test, y_test_pred))
print('Test RMSE : ', np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))

print("Accuracy: %s" % lr1.score(X, y))


# determine the false positive and false negative rates
fpr, tpr, _ = metrics.roc_curve(y, lr1.predict_proba(X)[:, 1])
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
plt.savefig('roc.pdf')
plt.show()


################################################################################################################################################
################################################################################################################################################
################################################################################################################################################


# %% K-means Model
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)
################################################################################
# Find number of clusters - Elbow Method
################################################################################
K = range(1, 10)
KM = [KMeans(n_clusters=k). fit(X) for k in K]
centroids = [k. cluster_centers_ for k in KM]
D_k = [cdist(X, cent, 'euclidean') for cent in centroids]
cIdx = [np . argmin(D, axis=1) for D in D_k]
dist = [np . min(D, axis=1) for D in D_k]
avgWithinSS = [sum(d)/X . shape[0] for d in dist]

wcss = [sum(d ** 2) for d in dist]
tss = sum(pdist(X) ** 2)/X.shape[0]
bss = tss - wcss
varExplained = bss / tss * 100
kIdx = 2
# plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(K, avgWithinSS, 'b*-')
plt.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12,
         markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters ')
plt.ylabel('Average within - cluster sum of squares')
plt.title('Elbow for k- means clustering ')
################################################################################
# Find number of clusters - Average Silhouette Method
################################################################################
score = []
for n_clusters in range(2, 10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=11)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    score.append(silhouette_score(X, labels, metric='euclidean'))
# plot
plt.figure(figsize=(10, 10))
plt.plot(score)
plt.grid(True)
plt.ylabel('Silhouette Score ')
plt.xlabel('k')
plt.title('Silhouette for k- means')
plt.show()

# %%
################################################################################
# Model k-means 3 Clusters
################################################################################
model = KMeans(n_clusters=2, random_state=12)
model.fit(X)
y_predict = np.choose(model.labels_, [0, 1]).astype(np.int64)
print("Accuracy: ", metrics.accuracy_score(y, y_predict))
print("Classification report:", metrics.classification_report(y, y_predict))


# %%
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_predict = np.choose(kmeans.labels_, [0, 1]).astype(np.int64)
print("Accuracy: ", metrics.accuracy_score(y, y_predict))
print("Classification report:", metrics.classification_report(y, y_predict))

# %%
