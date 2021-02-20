# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist
from sklearn.preprocessing import StandardScaler

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


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=.8, random_state=1)


#  K-means Model
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
    kmeans = KMeans(n_clusters=n_clusters, random_state=1)
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


################################################################################
print("\n")
print("Model k-means 3 Clusters")
################################################################################
model = KMeans(n_clusters=3, random_state=1)
model.fit(X_train)
y_predict = kmeans.predict(X_test)
print("Accuracy: ", metrics.accuracy_score(y_test, y_predict))
print("Classification report:", metrics.classification_report(y_test, y_predict))


################################################################################
print("\n")
print("Model k-means 5 Clusters")
################################################################################
kmeans = KMeans(n_clusters=5, random_state=1)
kmeans.fit(X_train)
y_predict = kmeans.predict(X_test)
print("Accuracy: ", metrics.accuracy_score(y_test, y_predict))
print("Classification report:", metrics.classification_report(y_test, y_predict))


################################################################################
print("\n")
print("Model k-means 7 Clusters")
################################################################################
kmeans = KMeans(n_clusters=7, random_state=1)
kmeans.fit(X_train)
y_predict = kmeans.predict(X_test)
print("Accuracy: ", metrics.accuracy_score(y_test, y_predict))
print("Classification report:", metrics.classification_report(y_test, y_predict))


################################################################################
print("\n")
print("Model k-means 9 Clusters")
################################################################################
kmeans = KMeans(n_clusters=9, random_state=1)
kmeans.fit(X_train)
y_predict = kmeans.predict(X_test)
print("Accuracy: ", metrics.accuracy_score(y_test, y_predict))
print("Classification report:", metrics.classification_report(y_test, y_predict))

# %%
