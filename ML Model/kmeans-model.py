# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist
from sklearn.preprocessing import StandardScaler

# Read Dataset file
df = pd.read_csv('pre-processed-dataset.csv')


independent_variables = ["mass_margin", "mass_density"]
dependent_variables = ["gravity"]

X = df[independent_variables]
y = df[dependent_variables]


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
# Find number of clusters -Average Silhouette Method
################################################################################
score = []
for n_clusters in range(2, 10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=10)
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
# Model k-means
################################################################################
model = KMeans(n_clusters=2, random_state=10)
model.fit(X)
y_predict = np.choose(model.labels_, [0, 1]).astype(np.int64)
print("Accuracy: ", metrics.accuracy_score(y, y_predict))
print("Classification report:", metrics.classification_report(y, y_predict))
