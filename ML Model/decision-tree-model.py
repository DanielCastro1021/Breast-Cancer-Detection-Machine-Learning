# %%
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree

# Read Dataset file
df = pd.read_csv('pre-processed-dataset.csv')


independent_variables = ["mass_margin", "mass_density"]
dependent_variables = ["gravity"]

X = df[independent_variables]
y = df[dependent_variables]


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=.8, random_state=10)


# Decision Tree Model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
print("Score on test: " + str(clf.score(X_test, y_test)))
print("Score on train: " + str(clf.score(X_train, y_train)))


tree.plot_tree(clf)

# %%
