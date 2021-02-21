# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Read Dataset file
df = pd.read_csv('pre-processed-dataset.csv')


independent_variables = ["mass_margin", "mass_density"]
dependent_variables = ["gravity"]

X = df[independent_variables]
y = df[dependent_variables]


#  Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=.8, random_state=10)


# Rede Neuronal

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(
    100), activation='logistic', max_iter=100, early_stopping=True)

mlp.fit(X_train_scaled, y_train)


print("Training set score : %f" % mlp.score(X_train_scaled, y_train))
print("Test set score : %f" % mlp.score(X_test_scaled, y_test))

plt.plot(mlp.loss_curve_)
plt.plot(mlp.validation_scores_)
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# %%
