# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras import layers
from keras import models
from keras import layers
from keras import models
from keras import regularizers
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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


X_std = StandardScaler().fit_transform(X)


# create covariance matrix
cov_mat = np.cov(X_std.T)
print(' Covariance matrix \n' % cov_mat)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print(' Eigenvectors \n\n % s' % eig_vecs)
print('\n Eigenvalues \n\n % s' % eig_vals)

tot = sum(eig_vals)
var_exp = [(i/tot) * 100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print('\n\n Cummulative Variance Explained \n\n', cum_var_exp)


plt.figure(figsize=(8, 6))
plt.bar(range(len(independent_variables)), var_exp, alpha=0.5, align='center',
        label='Individual Explained Variance')
plt.step(range(len(independent_variables)), cum_var_exp, where='mid',
         label='Cumulative explained variance ')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# to better understanding of interaction of the dimensions
# plot the first 3 PCA dimensions

X_reduced = PCA(n_components=len(independent_variables)).fit_transform(X)

#  Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, train_size=.8, random_state=10)

# Rede Neuronal

model = models.Sequential()
model.add(layers.Dense(2, kernel_regularizer=regularizers.l2(0.003),
                       activation='relu', input_shape=(len(independent_variables),)))
model.add(layers.Dense(
    2, kernel_regularizer=regularizers.l2(0.003), activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=100, batch_size=25,
                    validation_data=(X_test, y_test), verbose=0)


print("score on train: " + str(model.evaluate(X_train, y_train)[1]))
print("score on test: " + str(model.evaluate(X_test, y_test)[1]))


# summarize history for accuracy
plt.figure(1, figsize=(8, 6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.figure(1, figsize=(8, 6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %%
