# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import layers
from keras import models
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

model = models.Sequential()
model.add(layers.Dense(2, activation='relu',
                       input_shape=(2,)))
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
