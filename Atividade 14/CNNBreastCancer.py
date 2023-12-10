import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import keras
from keras.layers import Dense, Activation, Flatten, Conv2D
from keras.models import Sequential

class CNNBreastCancer:

    def __init__(self):
        self.data = load_breast_cancer()
        self.X = pd.DataFrame(self.data.data, columns=self.data.feature_names)
        self.y = pd.Series(self.data.target)

    def fit_predict(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.35, random_state=42,  stratify=self.y)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1, 1)
        X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1, 1)

        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(X_train_reshaped.shape[1], 1, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Activation('relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        model.add(Activation('softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        model.fit(X_train_reshaped, y_train, batch_size=32, epochs=2, validation_split=0.2, verbose=2)

        test_loss, test_acc = model.evaluate(X_test_reshaped, y_test)
        print('Test loss: ', test_loss)
        print('Test Accuracy: ', test_acc)

