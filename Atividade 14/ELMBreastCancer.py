from ELM import ELM
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class ELMBreastCancer:

    def __init__(self):
        self.data = load_breast_cancer()
        self.X = pd.DataFrame(self.data.data, columns=self.data.feature_names)
        self.y = pd.Series(self.data.target)

    def fit_predict(self):
        scaler = StandardScaler()
        x_scaler = scaler.fit_transform(self.X)

        X_train, X_test, y_train, y_test = train_test_split(x_scaler, self.y, test_size=0.30, random_state=42,
                                                            stratify=self.y)

        elm = ELM()
        elm.fit(X_train, y_train)

        predict = elm.predict(X_test)
        accuracy = accuracy_score(y_test, predict)

        print(accuracy)
