import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer


def fit_predict_eval(model, features_train, features_test, target_train, target_test):

    model.fit(features_train, target_train)

    # model.predict(features_test)

    score = model.score(features_test, target_test)

    return score


features, target = tf.keras.datasets.mnist.load_data()[0]

features = np.reshape(features, (features.shape[0], features.shape[1] * features.shape[2]))

features = pd.DataFrame(features)
target = pd.Series(target)

x_train, x_test, y_train, y_test = train_test_split(features[:6000], target[:6000],
                                                    test_size=0.3, random_state=40)

x_train_norm, x_test_norm = Normalizer().transform(x_train), Normalizer().transform(x_test)

models = [['KNeighborsClassifier', KNeighborsClassifier(), 0],
          ['DecisionTreeClassifier', DecisionTreeClassifier(random_state=40), 0],
          ['LogisticRegression', LogisticRegression(solver='liblinear'), 0],
          ['RandomForestClassifier', RandomForestClassifier(random_state=40), 0]]

for item in models:
    item[2] = fit_predict_eval(item[1], x_train_norm, x_test_norm, y_train, y_test)
    print(f'Model: {item[0]}\nAccuracy: {item[2]}\n')

models = sorted(models, key=lambda i: i[2], reverse=True)

print('The answer to the 1st question: yes')
print(f'The answer to the 2nd question: '
      f'{models[0][0]} - {round(models[0][2], 3)},'
      f'{models[1][0]} - {round(models[1][2], 3)}')
