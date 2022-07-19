import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer


features, target = tf.keras.datasets.mnist.load_data()[0]

features = np.reshape(features, (features.shape[0], features.shape[1] * features.shape[2]))

features, target = pd.DataFrame(features), pd.Series(target)

x_train, x_test, y_train, y_test = train_test_split(features[:6000], target[:6000],
                                                    test_size=0.3, random_state=40)

x_train_norm, x_test_norm = Normalizer().transform(x_train), Normalizer().transform(x_test)

models = [('K-nearest neighbors', KNeighborsClassifier(), {'n_neighbors': (3, 4),
                                                           'weights': ('uniform', 'distance'),
                                                           'algorithm': ('auto', 'brute')
                                                           }
           ),
          ('Random forest', RandomForestClassifier(random_state=40), {'n_estimators': (300, 500),
                                                                      'max_features': ('auto', 'log2'),
                                                                      'class_weight': ('balanced', 'balanced_subsample')
                                                                      }
           )
          ]

for item in models:
    gs = GridSearchCV(item[1], item[2], scoring='accuracy', n_jobs=-1)
    gs.fit(x_train_norm, y_train)
    print(f"{item[0]} algorithm"
          f"\nbest estimator: {gs.best_estimator_}"
          f"\naccuracy: {gs.score(x_test_norm, y_test)}\n")
