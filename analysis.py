import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

features, target = tf.keras.datasets.mnist.load_data()[0]

features = np.reshape(features, (features.shape[0], features.shape[1] * features.shape[2]))

features = pd.DataFrame(features)
target = pd.Series(target)

x_train, x_test, y_train, y_test = train_test_split(features[:6000], target[:6000],
                                                    test_size=0.3, random_state=40)

print(f"x_train shape: {x_train.shape}",
      f"x_test shape: {x_test.shape}",
      f"y_train shape: {y_train.shape}",
      f"y_test shape: {y_test.shape}",
      f"Proportion of samples per class in train set:",
      y_train.value_counts(normalize=True),
      sep='\n')
