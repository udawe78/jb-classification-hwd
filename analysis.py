import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

x_train, y_train = tf.keras.datasets.mnist.load_data()[0]

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))

print(f"Classes: {np.unique(y_train)}",
      f"Features' shape: {x_train.shape}",
      f"Target's shape: {y_train.shape}",
      f"min: {x_train.min()}, max: {x_train.max()}", sep='\n')

x_train, x_test, y_train, y_test = train_test_split(x_train[:6000], y_train[:6000],
                                                    test_size=0.3, random_state=40)
