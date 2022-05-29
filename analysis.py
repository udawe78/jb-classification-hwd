import tensorflow as tf
import numpy as np
x_train, y_train = tf.keras.datasets.mnist.load_data()[0]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
print(f"Classes: {np.unique(y_train)}")
print(f"Features' shape: {x_train.shape}")
print(f"Target's shape: {y_train.shape}")
print(f"min: {x_train.min()}, max: {x_train.max()}")
