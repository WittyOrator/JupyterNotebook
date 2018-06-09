import os
import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt

def load_cifar10(num_training=49000, num_validation=1000, num_test=10000):
    """
    Fetch the CIFAR-10 dataset from the web and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 dataset and use appropriate data types and shapes
    cifar10 = tf.keras.datasets.cifar10.load_data()
    (X_train, y_train), (X_test, y_test) = cifar10
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32).flatten()
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32).flatten()

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean pixel and divide by std
    mean_pixel = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std_pixel = X_train.std(axis=(0, 1, 2), keepdims=True)
    X_train = (X_train - mean_pixel) / std_pixel
    X_val = (X_val - mean_pixel) / std_pixel
    X_test = (X_test - mean_pixel) / std_pixel

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
NHW = (0, 1, 2)
X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape, y_train.dtype)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


class Dataset(object):
    def __init__(self, X, y, batch_size, shuffle=False):
        """
        Construct a Dataset object to iterate over data X and labels y

        Inputs:
        - X: Numpy array of data, of any shape
        - y: Numpy array of labels, of any shape but with y.shape[0] == X.shape[0]
        - batch_size: Integer giving number of elements per minibatch
        - shuffle: (optional) Boolean, whether to shuffle the data on each epoch
        """
        assert X.shape[0] == y.shape[0], 'Got different numbers of data and labels'
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i + B], self.y[i:i + B]) for i in range(0, N, B))


train_dset = Dataset(X_train, y_train, batch_size=64, shuffle=True)
val_dset = Dataset(X_val, y_val, batch_size=64, shuffle=False)
test_dset = Dataset(X_test, y_test, batch_size=64)

# We can iterate through a dataset like this:
for t, (x, y) in enumerate(train_dset):
    print(t, x.shape, y.shape)
    if t > 5: break

# Set up some global variables
USE_GPU = True

if USE_GPU:
    device = '/gpu:0'
else:
    device = '/cpu:0'

print('Using device: ', device)


with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
    b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
with tf.device('/gpu:0'):
    c = a + b

# 注意：allow_soft_placement=True表明：计算设备可自行选择，如果没有这个参数，会报错。
# 因为不是所有的操作都可以被放在GPU上，如果强行将无法放在GPU上的操作指定到GPU上，将会报错。
# sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.global_variables_initializer())
print(sess.run(c))


learning_rate = 1e-2
input_dim = (32, 32, 3)
num_classes = 10
channel_1, channel_2, num_classes = 16, 32, 10
filter_1, filter_2 = (5, 5), (3, 3)

tf.reset_default_graph()

with tf.device(device):
    sc_init = tf.variance_scaling_initializer(scale=2.0)

    layers = [
        tf.keras.layers.Conv2D(channel_1, filter_1, (1, 1), "same", \
                               use_bias=True, bias_initializer=tf.zeros_initializer(), \
                               kernel_regularizer=tf.keras.regularizers.l2(0.01), \
                               kernel_initializer=sc_init, input_shape=input_dim),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0),
        tf.keras.layers.Conv2D(channel_2, filter_2, (1, 1), "same", \
                               use_bias=True, bias_initializer=tf.zeros_initializer(), \
                               kernel_regularizer=tf.keras.regularizers.l2(0.01), \
                               kernel_initializer=sc_init),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, kernel_initializer=sc_init,kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Softmax(),
    ]

    model = tf.keras.Sequential(layers)

    optimizer = tf.keras.optimizers.SGD(learning_rate,0.9,1e-6,True)

    model.compile(optimizer,"sparse_categorical_crossentropy", metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val,y_val))

    score = model.evaluate(X_test,y_test,batch_size=32)

    print(score)