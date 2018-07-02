import tensorflow as tf
import keras

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
    b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
with tf.device('/gpu:0'):
    c = a + b