# coding=utf-8


'''
Author:zhouhuan
Email:18832832911@139.com

data:2020/12/12/012 16:29
desc:
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
print()

tf.logging.set_verbosity(tf.logging.ERROR)
mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784], name="x_placeholder")
y_ = tf.placeholder(tf.float32, [None, 10], name="y_placeholder")

W = tf.Variable(tf.zeros([784, 10], dtype=tf.float32, name="weight_variable"))
b = tf.Variable(tf.zeros([10]), name="bias_variable")

y = tf.matmul(x, W) + b

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y, name="lossFunction"), name="loss")

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss, name="gradDescent")

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

test_accuracy = sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels})

print("Test Accuracy: {}%".format(test_accuracy * 100.0))
sess.close()























































































