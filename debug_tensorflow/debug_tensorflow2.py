# coding=utf-8


'''
Author:zhouhuan
Email:18832832911@139.com

data:2020/12/12/012 19:44
desc:
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.ERROR)

mnist = input_data.read_data_sets("MNIST_DATA", one_hot=True)

x = tf.placeholder(tf.float32, shape=[None,784], name='x_placeholder')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_placeholder')

W = tf.Variable(tf.zeros([784, 10]), name='weight_variable')
b = tf.Variable(tf.zeros([10]), name="bias_variable")

assert x.get_shape().as_list() == [None, 784]
assert y_.get_shape().as_list() == [None, 10]
assert W.get_shape().as_list() == [784, 10]
assert b.get_shape().as_list() == [10]

y = tf.add(tf.matmul(x,W), b)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y, name=
                                                              "lossFunction"),
                      name="loss")
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss, name='gradDeacent')
init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs , y_:batch_ys})

    if i % 20 == 0:
        loss = tf.Print(loss, [loss], message="loss")
        loss.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        print(f"Loss of the model is : {sess.run(loss, feed_dict={x: mnist.test.images, y_: mnist.test.labels})}")


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32), name="accuracy")

print("===========================")

print(f"The bias parameter is : {sess.run(b, feed_dict={x: mnist.test.images, y_:mnist.test.labels})}")

print(f"Accuracy of the model is : {sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels})}")

print(f"Loss of the model is : {sess.run(loss, feed_dict={x: mnist.test.images, y_:mnist.test.labels})}")

sess.close()