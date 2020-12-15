# coding=utf-8


'''
Author:zhouhuan
Email:18832832911@139.com

data:2020/12/12/012 20:09
desc:
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime

subFolder = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = f"./tfb_logs/{subFolder}"

tf.logging.set_verbosity(tf.logging.ERROR)

mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)

with tf.name_scope("variable_scope"):
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x_placeholder")
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_placeholder")

    tf.summary.image("image_input", tf.reshape(x, [-1, 28, 28, 1]), 3)

    with tf.name_scope("bias_scope"):
        b = tf.Variable(tf.zeros([10]), name="bias_variable")
        tf.summary.histogram("bias_histogram", b)

    with tf.name_scope("weight_scope"):
        w = tf.Variable(tf.zeros([784,10]), name="weight_variable")
        tf.summary.histogram("weight_histogram", w)

    assert x.get_shape().as_list() == [None, 784]
    assert y_.get_shape().as_list() == [None, 10]
    assert w.get_shape().as_list() == [784, 10]
    assert b.get_shape().as_list() == [10]

    with tf.name_scope("yReal_scope"):
        y = tf.add(tf.matmul(x, w), b, name="y_calculated")
        tf.summary.histogram("yReal_histogram", y)
    assert y.get_shape().as_list() == [None, 10]


with tf.name_scope("loss_scope"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=y_, logits=y, name="lossFunction"
    ), name="loss")

with tf.name_scope("training_scope"):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss, name="gradDescent")
    tf.summary.histogram("loss_histogram", loss)
    tf.summary.scalar("loss_scalar", loss)

with tf.name_scope("accuracy_scope"):
    correct_prediction = tf.equal(tf.argmax(y, 1),tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
    tf.summary.histogram("accuracy_scala", accuracy)
    tf.summary.scalar("accuracy_scala", accuracy)

init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
sess.run(init)

merged_summary_op = tf.summary.merge_all()
tbWriter = tf.summary.FileWriter(logdir)

tbWriter.add_graph(sess.graph)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    if i % 5 == 0:
        summary = sess.run(merged_summary_op, feed_dict={x: batch_xs, y_: batch_ys})
        tbWriter.add_summary(summary, i)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

print("============================================")
print(
    f"Accuracy of the model is: {sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})*100}%"
)

sess.close()
