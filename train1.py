# coding=utf-8


'''
Author:zhouhuan
Email:18832832911@139.com

data:2020/12/11/011 16:14
desc:
'''

import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)

with tf.Session() as sess:
    print("a:",sess.run(a)," b:",sess.run(b))

c = tf.placeholder(tf.float32)
d = tf.placeholder(tf.float32)

add = tf.add(c,d)

mul = tf.multiply(c,d)

with tf.Session() as sess:
    print(sess.run(add,feed_dict={c:2,d:3}))

    print(sess.run(mul,feed_dict={c:2,d:6}))

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])

product = tf.matmul(matrix1,matrix2)

with tf.Session() as sess:
    result = sess.run(product)
    print(result)





















































































