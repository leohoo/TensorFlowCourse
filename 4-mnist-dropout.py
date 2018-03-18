import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

batch_size = 30

n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None,10])
keep_prob = tf.placeholder(tf.float32)

M = 500
W = tf.Variable(tf.truncated_normal([784, M], stddev=0.1))
b = tf.Variable(tf.truncated_normal([M], stddev=0.1))
L1 = tf.nn.tanh(tf.matmul(x, W)+b)
L1_drop = tf.nn.dropout(L1, keep_prob)

N = 500
W2 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
b2 = tf.Variable(tf.zeros([N]))
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2)+b2)
L2_drop = tf.nn.dropout(L2, keep_prob)

P = 200
W3 = tf.Variable(tf.truncated_normal([N, P], stddev=0.1))
b3 = tf.Variable(tf.zeros([P]))
L3 = tf.nn.tanh(tf.matmul(L2_drop, W3)+b3)
L3_drop = tf.nn.dropout(L3, keep_prob)

W4 = tf.Variable(tf.truncated_normal([P, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10]))

prediction = tf.nn.softmax(tf.matmul(L3_drop, W4)+b4)

# loss = tf.reduce_mean(tf.square(y-prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
train_step = tf.train.GradientDescentOptimizer(0.3).minimize(loss)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(prediction, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(28):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys, keep_prob:0.6})

        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
        train_acc = sess.run(accuracy, feed_dict={x:mnist.train.images, y:mnist.train.labels, keep_prob:1.0})
        print("Iter " + str(epoch) + ", Testing accuracy " + str(acc) + ", Train acc: " + str(train_acc))
