#the following lines will grab the mnist dataset from the online server
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#import tensorflow
import tensorflow as tf
#create a placeholder x, which we will give a value when tensorflow,
#the image is converted into a 784-dimensional vector
#we can insert a image using a 2D tensor with the shape [None,784]
#none means that the first dimension can be of any length
x = tf.placeholder(tf.float32, [None, 784])
#declare the weight and the bias as variables which can be modified by computation
W = tf.Variable(tf.zeros([784, 10]))
#w has the shape [784,10] this is because we want to multiply it with the image vector
# to produce a vector of size 10
#this will be represent the class of 10 different images
b = tf.Variable(tf.zeros([10]))
#the model is defined below
y = tf.nn.softmax(tf.matmul(x, W) + b)
#First, we multiply x by W with the expression tf.matmul(x, W)
#then add the bias
#Training of the model begins here, we need to input our training data into the model
y_ = tf.placeholder(tf.float32, [None, 10])
#this example will use cross entropy as the error measure of the model
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#launching the session will allow us to input data into the placeholder
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
