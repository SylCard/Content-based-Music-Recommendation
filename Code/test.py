import tensorflow as tf
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

# Parameters
learning_rate = 0.001
training_iters = 3210
batch_size = 107
display_step = 10

# Network Parameters
n_input =  82432 # 644x128
n_classes = 4
dropout = 0.75 # Dropout, probability to keep units

# load training data
trainData = []
batch1 = []
batch2 = []
batch1labels = []
batch2labels = []
trainLabels = []
with open('../../4genreBatch1.data', 'r') as f:
  batch1 = pickle.load(f)
with open('../../4genreBatch2.data', 'r') as f:
  batch2 = pickle.load(f)
with open('../../4genreBatch1.labels', 'r') as f:
  batch1labels = pickle.load(f)
with open('../../4genreBatch2.labels', 'r') as f:
  batch2labels = pickle.load(f)

trainData = np.concatenate((batch1, batch2), axis=0)
trainLabels = np.concatenate((batch1labels, batch2labels), axis=0)

# Shuffle training data
permutation = np.random.permutation(trainData.shape[0])
trainData = trainData[permutation]
trainLabels = trainLabels[permutation]

#load evaluation data
testData = []
testLabels = []
with open('../../4GenreTest.data', 'r') as f:
  testData = pickle.load(f)
with open('../../4GenreTest.labels', 'r') as f:
  testLabels = pickle.load(f)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input]) # input, i.e. pixels that constitute the image
y = tf.placeholder(tf.float32, [None, n_classes]) # labels, i.e which genre the song is
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Store layers weight & bias
weights = {
        # 4x4 conv, 1 input, 256 filters
        'wc1': tf.Variable(tf.random_normal([4, 4, 1, 128])),
        # 4x4 conv, 149 inputs, 73 outputs
        'wc2': tf.Variable(tf.random_normal([4, 4, 128, 64])),
        # 4x4 conv, 73 inputs, 32 outputs
        'wc3': tf.Variable(tf.random_normal([4, 4, 64, 32])),
        # fully connected, 38*8*32 inputs, 2^10 outputs
        'wd1': tf.Variable(tf.random_normal([41*8*32, 1024])),
        # 2^10 inputs, 4 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
}
biases = {
        'bc1': tf.Variable(tf.random_normal([128])+0.01),
        'bc2': tf.Variable(tf.random_normal([64])+0.01),
        'bc3': tf.Variable(tf.random_normal([32])+0.01),
        'bd1': tf.Variable(tf.random_normal([1024])+0.01),
        'out': tf.Variable(tf.random_normal([n_classes])+0.01)
}

def getBatch(data, labels, batchSize, iteration):
    startOfBatch = (iteration * batchSize) % len(data)
    endOfBatch = (iteration * batchSize + batchSize) % len(data)

    if startOfBatch < endOfBatch:
        return data[startOfBatch:endOfBatch], labels[startOfBatch:endOfBatch]
    else:
        dataBatch = np.vstack((data[startOfBatch:],data[:endOfBatch]))
        labelsBatch = np.vstack((labels[startOfBatch:],labels[:endOfBatch]))

        return dataBatch, labelsBatch

def conv2d(sound, w, b):
    # stride = 1
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(sound, w, strides=[1, 1, 1, 1],padding='SAME'), b))

def maxpool(x, k):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 644 ,128, 1])

    # 1st Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool(conv1,4)

    # 2nd Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool(conv2,2)

    # 3rd Convolution Layer
    conv3 = conv2d(conv2,weights['wc3'],biases['bc3'])
    # Max Pooling (down-sampling)
    conv3 = maxpool(conv3,2)

    # Reshape conv3 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1,weights['wd1'].get_shape().as_list()[0]])
    # Fully connected layer
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    # Relu activation
    fc1 = tf.nn.relu(fc1)
    # Apply dropout
    fc1 = tf.nn.dropout(fc1, dropout)


    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return out

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Add ops to save and restore all the variables.
# saver = tf.train.Saver()

# Launch the graph
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_xs, batch_ys = getBatch(trainData, trainLabels, batch_size, step)
        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})

        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

            # save_path = saver.save(sess, "model.ckpt")
            # print("Model saved in file: %s" % save_path)
        step += 1
    print("Optimization Finished!")

    # save_path = saver.save(sess, "model.final")
    # print("Model saved in file: %s" % save_path)

    # Calculate accuracy
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: testData,y: testLabels,keep_prob: 1.}))
