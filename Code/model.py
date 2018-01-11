import tensorflow as tf
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
np.set_printoptions(threshold='nan')
#function to create convolutional layers
def conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),name=name+'_W')

    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

    # add the bias
    out_layer += bias

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)

    # now perform max pooling
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides,
                               padding='SAME')

    return out_layer

#function to get batch data for given epoch
def getBatch(data, labels, batchSize, iteration):
    startOfBatch = (iteration * batchSize) % len(data)
    endOfBatch = (iteration * batchSize + batchSize) % len(data)

    if startOfBatch < endOfBatch:
        return data[startOfBatch:endOfBatch], labels[startOfBatch:endOfBatch]
    else:
        dataBatch = np.vstack((data[startOfBatch:],data[:endOfBatch]))
        labelsBatch = np.vstack((labels[startOfBatch:],labels[:endOfBatch]))

        return dataBatch, labelsBatch


#import dataset here using pickle
data = []
labels = []
with open('training.data', 'r') as f:
  data = pickle.load(f)
with open('training.labels', 'r') as f:
  labels = pickle.load(f)

# Shuffle data
permutation = np.random.permutation(data.shape[0])
data = data[permutation]
labels = labels[permutation]

#split dataset into training and evaluation datasets  = 75% training, 25% test
trainData, testData = train_test_split(data,test_size=0.25)
trainLabels, testLabels = train_test_split(labels,test_size=0.25)


# #convert labels into one-hot form
# trainLabels = tf.one_hot(indices=trainLabels, depth=2)
# testLabels = tf.one_hot(indices=testLabels, depth=2)

# Python optimisation variables
learning_rate = 0.001
num_of_epochs = 5
batch_size = 150
dropout = 0.75

# declare the training data placeholders
# input x - for 644 x 128 pixels = 82816 - this is the flattened image data that is drawn from
x = tf.placeholder(tf.float32, [None, 82432])
x_reshaped = tf.reshape(x, [-1, 644, 128, 1]) # dynamically reshape the input
y = tf.placeholder(tf.float32, [None, 10]) # placeholder for corresponding labels- 10 genres
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

#output from first layer will halve the dimensions of the input 322 x 64 due to max pooling having a stride of 2
# create some convolutional layers
hidden_layer1 = conv_layer(x_reshaped, 1, 32, [4, 1], [2, 2], name='layer1')
hidden_layer2 = conv_layer(hidden_layer1, 32, 64, [4, 1], [2, 2], name='layer2')

flattened = tf.reshape(hidden_layer2, [-1,  161 * 32 * 64])

# setup some weights and bias values for the fully connected layer, then activate with ReLU
wd1 = tf.Variable(tf.truncated_normal([161 * 32 * 64, 1000], stddev=0.03), name='wd1')
bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')
dense_layer1 = tf.matmul(flattened, wd1) + bd1
#Relu activation
dense_layer1 = tf.nn.relu(dense_layer1)
#Apply dropout here
dense_layer1 = tf.nn.dropout(dense_layer1, keep_prob)
# Softmax Classifier layer
wd2 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.03), name='wd2')
bd2 = tf.Variable(tf.truncated_normal([10], stddev=0.01), name='bd2')
logits = tf.matmul(dense_layer1, wd2) + bd2
y_ = tf.nn.softmax(logits)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

# add an optimiser
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# setup the initialisation operator
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)
    #currently have 150 songs for 5 batches
    # total_batch = int(44 / batch_size)
    for epoch in range(num_of_epochs):
        #average cross_entropy for each epoch
        avg_cost = 0
        # for i in range(batch_size):
        batch_x, batch_y = getBatch(trainData, trainLabels, batch_size, epoch)
        # train network with batch data
        _, cost, batch_acc = sess.run([optimiser, cross_entropy,accuracy],feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        #output accuracy and loss for batch
        print "Epoch:", (epoch + 1) , "accuracy: {:.3f}".format(batch_acc), "loss: {:.3f}".format(cost)

    print "Training complete!"

    final_acc, softmaxOutput = sess.run([accuracy,y_], feed_dict={x: testData, y: testLabels,keep_prob: 1.0})
    print "final accuracy: " , final_acc
    print softmaxOutput
