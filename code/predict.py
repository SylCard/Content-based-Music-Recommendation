import tensorflow as tf
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
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
    strides = [1, pool_shape[0], pool_shape[1], 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides,padding='SAME')

    return out_layer
#function to get batch data for given epoch
def getBatch(data, batchSize, iteration):
    startOfBatch = (iteration * batchSize) % len(data)
    endOfBatch = (iteration * batchSize + batchSize) % len(data)

    if startOfBatch < endOfBatch:
        return data[startOfBatch:endOfBatch]
    else:
        dataBatch = np.vstack((data[startOfBatch:],data[:endOfBatch]))


        return dataBatch
#import target data here using joblib for the user testing !!! :)
targetData = joblib.load('userChosenSongs.data')
# classicData = joblib.load('classicalFINAL.data')
# rockData = joblib.load('rockFINAL.data')
# hipData = joblib.load('hiphopFINAL.data')
#
# targetData = np.append(classicData,rockData, axis=0)
# targetData = np.append(targetData,hipData, axis=0)

print targetData.shape

num_of_epochs = 2
batch_size = 7

# input x - for 1288 x 128 pixels = 164864 - this is the flattened image data that is drawn from
x = tf.placeholder(tf.float32, [None, 164864])
x_reshaped = tf.reshape(x, [-1, 1288, 128, 1]) # dynamically reshape the input

keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

# create some convolutional layers
hidden_layer1 = conv_layer(x_reshaped, 1, 128, [4, 4], [4, 4], name='layer1')
hidden_layer2 = conv_layer(hidden_layer1, 128, 64, [4, 4], [2, 2], name='layer2')
hidden_layer3 = conv_layer(hidden_layer2, 64, 32, [4, 4], [2, 2], name='layer3')


flattened = tf.reshape(hidden_layer3, [-1,  81 * 8 * 32])
# setup some weights and bias values for the fully connected layer, then activate with ReLU
wd1 = tf.Variable(tf.truncated_normal([81 * 8 * 32, 1024], stddev=0.03), name='wd1')
bd1 = tf.Variable(tf.truncated_normal([1024], stddev=0.01), name='bd1')
dense_layer1 = tf.matmul(flattened, wd1) + bd1
#Relu activation
dense_layer1 = tf.nn.relu(dense_layer1)
#Apply dropout here
dense_layer1 = tf.nn.dropout(dense_layer1, keep_prob)
# Softmax Classifier layer
wd2 = tf.Variable(tf.truncated_normal([1024, 3], stddev=0.03), name='wd2')
bd2 = tf.Variable(tf.truncated_normal([3], stddev=0.01), name='bd2')
logits = tf.matmul(dense_layer1, wd2) + bd2
y_ = tf.nn.softmax(logits)


# setup the initialisation operator
init_op = tf.global_variables_initializer()

# allows us to save and restore the model's weights and biases for further use after training
saver = tf.train.Saver()

predictions = []

with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)

    saver.restore(sess, 'tmp/model.ckpt')

    for epoch in range(num_of_epochs):
        data = getBatch(targetData,batch_size,epoch)
        # predict and store output for these songs
        softmaxOutput = sess.run(y_, feed_dict={x: data,keep_prob: 1.0})
        print softmaxOutput
        if epoch == 0 :
            predictions = softmaxOutput
        else:
            predictions = np.vstack((predictions,softmaxOutput))

print predictions.shape
np.set_printoptions(suppress=True)
print predictions

# store predictions in serialised format for later processing
joblib.dump(predictions,'UserChosenSongs.prediction')
