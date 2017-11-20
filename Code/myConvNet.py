#imports
import math
import numpy as np
import tensorflow as tf
import pickle
    if __name__ == "__main__":

    # Parameters
    learning_rate = 0.001
    training_iters = 100000
    batch_size = 64
    display_step = 1
    train_size = 23

    # Network Parameters
    # n_input = 599 * 128
    n_input = 599*128*2
    n_classes = 10
    dropout = 0.75  # Dropout, probability to keep units

    # Load data
    data = []
    with open("data", 'r') as f:
        content = f.read()
        data = pickle.loads(content)
    data = np.asarray(data)
    data = data
    data = data.reshape((data.shape[0], n_input))

    labels = []
    with open("labels", 'r') as f:
        content = f.read()
        labels = pickle.loads(content)

    # #Hack
    # data = np.random.random((1000, n_input))
    # labels = np.random.random((1000, 10))

    # Shuffle data
    permutation = np.random.permutation(len(data))
    data = data[permutation]
    labels = labels[permutation]

    # Split Train/Test
    trainData = data[:train_size]
    trainLabels = labels[:train_size]

    testData = data[train_size:]
    testLabels = labels[train_size:]


    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)




    def conv_net(_X, _weights, _biases, _dropout):
        # Reshape the input _X
        #The -1 argument shapes the tensor into one dimension
        _X = tf.reshape(_X, shape=[-1, 599, 128, 2])

        # Convolution Layer #1
        conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
        # Max Pooling  (down-sampling)
        conv1 = max_pool(conv1, k=4)
        # Apply Dropout
        conv1 = tf.nn.dropout(conv1, _dropout)

        # Convolution Layer #2
        conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = max_pool(conv2, k=2)
        # Apply Dropout
        conv2 = tf.nn.dropout(conv2, _dropout)

        # Convolution Layer #3
        conv3 = conv2d(conv2, _weights['wc3'], _biases['bc3'])
        # Max Pooling (down-sampling)
        conv3 = max_pool(conv3, k=2)
        # Apply Dropout
        conv3 = tf.nn.dropout(conv3, _dropout)

        # Fully connected layer
        # Reshape conv3 output to fit dense layer input
        dense1 = tf.reshape(conv3, [-1, _weights['wd1'].get_shape().as_list()[0]])
        # Relu activation
        dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1']))
        # Apply Dropout
        dense1 = tf.nn.dropout(dense1, _dropout)  # Apply Dropout

        # Output, class prediction
        out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
        return out
