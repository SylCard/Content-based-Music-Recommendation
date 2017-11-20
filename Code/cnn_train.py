from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
from cnn_input import extract_features,one_hot_encode
tf.logging.set_verbosity(tf.logging.INFO)



#this method defines the convolutional network
def cnn_model(features, labels, mode):
  """Model function for CNN."""
  # Input Layer

  # Reshape X to 2-D tensor: [batch_size, width, height, channels]
  # mel-spectrogram are 128x1293 pixels, and have one color channel
  input_layer = tf.cast(tf.reshape(features["x"], [-1, 1292,128, 1]),tf.float32)

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=5,
      padding="same",
      activation=tf.nn.relu)
  """kernel_size specifies the dimenstions of the filter
  padding ensures the output tensor is the same size as the input tensor"""
  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=5,
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  # We flatten the mp of layer 2 before feeding into fc layer
  # Tensor only has two dimensions
  pool2_flat = tf.reshape(pool2, [-1, 32 * 323 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=2)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  # For classification problems, cross entropy is typically used as the loss metric
  # The labels tensor contains a list of genres for our examples and we convert them to
  # one-hot encoding which will use boolean values to represent a category
  onehot_labels = tf.cast(labels, tf.int32)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # The following code will optimise the cnn so that it learns after each example
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode) which will output an accuracy value
  eval_metric_ops = { "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}

  # set up tensorboard visualisation
  writer_1 = tf.summary.FileWriter("/tmp/train")
  writer_2 = tf.summary.FileWriter("/tmp/test")
  writer_1.add_graph(sess.graph)
  tf.summary.scalar('Loss', loss)
  tf.summary.scalar('Accuracy',  eval_metric_ops['accuracy'])
  write_op = tf.summary.merge_all()


  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

#the main method will allow the loading of training and test data
def main(unused_argv):
  # Load training and eval data
  train_data, train_labels  = extract_features("../../miniDataset") # Returns np.array of features and labels for each song
  eval_data, eval_labels  = extract_features("../../evalDataset")  # Returns np.array
  print (train_labels)

  #the estimator allows us to carry out high level training
  #model_dir will store the dir that will hold model checkpoints
  genre_classifier = tf.estimator.Estimator(model_fn=cnn_model, model_dir="tmp")

  # Set up logging for predictions
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=1)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=1,
    num_epochs=None,
    shuffle=True)

  genre_classifier.train(input_fn=train_input_fn,steps=22,hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)
  eval_results = genre_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
    tf.app.run()
