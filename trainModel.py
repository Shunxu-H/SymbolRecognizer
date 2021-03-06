# encoding: UTF-8
# Copyright 2016 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import tensorflowvisu
import math
import numpy
import os.path
from pathlib import Path
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

tf.set_random_seed(0)
width = 28;
height = 28;
# metaFile = 'digitRecognizer.meta'
sessionName = 'my-model'


# neural network with 1 layer of 10 softmax neurons
#
# · · · · · · · · · ·       (input data, flattened pixels)       X [batch, 784]        # 784 = 28 * 28
# \x/x\x/x\x/x\x/x\x/    -- fully connected layer (softmax)      W [784, 10]     b[10]
#   · · · · · · · ·                                              Y [batch, 10]

# The model is:
#
# Y = softmax( X * W + b)
#              X: matrix for 100 grayscale images of 28x28 pixels, flattened (there are 100 images in a mini-batch)
#              W: weight matrix with 784 lines and 10 columns
#              b: bias vector with 10 dimensions
#              +: add with broadcasting: adds the vector to each line of the matrix (numpy)
#              softmax(matrix) applies softmax on each line
#              softmax(line) applies an exp to each value then divides by the norm of the resulting line
#              Y: output matrix with 100 lines and 10 columns

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)


# saver = tf.train.Saver();
metaFile = Path(sessionName + '.meta');

if metaFile.is_file():
	saver = tf.train.import_meta_graph(sessionName + '.meta');
else:
	mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

	# # input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
	X = tf.placeholder(tf.float32, [None, width, height, 1])
	# # correct answers will go here
	Y_ = tf.placeholder(tf.float32, [None, 10])
	# # weights W[784, 10]   784=28*28
	# W = tf.Variable(tf.zeros([784, 10]))
	# # biases b[10]
	# b = tf.Variable(tf.zeros([10]))

	K = 6
	L = 12
	M = 24
	N = 200

	pkeep = tf.placeholder(tf.float32)
	# 5 x 5 fully output
	# shrink to 

	W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1))  # 5x5 patch, 1 input channel, K output channels
	B1 = tf.Variable(tf.ones([K])/10)
	W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
	B2 = tf.Variable(tf.ones([L])/10)
	W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
	B3 = tf.Variable(tf.ones([M])/10)

	W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
	B4 = tf.Variable(tf.ones([N])/10)
	W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
	B5 = tf.Variable(tf.ones([10])/10)

	# The model
	stride = 1  # output is 28x28
	Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
	stride = 2  # output is 14x14
	Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
	stride = 2  # output is 7x7
	Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

	# reshape the output from the third convolution for the fully connected layer
	YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])
	YYd = tf.nn.dropout(YY, pkeep)
	Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
	Ylogits = tf.matmul(Y4, W5) + B5
	Y = tf.nn.softmax(Ylogits)

	tf.add_to_collection('Y', Y)
	tf.add_to_collection('X', X)


	lr = tf.placeholder(tf.float32)
	# Y = tf.nn.softmax(tf.matmul(XX, W) + b)
	# loss function: cross-entropy = - sum( Y_i * log(Yi) )
	#                           Y: the computed output vector
	#                           Y_: the desired output vector

	# cross-entropy
	# log takes the log of each element, * multiplies the tensors element by element
	# reduce_mean will add all the components in the tensor
	# so here we end up with the total cross-entropy for all images in the batch
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)  # normalized for batches of 100 images,
	cross_entropy = tf.reduce_mean(cross_entropy)*100              # *10 because  "mean" included an unwanted division by 10

	# accuracy of the trained model, between 0 (worst) and 1 (best)
	correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# training, learning rate = 0.005
	train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

	# matplotlib visualisation
	allweights = tf.reshape(W2, [-1])
	allbiases = tf.reshape(B2, [-1])
	I = tensorflowvisu.tf_format_mnist_images(X, Y, Y_)  # assembles 10x10 images by default
	It = tensorflowvisu.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)  # 1000 images on 25 lines
	datavis = tensorflowvisu.MnistDataVis()

	# init
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)


	# You can call this function in a loop to train the model, 100 images at a time
	def training_step(i, update_test_data, update_train_data):

	    # training on batches of 100 images with 100 labels
	    batch_X, batch_Y = mnist.train.next_batch(100)
	    lrmax = 0.003;
	    lrmin = 0.0001;
	    decay_speed = 2000.0;
	    learningRate = lrmin+(lrmax-lrmin)*math.exp(-i/decay_speed);

	    # compute training values for visualisation
	    if update_train_data:
	        a, c, im, w, b = sess.run([accuracy, cross_entropy, I, allweights, allbiases], feed_dict={X: batch_X, Y_: batch_Y, pkeep: 1.0})
	        datavis.append_training_curves_data(i, a, c)
	        datavis.append_data_histograms(i, w, b)
	        datavis.update_image1(im)
	        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " with learning " + str(learningRate))

	    # compute test values for visualisation
	    if update_test_data:
	        a, c, im = sess.run([accuracy, cross_entropy, It], feed_dict={X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1.0})
	        datavis.append_test_curves_data(i, a, c)
	        datavis.update_image2(im)
	        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))

	    # the backpropagation training step
	    sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y, lr: learningRate, pkeep: 0.75})


	datavis.animate(training_step, iterations=1000+1, train_data_update_freq=10, test_data_update_freq=50, more_tests_at_start=True)

	# to save the animation as a movie, add save_movie=True as an argument to datavis.animate
	# to disable the visualisation use the following line instead of the datavis.animate line
	# for i in range(2000+1): training_step(i, i % 50 == 0, i % 10 == 0)
	saver = tf.train.Saver()
	saver.save(sess, sessionName)

	print("max test accuracy: " + str(datavis.get_max_test_accuracy()))
