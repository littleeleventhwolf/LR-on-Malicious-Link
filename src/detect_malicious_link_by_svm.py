#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import random

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn import svm
from sklearn.cross_validation import train_test_split
#from sklearn.lda import LDA

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def getTokens(input):
	"""
	split input data and get the tokens
	"""
	# define the result tokens list
	allTokens = []
	# get tokens after splitting by slash
	tokensBySlash = str(input.encode('utf-8')).split('/')
	for i in tokensBySlash:
		# get tokens after splitting by dash
		tokensByDash = str(i).split('-')
		# record the tokens after splitting by dot
		tokensByDot = []
		for j in range(0, len(tokensByDash)):
			# get tokens after spliiting by dot
			tempTokens = str(tokensByDash[j]).split('.')
			tokensByDot = tokensByDot + tempTokens
		allTokens = allTokens + tokensByDash + tokensByDot
	# remove redundant tokens
	allTokens = list(set(allTokens))
	#print(allTokens)
	if 'com' in allTokens:
		# remove 'com' since it occurs a lot of times and
		# it should not be included in our features
		allTokens.remove('com')
	return allTokens

# the path of colleted link data
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
						 'data/data.csv')
#print(data_path)
# read file by pd
csv_data = pd.read_csv(data_path, ',', error_bad_lines=False)
#print(csv_data)
# converting to a dataframe
data_frame = pd.DataFrame(csv_data)
#print(data_frame)

# converting it into an array
data_array = np.array(data_frame)
#print(data_array)
# random shuffle the data
random.shuffle(data_array)


# get the label of data
# one-hot vector
y = []
for row in data_array:
	if row[1] == 'bad':
		y.append([1, 0])
	else:
		y.append([0, 1])
#print("=======")
#print(y)
# all links coressponding to a label (whether malicious or clean)
corpus = [row[0] for row in data_array]
#print("=======")
#print(corpus)
# get a vector for each url but use our customized tokenizer
vectorizer = TfidfVectorizer(tokenizer=getTokens)
print("=======")
#print(vectorizer)
# get the X vector
X = vectorizer.fit_transform(corpus)
print("=======")
# get the columns dimensions
vec_dimensions = np.shape(X)[1]

# split into training data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.005)
# svm training
# clf = svm.SVC(kernel='linear')
# clf.fit(X_train, y_train)
# print the test score
# print(clf.score(X_test, y_test))
# LDA training
#clf = LDA()
#clf.fit(X_train.toarray(), y_train)
#print(clf.score(X_test.toarray(), y_test))

#train_data = [X_train, y_train]
#test_data = [X_test, y_test]

x_tf = tf.placeholder(tf.float32, [None, vec_dimensions])

# Params
W_tf = tf.Variable(tf.zeros([vec_dimensions, 2]))
b_tf = tf.Variable(tf.zeros([2]))

y_tf = tf.nn.softmax(tf.matmul(x_tf, W_tf) + b_tf)
y_tf_ = tf.placeholder(tf.float32, [None, 2])

# loss function
cross_entropy = -tf.reduce_sum(y_tf_ * tf.log(y_tf))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# init
init = tf.initialize_all_variables()

# create Session
sess = tf.Session()
sess.run(init)

# train
n = np.shape(X_train)[0]
for i in range(10):
	print("Iteration ", i+1)
	#random.shuffle(train_data)
	for k in xrange(n / 100):
		print(k+1)
		# batch size : 100
		#batchesX = [X_train[k: k+100] for k in xrange(0, n, 100)]
		#batchesY = [y_train[k: k+100] for k in xrange(0, n, 100)]
		batch_xs, batch_ys = X_train[k: k+100].toarray(), y_train[k: k+100]
		#print(batch_xs)
		#print(batch_ys)
		sess.run(train_step, feed_dict={x_tf: batch_xs, y_tf_: batch_ys})

correct_prediction = tf.equal(tf.arg_max(y_tf, 1), tf.arg_max(y_tf_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print("Accuracy on Test-Data-Sets: ", sess.run(accuracy, feed_dict={x_tf: X_test.toarray(), y_tf_: y_test}))

sess.close()