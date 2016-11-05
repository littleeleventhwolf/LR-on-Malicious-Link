#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import random

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.lda import LDA

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
#random.shuffle(data_array)


# get the label of data
y = []
for row in data_array:
	if row[1] == 'bad':
		y.append(0)
	else:
		y.append(1)
#print("=======")
#print(y)
# all links coressponding to a label (whether malicious or clean)
corpus = [row[0] for row in data_array]
#print("=======")
#print(corpus)
# get a vector for each url but use our customized tokenizer
vectorizer = TfidfVectorizer(tokenizer=getTokens)
print("=======")
print(vectorizer)
# get the X vector
X = vectorizer.fit_transform(corpus)
print("=======")
print(X)


#  split into training data and test data
X_train, X_test, y_train, y_test = train_test_split(X[42700:42800], y[42700:42800], test_size=0.2, random_state=42)

# svm training
# clf = svm.SVC(kernel='linear')
# clf.fit(X_train, y_train)
# print the test score
# print(clf.score(X_test, y_test))

# LDA training
clf = LDA()
clf.fit(X_train.toarray(), y_train)
print(clf.score(X_test.toarray(), y_test))