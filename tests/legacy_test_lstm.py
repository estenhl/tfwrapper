import sys
sys.path.append('.')

import numpy as np
import tensorflow as tf
from nets.lstm import LSTM
from random import shuffle
 
train_input = ['{0:020b}'.format(i) for i in range(10000)]
shuffle(train_input)
train_input = [map(int,i) for i in train_input]
ti  = []
for i in train_input:
	temp_list = []
	for j in i:
		temp_list.append([j])
	ti.append(np.array(temp_list))
train_input = ti

train_output = []
 
for i in train_input:
	count = 0
	for j in i:
		if j[0] == 1:
			count+=1
	temp_list = ([0]*21)
	temp_list[count]=1
	train_output.append(temp_list)

print(train_input[:5])
print(train_output[:5])
with tf.Session() as sess:
	lstm = LSTM([20, 1], 21, 24, sess=sess, name='Test')
	lstm.train(np.asarray(train_input), np.asarray(train_output), epochs=100, sess=sess)
