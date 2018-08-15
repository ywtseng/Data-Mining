import sys
import ssl
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.optimizers import SGD
from keras.datasets import cifar10
import keras

"""def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict"""
	
def Normalize(train,test):
	mean = np.mean(train,axis=(0,1,2,3))
	std = np.std(train, axis=(0, 1, 2, 3))
	train = (train-mean)/(std+1e-7)
	test = (test-mean)/(std+1e-7)
	return train, test

def WriteFile(predict_y):
	with open('predict.txt', 'w') as writer:
		for line in predict_y :
			writer.write(str(line))
			writer.write('\n')
	print("write the predict.txt")

def main():
	"""argu = sys.argv
	if len(argu) < 7 :
		print("Error augument")
		exit()
	x_train = []
	y_train = []
	#for i in range(1,6):
	for i in range(1,2):
		dict = unpickle(argu[i])
		for j in dict[b'data']:
			x_train.append(list(j))
		for j in dict[b'labels']:
			y_train.append(j)
	#set test data
	dict = unpickle(argu[6])
	x_test = dict[b'data']
	y_test = dict[b'labels']"""
	
	# Step 1:Parser
	(x_train,y_train),(x_test,y_test) = cifar10.load_data()
	x_train = x_train[0:10000]
	y_train = y_train[0:10000]
	
	# Step 2:Normalize
	y_train = keras.utils.to_categorical(y_train, 10)
	y_test = keras.utils.to_categorical(y_test, 10)
	x_train, x_test = Normalize(x_train, x_test)
	# Step 3:Model
	model = Sequential()
	#Conv layer 1 output shape (64, 32, 32)
	model.add( Convolution2D(64,3,3,border_mode='same',dim_ordering='tf',input_shape=(32,32,3)) )
	model.add( Activation('relu') )
	#Pooling layer 1 (max pooling) output shape (32, 16, 16)
	model.add( MaxPooling2D(pool_size=(2,2)) )
	
	#Conv layer 2 output shape (64, 32, 32)
	model.add( Convolution2D(64,3,3,border_mode='same') )
	model.add( Activation('relu') )
	#Pooling layer 2 (max pooling) output shape (32, 16, 16)
	model.add( MaxPooling2D(pool_size=(2,2)) )
	
	#Conv layer 3 output shape (256, 32, 32)
	model.add( Convolution2D(256,3,3,border_mode='same') )
	model.add( Activation('relu') )
	#Pooling layer 3 (max pooling) output shape (32, 16, 16)
	model.add( MaxPooling2D(pool_size=(2,2)) )
	
	model.add( Flatten() )
	model.add( Dense(1024) )
	model.add( Activation('relu') )
	model.add( Dropout(0.5) )
	model.add( Dense(10) )
	model.add( Activation('softmax') )
	model.summary()
	
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(x_train,y_train) 
	# Step 4 :Predict data
	predict_y = model.predict(x_test)
	WriteFile(predict_y)
	print("------------Finish-------------")

if __name__ =='__main__':
	main()