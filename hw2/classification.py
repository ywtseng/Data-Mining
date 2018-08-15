import sys
import ssl
import matplotlib.pyplot as plt
import urllib.request
import numpy as np
import csv
#Regression
from sklearn import datasets
from sklearn import linear_model
#SVM
from sklearn import svm
#DecisionTree
from sklearn.tree import DecisionTreeClassifier
#NeuralNetwork
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

def ChooseModel(mode):
	if mode=="R":
		#model = linear_model.LogisticRegression(C=4,solver='liblinear',intercept_scaling=4)
		model = linear_model.LogisticRegression()
	elif mode=="D":
		#model = DecisionTreeClassifier(max_depth=None,min_samples_split=4,min_samples_leaf=1,min_weight_fraction_leaf=0.0)
		model = DecisionTreeClassifier()
	elif mode=="S":
		#model = svm.SVC(C=50000,kernel='rbf',gamma=0.00001)
		#model=svm.SVC(decision_function_shape='ovo')
		#model=svm.SVC(gamma=0.001,C=100,kernel='rbf')
		model = svm.SVC()
	elif mode=="N":
		#model = MLPClassifier(hidden_layer_sizes=(5000), max_iter=20, alpha=1e-5,solver='adam', verbose=10, tol=1e-4, random_state=1,learning_rate_init=0.01)
		model = MLPClassifier()
	return model

def StringtoFloat(string_data):
	#convert string to float
	float_data = []
	for i in string_data:
		row = []
		for j in i:
			row.append(float(j))
		float_data.append(row)
	return float_data
	
def Verification(model,train_x,train_y):
	x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.33,shuffle=False)
	model.fit(x_train,y_train)
	predict_y = model.predict(x_test)
	#compare whether correct
	wrong_num = 0
	num = 0
	for i in predict_y:
		if i != y_test[num]:
			wrong_num = wrong_num+1
		num = num+1
	print("Total test    = ",num)
	print("Wrong number  = ",wrong_num)
	print("Correct ratio = ",(num-wrong_num)/num)
	
def main():	
	argu = sys.argv
	if len(argu) < 4 :
		print("No argument")
		sys.exit()
	# step 1: parser -----------------------------------------------------
	# read train.csv file #argu[2] -> train.csv #argu[3] -> test.csv
	trainfile = csv.reader(open(argu[2]))
	testfile = csv.reader(open(argu[3]))
	# train data
	train_x = list(trainfile)
	train_y = []
	for data_row in train_x:
		test_value=data_row.pop()
		train_y.append(int(test_value))
	train_x = StringtoFloat(train_x)
	# test data
	test_x = list(testfile)
	
	# step 2: choose model(Regression / Decision Tree / SVM / Neural Network)
	model = ChooseModel(argu[1])
	if argu[1]== "N" or "R":
		test_x = StringtoFloat(test_x)
		
	# step 3: verification -----------------------------------------------
	Verification(model,train_x,train_y)

	# step 4: machine learngin--------------------------------------------
	model.fit(train_x,train_y)
	test_y = []
	test_y = model.predict(test_x)
	
	# step 5: write predict.csv-------------------------------------------
	if len(test_y)!=0 :
		with open('predict.csv', 'w') as writer:
			for line in test_y :
				writer.write(str(line))
				writer.write('\n')
		print("write the predict.csv")
	
	
		
if __name__ =='__main__':
	main()
