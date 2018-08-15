import numpy 	as np
import pandas 	as pd
import sklearn
import matplotlib.pyplot as plt


from sklearn.preprocessing 		import LabelEncoder
from sklearn.model_selection 	import cross_val_score,cross_val_predict
from sklearn.neighbors 			import KNeighborsClassifier
from sklearn.linear_model 		import LogisticRegression
from sklearn				 	import datasets,tree
from sklearn.neural_network  	import MLPClassifier
from sklearn.metrics			import accuracy_score
from matplotlib.colors 			import ListedColormap

def parse_data():
	battles = pd.read_csv("battles.csv")
	character_deaths = pd.read_csv("character-deaths.csv")
	character_predictions = pd.read_csv("character-predictions.csv")

	# convert wins and loss into int
	battles['attacker_outcome'].replace('win',1,inplace=True)
	battles['attacker_outcome'].replace('loss',0,inplace=True)	
	
	battles.attacker_1.fillna(0,inplace=True)
	battles.attacker_2.fillna(0,inplace=True)
	battles.attacker_3.fillna(0,inplace=True)
	battles.attacker_4.fillna(0,inplace=True)
	battles.defender_1.fillna(0,inplace=True)
	battles.defender_2.fillna(0,inplace=True)
	battles.defender_3.fillna(0,inplace=True)
	battles.defender_4.fillna(0,inplace=True)
	
	battles.attacker_king.fillna('None',inplace=True)
	battles.defender_king.fillna('None',inplace=True)
	battles.battle_type.fillna('None',inplace=True)
	battles.attacker_commander.fillna('None',inplace=True)
	battles.defender_commander.fillna('None',inplace=True)
	battles.location.fillna('None',inplace=True)
	battles.region.fillna('None',inplace=True)

	battles.fillna(0,inplace=True)

	le = LabelEncoder()
	battles.attacker_king 		= le.fit_transform(battles.attacker_king)
	battles.defender_king 		= le.fit_transform(battles.defender_king)
	battles.battle_type 		= le.fit_transform(battles.battle_type)
	battles.attacker_commander 	= le.fit_transform(battles.attacker_commander)
	battles.defender_commander 	= le.fit_transform(battles.defender_commander)
	battles.location 			= le.fit_transform(battles.location)
	battles.region 				= le.fit_transform(battles.region)
	
	
	#variations on attacker_outcom based on year are very less hence dropping the column
	battles.drop('year',axis=1,inplace=True)
	#name of the battle is insignificant to the outcome
	battles.drop(['name','battle_number','note'],axis=1,inplace=True)
	#drop attacker_2 , attacker_3, attacker_4
	battles.drop(['attacker_2','attacker_3','attacker_4'],axis=1,inplace=True)
	battles.drop(['defender_2','defender_3','defender_4'],axis=1,inplace=True)
	battles.drop(['attacker_1','defender_1'],axis=1,inplace=True)
	
	
	X = battles.drop('attacker_outcome',axis=1)
	y = battles.attacker_outcome
	
	return X, y


def knn_model():
	knn = KNeighborsClassifier(weights='distance')
	
	return knn
	
def logistic_regression_model():
	logr = LogisticRegression()
	
	return logr

def decision_tree_model():
	dct = sklearn.tree.DecisionTreeClassifier()
	
	return dct
	
def svm_model():
	sklearn.svm.SVC(
		C=1.0, 
		kernel='rbf', 
		degree=5, 
		gamma='auto', 
		decision_function_shape='ovr'
		)
	
	svc_model = sklearn.svm.SVC(gamma=0.001, C=100.)
	
	return svc_model
	
def neuron_network_model():
	MLPClassifier(
		hidden_layer_sizes=(100, ), 
		activation='relu', 
		solver='adam', 
		alpha=0.0001, 
		batch_size='auto', 
		learning_rate='constant', 
		learning_rate_init=0.001, 
		max_iter=200, 
		shuffle=True, 
		momentum=0.9,)		
	
	mlp = MLPClassifier(hidden_layer_sizes = (20,))

	return mlp

def plot(X,y,model,title):


	# we only take the first two features. We could avoid this ugly
	# slicing by using a two-dim dataset
	X = X.values[:,[2,4]]
	y = y.values	
	
	h = .02  # step size in the mesh

	# Create color maps
	cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
	cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

	# we create an instance of Neighbours Classifier and fit the data.
	#clf = KNeighborsClassifier()
	model.fit(X, y)

	# Plot the decision boundary. For that, we will assign a color to each
	# point in the mesh [x_min, x_max]x[y_min, y_max].
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
						 np.arange(y_min, y_max, h))
	Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

	# Put the result into a color plot
	Z = Z.reshape(xx.shape)
	plt.figure()
	plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

	# Plot also the training points
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
				edgecolor='k', s=20)
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())

	plt.title( title )


	plt.show()
	plt.close()
	
	
if __name__ == '__main__':

	X, y = parse_data()
	
	
	knn = knn_model()
	log = logistic_regression_model()
	dct = decision_tree_model()
	svm = svm_model()
	mlp = neuron_network_model()
	
	knn_pred = cross_val_predict(knn,X,y)
	log_pred = cross_val_predict(log,X,y)	
	dct_pred = cross_val_predict(dct,X,y)	
	svm_pred = cross_val_predict(svm,X,y)
	mlp_pred = cross_val_predict(mlp,X,y)
	
	
	print( "knn: ", accuracy_score(knn_pred,y).mean() )
	print( "log: ", accuracy_score(log_pred,y).mean() )
	print( "dct: ", accuracy_score(dct_pred,y).mean() )
	print( "svm: ", accuracy_score(svm_pred,y).mean() )
	print( "mlp: ", accuracy_score(mlp_pred,y).mean() )
	
	# plot decision tree
	tree.export_graphviz(dct.fit(X,y), out_file='tree.dot')  
	
	plot(X,y,knn,"knn")
	plot(X,y,log,"log")
	plot(X,y,dct,"dct")
	plot(X,y,svm,"svm")
	plot(X,y,mlp,"mlp")
	
	
	
	