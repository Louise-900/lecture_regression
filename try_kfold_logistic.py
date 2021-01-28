import pandas
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn import metrics


dataset = pandas.read_csv("dataset_3_outputs.csv")

target = dataset.iloc[:,2].values   #column 1#
data = dataset.iloc[:,3:9].values

kfold_object = KFold(n_splits = 4) #chop our dataset into 4 parts

kfold_object.get_n_splits(data)

#print(kfold_object)
# k fold runs 4 time the original time if you run the same thing 4 times.
i = 0
for training_index, test_index in kfold_object.split(data):
	
	print(i)
	i = i+1
	print("trai;ning:", training_index) 
	print("test", test_index)	
	data_training = data[training_index]
	data_test = data[test_index]
	target_training = target[training_index]
	target_test = target[test_index]

	machine = linear_model.LogisticRegression()
	machine.fit(data_training, target_training)
	new_target = machine.predict(data_test)
	#print("R2 score:", metrics.r2_score(target_test, new_target))
	print("accuracy score:", metrics.accuracy_score(target_test,new_target))
	#accuracy score: how m any of the values in the target test, is exactly the same with the new_target#
	#accuracy score is good prediction for dummy variable (categorical variable), R2 is actually a bad prediction for dummy
	print("Confusion Matrix:\n", metrics.confusion_matrix(target_test,new_target))
	

#qualitative test... relation,policy significancy... 
#depends on the assumption.
#assumptions required for a dummy variable: the distance between wage=1 and wage=2, and wage=2 and wage =3, 
#in agreement level, agreement and disagreement, 0-3, the distance is different.


#wage = [0 1 2 3]  assuming the distance between different number is the same
#happiness = [0 1 2 3]   4 levels--> culatative--> distance between 3 and 2, has nothing to do with 2 and 1. 

