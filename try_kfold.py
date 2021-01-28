import pandas
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn import metrics
from matplotlib import pyplot
import numpy

dataset = pandas.read_csv("dataset_3_outputs.csv")

target = dataset.iloc[:,0].values
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
	machine = linear_model.LinearRegression()
	machine.fit(data_training, target_training)
	new_target = machine.predict(data_test)
	print("R2 score:", metrics.r2_score(target_test, new_target))

	new_target_newaxis = new_target[:,numpy.newaxis]
	results_machine = linear_model.LinearRegression()
	results_machine.fit(new_target_newaxis,target_test)
	pyplot.scatter(new_target_newaxis,target_test)
	pyplot.plot(new_target_newaxis, results_machine.predict(new_target_newaxis), color='k')
	pyplot.savefig('scatterplot' + str(i) +'.png')
	#pyplot.clf()

