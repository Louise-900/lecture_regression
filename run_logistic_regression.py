import pandas
import sklearn
from sklearn import linear_model  


#read csv file using pandas## pd.read_csv is also fine
dataset = pandas.read_csv("dataset_3_outputs.csv")
#set everything equals to a variable called 'dataset', and it will contain everything incide it

#print(dataset)

#read the first collumn
#0- 0 collumn
# :  you target for the whole collumn
# [0,0] is for collumn 0 and row 0, which locates a single number. to reveal it, dont need to include ".values"1d
target = dataset.iloc[:,1].values

data = dataset.iloc[:,3:9].values
# 6 collumn

#print(data)

#machine = linear_model.LinearRegressoin()  
machine = linear_model.LogisticRegression()
#set up linear machine#
machine.fit(data, target)

print(machine)

#same collumn with the my data
new_data = [
	[0.01, -0.2, 0.5, 1.1, 0, 0],#1st object
	[-0.5, -0.1, 0.44, -0.9, 1, 0.5] #2nd object
]

new_target = machine.predict(new_data)

print(new_target)
