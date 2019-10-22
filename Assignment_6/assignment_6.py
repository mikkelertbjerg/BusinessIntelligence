import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors

df = pd.read_json('users.json', orient='records')

#Unnesscary conversion (apparently)
#----------------------------------
#Split the strings and remove "e+09"
#df['time'] = df['created'].astype(str)
#df['time'] = df['time'].str.split('e').str[0]
#print("df 'time' converted to str and e+09 has been stripped away")
#Divide by 3600 to get time into hours rather than seconds
#Doing division first, otherwise we run out of ram
#df['time'] = df['time']*10**9
#[x * 10**9 for x in df['time']]
#print("df 'time' has been multiplied.")
#----------------------------------

#Prepare the time (submitted) and points (karma) data
#Submitted is divided by 60, as it's presumably minutes, and an hour representation is more relevant
df['time'] = df['created']/3600
df['points'] = df['karma']

#Prepare the scatter plot
x = df['time']
y = df['points']
plt.scatter(x, y)
plt.xlabel('Time (Hours)')
plt.ylabel('Points')
plt.show()

#Prepare the data for the machine
df = df[["time", "points"]]
#Got a conversion error, filled out all "nan" values with 0.
df = df.fillna(0)
df.astype('int64')

predict = "points"

x = np.array(df.drop([predict], 1))
y = np.array(df[predict])

#Didn't need 2d array anyway
#xy = np.column_stack((x, y))

#Verifying the arrays are as expected
#print(x, x.dtype, x.size, x.shape)
#print(y, y.dtype, y.size, x.shape)
#print(xy, xy.dtype, xy.size, xy.shape)


#Split the data in a 80-20 ratio
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
#xy_train, xy_test = train_test_split(xy, test_size = 0.2, random_state = 0)

#Verify the new datasets
#print("xy_train: ", xy_train.size)
#print("xy_test: ", xy_test.size)

#Should the data be pre-processed?
#scaler = StandardScaler().fit(x_train)
#standardized_x = scaler.transform(x_train)
#standardized_x_test = scaler.transform(x_test)
#print(standardized_x, standardized_x_test)
#-----------------------------------------------------------------------------

#Linear Regression
lr = LinearRegression(normalize=True)
lr.fit(x_train, y_train)
acc = lr.score(x_test, y_test)
print("Linear Regression Score: ", acc)

#Verifying the test results
"""
predictions = lr.predict(x_test)
for p in range(len(predictions)):
    print(predictions[p], x_test[p], y_test[p])
"""

#Linear regression appears to be a decent fit, seeing as we're trying to find
#the cooralation between a and b, and that's exactly what it does.
#-----------------------------------------------------------------------------

#Support Vector Machines (SVM)
"""
svc = SVC(kernel='linear')
svc.fit(x_train, y_train)
acc = svc.score(x_test, y_test)
print("SVM Score: ", acc)
"""

#SVM is a poor fit for our data set, seeing as we're not looking to clasify our data,
#in other words find the difference between a and b, but rather find a cooralation between a and b.
#-----------------------------------------------------------------------------

#Naive Bayes
"""
gnb = GaussianNB()
gnb.fit(x_train, y_train)
acc = gnb.score(x_test, y_test)
print("GNB Score: ", acc)
"""

#Naive Bayes is also a poor fit for our data set, like SVM, NB is a classification model, and once again
#We're not looking to classify our data, but rather find a cooralation in it.
#-----------------------------------------------------------------------------

#K Nearest Neighbor
"""
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
acc = knn.score(x_test, y_test)
print("KNN Score: ", acc)
"""

#KNN isn't the best fit either, as the algorithm, much like the two above, clasifies data.
#And once again, we're not trying to classify the data.
#However I can see how KNN could possibly be used in this particular case,
#but the results, as acc implies as well, are poor.
#-----------------------------------------------------------------------------