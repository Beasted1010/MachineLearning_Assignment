import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

housing_data_set = pd.read_csv('boston_housing.txt', header=None)
housing_data_set = housing_data_set.values
#print(housing_data_set)
shape_list = housing_data_set.reshape(housing_data_set.shape[0])
split_list = [val.split() for val in shape_list]
float_list = [ [float(y) for y in x] for x in split_list ]

median_values = [x[13] for x in float_list]
features = [x[:13] for x in float_list]

# Training set
X_train = features[:400]
Y_train = median_values[:400]

# Test set
X_test = features[400:]
Y_test = median_values[400:]

#Linear regression object
regr = linear_model.LinearRegression()

# Train the model
regr.fit(X_train, Y_train)

# Make the prediction
housing_pred = regr.predict(X_test)
#print(housing_pred)
#print('\n\n')
#print(Y_test)

# Compute the error
print("Mean squared error: {}".format(mean_squared_error(Y_test, housing_pred)))

#Plot output
#import matplotlib.pyplot as plt
#plt.scatter(X_test, Y_test, color='black')
#plt.plot(X_test, housing_pred, color='blue', linewidth=3)



