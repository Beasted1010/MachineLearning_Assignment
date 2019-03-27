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
X_train = features[:20]
Y_train = median_values[:20]

# Test set
X_test = features[20:]
Y_test = median_values[20:]

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
print("Linear regression mean squared error: {}".format(mean_squared_error(Y_test, housing_pred)))

# RIDGE REGRESSION
regr = linear_model.RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]) #, cv=3)
#regr = linear_model.Ridge(alpha = 0.5)
regr.fit(X_train, Y_train)
housing_pred = regr.predict(X_test)
print("Alpha used: {}".format(regr.alpha_))

print("Ridge regression mean squared error: {}".format(mean_squared_error(Y_test, housing_pred)))



