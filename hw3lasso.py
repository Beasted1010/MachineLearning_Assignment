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

# Adding some SWEngineering change that requires no intervention
# Adding some important comment and new variable
# ADDING SOME STUFF TO FEATURE BRANCH, VERY COOL FEATURE BRANCH
useless_variable_for_SWEngineering = 780
# This value should be larger than 102, coming in real quick for a hot fix
new_feature_variable = 1000002
# ADDING SOME HUMAN INTERVENING CHANGE, CHANGING NUMBER IN BOTH REPOs
# Some things being done to the master branch
# Some other thing being added

# ANOTHER non human intervening change needed for SWEngineering

# Training set
X_train = features[:100]
Y_train = median_values[:100]

# Test set
X_test = features[100:]
Y_test = median_values[100:]

print("Lasso with alpha set to 10...")

# Create the model
regr = linear_model.Lasso(alpha=10)

# Train the model
regr.fit(X_train, Y_train)

# Make the prediction
housing_pred = regr.predict(X_test)
#print(housing_pred)
print("Coefficients used:")
print(regr.coef_)

# Compute the error
print("Lasso mean squared error: {}".format(mean_squared_error(Y_test, housing_pred)))

print("\n\nLasso with cross validation to select alpha value...")
regr = linear_model.RidgeCV(alphas=[0.01, 0.1, 1, 10, 100]) #, cv=3)
regr.fit(X_train, Y_train)
housing_pred = regr.predict(X_test)
print("Alpha used: {}".format(regr.alpha_))
#print(housing_pred)
#print("Coefficients used: {}".format(regr.coef_))
print("Number of zero coefficients: {}".format(len([z for z in regr.coef_ if z == 0])))
print("Lasso mean squared error: {}".format(mean_squared_error(Y_test, housing_pred)))


