import numpy as np
from sklearn.linear_model import Lasso

# Assign the data to predictor and outcome variables
# Load the data
train_data = np.loadtxt("data.csv",delimiter=",")
X = train_data[:,:6]
y = train_data[:,6]

# Create the linear regression model with lasso regularization.
lasso_reg = Lasso()

# Fit the model.
lasso_model = lasso_reg.fit(X, y)

# Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_model.coef_
print(reg_coef)