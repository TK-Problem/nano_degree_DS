import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
import pandas as pd

# Assign the data to predictor and outcome variables
# Load the data
train_data = pd.read_csv('data.csv')
X = train_data.iloc[:,:-1]
y = train_data.iloc[:,-1]

# Create the standardization scaling object.
scaler = StandardScaler()

# Fit the standardization parameters and scale the data.
X_scaled = scaler.fit_transform(X)

# Create the linear regression model with lasso regularization.
lasso_reg = Lasso()

# Fit the model.
lasso_reg.fit(X_scaled, y)

# Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_reg.coef_
print(reg_coef)