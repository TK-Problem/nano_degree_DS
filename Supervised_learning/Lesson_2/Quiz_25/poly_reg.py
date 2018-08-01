import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Assign the data to predictor and outcome variables
train_data = pd.read_csv("data.csv")
X = train_data['Var_X'].values.reshape(-1,1)
y = train_data['Var_Y'].values

# Create polynomial features
poly_feat = PolynomialFeatures(degree=4)
X_poly = poly_feat.fit_transform(X)

# Make and fit the polynomial regression model
poly_model = LinearRegression(fit_intercept = False).fit(X_poly, y)

# Once you've completed all of the steps, select Test Run to see your model
# predictions against the data, or select Submit Answer to check if the degree
# of the polynomial features is the same as ours!