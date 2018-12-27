# Import statements
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Read the data.
data = np.asarray(pd.read_csv('data.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y.
X = data[:,0:2]
y = data[:,2]

# Create the model and assign it to the variable model.
# Find the right parameters for this model to achieve 100% accuracy on the dataset.
model = SVC(C=1.0,kernel = 'rbf',degree=3,gamma=27.0)

# Fit the model.
model.fit(X, y)

# Make predictions. Store them in the variable y_pred.
y_pred = model.predict(X)

# Calculate the accuracy and assign it to the variable acc.
acc = accuracy_score(y,y_pred)