"""
    Simple file to create a sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pickle
# suppress warnings from final output
import warnings

# Fetch training data and preprocess for modeling
train = pd.read_csv('train_clean.csv')

y_train = train[['load_shortfall_3h']]
X_train = train.drop(columns = ['load_shortfall_3h'])

# Fit model
dtr = DecisionTreeRegressor(random_state=42)
print ("Training Model...")
dtr.fit(X_train, y_train)

# Pickle model for use within our API
save_path = './load-shortfall-regression-predict-api/assets/trained-models/load_shortfall_dt_regression.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(dtr, open(save_path,'wb'))
