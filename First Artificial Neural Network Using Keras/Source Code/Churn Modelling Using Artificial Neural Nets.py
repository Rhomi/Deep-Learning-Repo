# coding: utf-8

# Lets Do a Churn Modelling Using an Artificial Neural Network

# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
# Including all columns from 3 through 12 (Remember that the upper bound should be 13 to add 12th row)
# We believe all these features have an impact on customer churn.
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Check the dimensions of the X and y arrays.
print(X.shape)
print(y.shape)


# Now that we have the dataset, we need to make sure that the data is in the best shape  for us to apply a Neural Network. 
# Turns out our data has Categorical variables  in the form of Strings as we could see from the xls file. 
# Hence, they need to be encoded before inputting them into a Neural Net.
# 
# Our dependent variable (churn) is also categorical, but its binary and takes only 1s and 0s. 
# So we don't need to encode it into numbers cos its already in numerical form. 
# So, right now, we need only to encode our dependent variables that are strings and are categorical variables.
# 
# We'll use the LabelEncoder and OneHotEncoder from Python Scikit library.

# We'll define a function to take care of encoding for us. Before that, we'll import the packages.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def categorical_encoder(data, index):
    label_encoder = LabelEncoder()
    data[:, index] = label_encoder.fit_transform(data[:, index])
    return(data)


# The function takes in the numpy array to be processed and the index of the field to be encoded.

# Lets try that
# The index of the field Country is 1
X = categorical_encoder(X,1)


# Index is 2 for gender
X = categorical_encoder(X, 2)

# Looks like the countries have been encoded with France as 0, Germany as 1 and Spain as 2. 
# Wait a sec. They are not ordinal and hence there is no relational ordering between them. 
# Even though the numbers assigned to these countries are purely random in nature, they are in no way
# better or worse than the others as the numbers suggest. We need to fix this.
# 
# We could create Dummy Variables to fix this issue.

# Lets create another function to do this.
# Excuse me for the bad choice of names for the function. But its descriptive!! 

# Once again, the numpy array and the index of the field to be encoded are the inputs to this function.
def dummy_variable_maker(data, index):
    onehotencoder = OneHotEncoder(categorical_features = [index])
    data = onehotencoder.fit_transform(data).toarray()
    return(data)

X = dummy_variable_maker(X,1)

# As we could see, we have 11 features instead of 9 before. 
# That's because we have created dummy variables for the country variable and since there are 
# 3 countries, 2 additional fields have been added to accomodate them. 
# Rows countaining France will say 1.0 on those corresponding rows and the same logic applies 
# to the other countries as well.
# 
# But before we go ahead with modelling. Lets think again. 
# We don't need 3 dummy variables for 3 countries. We need only 2 as the 3rd one will be 
# represented by the values that aren't the other 2 dummy variables.
# 
# So, let's remove one of the dummy variable field and avoid falling into the dummy variable trap.

# Remove field 1
# We'll take all columns except the first one.
X = X[:, 1:]


# ## Hurray!! Now we are ready to split the data into train and test sets.

# Note that cross_validation has been replaced by model_selection in the latest version
from sklearn.model_selection import train_test_split
def split_date(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    return(X_train, X_test, y_train, y_test)

X_train, X_test, y_train, y_test = split_date(X,y)

# ### Lets Apply Feature Scaling.
# 
# Feature scaling is absolutely necessarily as there is a lot of computations -- 
# highly compute intensive calculations and a lot of parallel computing. 
# Feature scaling eases up all these calculations. 
# We don't need one independent variable dominating another.

from sklearn.preprocessing import StandardScaler
def feature_scaler(data):
    sc = StandardScaler()
    return(sc.fit_transform(data))


X_train = feature_scaler(X_train)
X_test = feature_scaler(X_test)

# ## And that finishes the Preprocessing Stage Completely!!