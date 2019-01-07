# Data Preprocessing

# Imoporting Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing the dataset
dataset = pd.read_csv('Data.csv')

#Independent variables from dataset 
independentVariables = dataset.iloc[:, :-1].values

#Dependent variables from dataset 
dependentVariables = dataset.iloc[:, 3].values 


#Handling missing data
"""from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy='mean',axis=0)

imputer = imputer.fit(independentVariables[:, 1:3])

independentVariables[:, 1:3] = imputer.transform(independentVariables[:, 1:3])
"""
#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_independentVariables = LabelEncoder()
independentVariables[:, 0] = labelencoder_independentVariables.fit_transform(independentVariables[:, 0])
oneHotEncoder = OneHotEncoder(categorical_features = [0])
independentVariables = oneHotEncoder.fit_transform(independentVariables).toarray()
labelencoder_dependentVariables = LabelEncoder()
dependentVariables = labelencoder_dependentVariables.fit_transform(dependentVariables)


#Splittingthe dataset into Training set and Test set
from sklearn.model_selection import train_test_split
independentVariables_train,independentVariables_test,dependentVariables_train,dependentVariables_test= train_test_split(independentVariables,dependentVariables,test_size= 0.2, random_state = 0)


#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_independentVariables = StandardScaler()
independentVariables_train = sc_independentVariables.fit_transform(independentVariables_train)
independentVariables_test = sc_independentVariables.transform(independentVariables_test)
"""