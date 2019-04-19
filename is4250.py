import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv("diabetes_dataset.csv")

"""
This script processes the raw data and outputs the logistic regression model in a Pickle object.
"""

# FEATURES = ['_STATE','_RFHYPE5','_RFCHOL1','_MICHD','_ASTHMS1',
# '_DRDXAR1','_AGE_G','HTM4','WTKG3','_BMI5CAT','_RFBMI5','_EDUCAG',
# '_INCOMG','_RFSMOK3','_CURECIG','_RFDRHV5','_FRTLT1A','_VEGLT1A','_PAINDX1']

FEATURES = ['_AGE_G','HTM4','WTKG3','_BMI5CAT']
TARGET = ['DIABETE3']

# Prepare the dataframe.
df = df.dropna(subset=FEATURES)
df = df[df['DIABETE3'].isin([1.0,2.0,3.0,4.0])]
df['DIABETE3'] = df['DIABETE3'].replace(2.0,1.0)
df['DIABETE3'] = df['DIABETE3'].replace(4.0,1.0)
for col in FEATURES:
    df.drop(df.index[df[col] == 9], inplace = True)
df['DIABETE3'] = df['DIABETE3'].replace(1.0,1)
df['DIABETE3'] = df['DIABETE3'].replace(3.0,0)

# Prepare for training
X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4250)
print("Training...")

# Logistic regression
lr = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train, y_train)
y_pred = lr.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Save the model
filename = 'diabetes_model.p'
pickle.dump(lr, open(filename, 'wb'))


