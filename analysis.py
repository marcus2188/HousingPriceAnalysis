# IMPORT ALL MODULES
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

# MAKE CREATE DATASET
dataobj = pd.read_csv("resale-flat-prices/resale-flat-prices-based-on-registration-date-from-jan-2017-onwards.csv")
housingDF = pd.DataFrame(dataobj)
housingDF.columns = [x.replace("_", " ") for x in housingDF.columns]
print(housingDF.head(5))

# MEASURE DATA SET
for x in housingDF.columns:
    print(housingDF[x].nunique())
print(housingDF.shape)

# IMPLEMENT MACHINE LEARNING MODEL
allvalues = dataobj.values
housingdata = allvalues[:, :4]
housingtarget = allvalues[:, -1].astype("float")
xtrain, xtest, ytrain, ytest = train_test_split(housingdata, housingtarget, random_state = 8)
encoderobj = OneHotEncoder(sparse = False)
xtrain = xtrain[:5000]
xtest = xtest[:5000]
ytrain = ytrain[:5000]
ytest = ytest[:5000]
xtrainencoded = encoderobj.fit_transform(xtrain).astype("float")
xtrainencoded.shape
rfrobj = RandomForestRegressor(n_estimators = 10, random_state = 8)
rfrobj.fit(xtrainencoded, ytrain)
print(rfrobj.score(xtrainencoded, ytrain))

# PREDICT VALUES



