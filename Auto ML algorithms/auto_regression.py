from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor 
from sklearn.linear_model import LinearRegression , Ridge 
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np

def AutoEncoder(data):
  for column in data.columns:
    if data[column].dtype == 'O':
      encoder = LabelEncoder()
      data[column] = encoder.fit_transform(data[column])

def AutoSVR(dataPath,label):
  data = pd.read_csv(dataPath)
  kernals = ['linear', 'poly', 'rbf', 'sigmoid']
  AutoEncoder(data)
  data.head()
  if label in data.columns:
    label = data.pop(label)
    features = []
    for i in range(len(data.columns)):
      features.append(data.columns[i])

    features = data[list(features)]
    Xtrain , Xtest , Ytrain , Ytest = train_test_split(features,label,test_size=.3)
    bestAcc = []
    bestC = []
    Cs = np.arange(0.1,1.1,0.1)
    for kernal in kernals:
      best = None
      for C in Cs:
        model = SVR(kernel=kernal,C=C)
        model.fit(Xtrain,Ytrain)
        acc = model.score(Xtest,Ytest)
        if best == None: 
          best = acc
        elif best > pastAcc: 
          best = acc 
        elif best <= acc:
          bestAcc.append(pastAcc)
          bestC.append(pastC)
          break
        pastAcc = best
        pastC = C
    model = SVR(kernel=kernals[np.argmax(bestAcc)],C=bestC[np.argmax(bestAcc)])
    return model
  else:
    return None

def RandomForestRegressorAuto(dataPath,label):
  data = pd.read_csv(dataPath)
  AutoEncoder(data)
  if label in data.columns:
    Acc = []
    estimators = []
    label = data.pop(label)
    features = []
    for i in range(len(data.columns)):
      features.append(data.columns[i])

    features = data[list(features)]
    Xtrain , Xtest , Ytrain , Ytest = train_test_split(features,label,test_size=.3)
    for estimator in range(100,1000,50):
      model = RandomForestRegressor(n_estimators=estimator)
      model.fit(Xtrain,Ytrain)
      estimators.append(estimator)
      Acc.append(model.score(Xtest,Ytest))
    
    model = RandomForestRegressor(n_estimators=estimators[np.argmax(Acc)])
  else:
    return None

def LinearRegressionAuto(dataPath,label):
  data = pd.read_csv(dataPath)
  if label in data.columns:
    AutoEncoder(data)
    bools = [False,True]
    acc = []
    label = data.pop(label)
    features = []
    for feature in features:
      features.append(feature)
    features = data[list(features)]
    Xtrain , Xtest , Ytrain , Ytest = train_test_split(features,label,test_size=.3)
    for x in bools:
      model = LinearRegression(normalize=x)
      model.fit(Xtrain,Ytrain)
      acc.append(model.score(Xtest,Ytest))
    model = LinearRegression(normalize=bools[np.argmax(acc)])
    return model
  else:
    return None

def AutoRidge(dataPath,label):
  df = pd.read_csv(dataPath)
  if label in df.columns:
    AutoEncoder(df)
    alphas = np.arange(0.1,1.1,.1)
    acc = []
    label = df.pop(label)
    features = []
    for feature in df.columns:
      features.append(feature)
    features = df[list(features)]
    Xtrain , Xtest , Ytrain , Ytest = train_test_split(features,label,test_size=.3)
    for alpha in alphas:
      model = Ridge(alpha=alpha)
      model.fit(Xtrain,Ytrain)
      acc.append(model.score(Xtest,Ytest))
    model = Ridge(alpha=alphas[np.argmax(acc)])
    return model
  else:
    return None

def KNeighborsRegressorAuto(dataPath,label):
  data = pd.read_csv(dataPath)
  if label in data.column:
    AutoEncoder(data)
    label = data.pop(label)
    features = []
    for feature in features:
      features.append(feature)
    features = data[list(features)]
    neighbors = np.arange(1,21,1)
    acc = []
    Xtrain , Xtest , Ytrain , Ytest = train_test_split(features,label,test_size=.3)
    for neighbor in neighbors:
      model = KNeighborsRegressor(n_neighbors=neighbor)
      model.fit(Xtrain,Ytrain)
      acc.append(model.score(Xtest,Ytest))
    model = KNeighborsRegressor(n_neighbors=np.argmax(acc)+1)
    return model
  else:
    return None
