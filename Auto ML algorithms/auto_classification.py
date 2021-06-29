from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB , BernoulliNB , MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def AutoEncoder(data):
  for column in data.columns:
    if data[column].dtype == 'O':
      encoder = LabelEncoder()
      data[column] = encoder.fit_transform(data[column])

def AutoSVC(dataPath,label):
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
        model = SVC(kernel=kernal,C=C)
        model.fit(Xtrain,Ytrain)
        acc = model.score(Xtest,Ytest)
        if best == None: 
          best = acc
        elif best > pastAcc: 
          best = acc 
        elif best < acc:
          bestAcc.append(pastAcc)
          bestC.append(pastC)
          break
        pastAcc = best
        pastC = C
    model = SVC(kernel=kernals[np.argmax(bestAcc)],C=bestC[np.argmax(bestAcc)])
    return model
  else:
    return None

def RandomForestCauto(dataPath,label):
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
      model = RandomForestClassifier(n_estimators=estimator)
      model.fit(Xtrain,Ytrain)
      estimators.append(estimator)
      Acc.append(model.score(Xtest,Ytest))
    
    model = RandomForestClassifier(n_estimators=estimators[np.argmax(Acc)])
  else:
    return None

def NBauto(dataPath,label):
  data = pd.read_csv(dataPath)
  AutoEncoder(data)
  if label in data.columns:
    models = [GaussianNB() , BernoulliNB() , MultinomialNB()]
    Acc =[]
    label = data.pop(label)
    features = []
    for i in range(len(data.columns)):
      features.append(data.columns[i])

    features = data[list(features)]
    Xtrain , Xtest , Ytrain , Ytest = train_test_split(features,label,test_size=.3)
    for model in models:
      model.fit(Xtrain,Ytrain)
      Acc.append(model.score(Xtest,Ytest))

    return models[np.argmax(Acc)]
  return None

def AutoKNeighborsClassifier(dataPath,label):
  data = pd.read_csv(dataPath)
  AutoEncoder(data)
  if label in data.columns:
    neighbors = np.arange(1,21)
    Acc =[]
    label = data.pop(label)
    features = []
    for i in range(len(data.columns)):
      features.append(data.columns[i])

    features = data[list(features)]
    Xtrain , Xtest , Ytrain , Ytest = train_test_split(features,label,test_size=.3)
    for neighbor in neighbors:
      model = KNeighborsClassifier(n_neighbors=neighbor)
      model.fit(Xtrain,Ytrain)
      Acc.append(model.score(Xtest,Ytest))
     
    model = KNeighborsClassifier(n_neighbors=np.argmax(Acc)+1)
    return model
  return None

def AutoLogisticRegression(dataPath,label):
  data = pd.read_csv(dataPath)
  AutoEncoder(data)

  if label in data.columns:
    L1 = ['saga','liblinear']
    L2 = ['newton-cg','lbfgs','sag','saga']
    L1_acc = []
    L2_acc = []
    label = data.pop('gender')
    features = []
    for i in range(len(data.columns)):
      features.append(data.columns[i])

    features = data[list(features)]
    Xtrain , Xtest , Ytrain , Ytest = train_test_split(features,label,test_size=.3)

    for solver in L1:
      model = LogisticRegression(penalty='l1',solver=solver)
      model.fit(Xtrain,Ytrain)
      L1_acc.append(model.score(Xtest,Ytest))

    for solver in L2: 
      model = LogisticRegression(penalty='l2',solver=solver)
      model.fit(Xtrain,Ytrain)
      L2_acc.append(model.score(Xtest,Ytest))

      if L1[np.argmax(L1_acc)] >= L2[np.argmax(L2_acc)]:
        model = LogisticRegression(penalty='l1',solver=L1[np.argmax(L1_acc)])
      else:
        model = LogisticRegression(penalty='l2',solver=L2[np.argmax(L2_acc)])

      return model
  else:
    return None