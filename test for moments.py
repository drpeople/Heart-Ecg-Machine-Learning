import scipy
import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import pandas as pd
import plotly.express as px
from scipy.signal import butter, lfilter, freqz ,filtfilt
import pywt
from sklearn.preprocessing import  MinMaxScaler
from sklearn import preprocessing,random_projection
from scipy.signal import find_peaks,find_peaks_cwt,peak_widths
from scipy.fftpack import fft, dct,ifft,idct,rfft
np.set_printoptions(threshold=np.inf) #einai gia na borw na kanw print olo to array
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from csv import reader
import random
np.random.seed(42)
from scipy.signal import resample
from copy import deepcopy


# use of normal dataset
train_df = pd.read_csv("trainnewww.csv", header=None)
print(train_df[187].value_counts())
test_data = pd.read_csv("tess.csv", header=None)

from numpy import loadtxt
dtt = loadtxt('augmented_data.csv', delimiter=',')
dtt = pd.DataFrame(data=dtt)


#### dataset creation for augment
# train_data = pd.read_csv("trainnewww.csv", header=None)
# print(train_data[187].value_counts())
# test_data = pd.read_csv("tess.csv", header=None)
#
# from numpy import loadtxt
# dtt = loadtxt('augmented_data.csv', delimiter=',')
# dtt = pd.DataFrame(data=dtt)
#
# rows_N = train_data.loc[train_data.iloc[:,-1] == 0].sample(7000) ## kanw sample gia to N
# rows_S = train_data.loc[train_data.iloc[:,-1] == 1].head(1723)## kanw sample gia to S
# rows_V = train_data.loc[train_data.iloc[:,-1] == 2]## kanw sample gia to V
# rows_Q = train_data.loc[train_data.iloc[:,-1] == 4]## kanw sample gia to Q
# rows_F = train_data.loc[train_data.iloc[:,-1] == 3].head(600)## kanw sample gia to F
#
#
#
# augment_train_df=train_data.append(dtt)
#
# augment_S = augment_train_df.loc[augment_train_df.iloc[:,-1] == 1].sample(8000)# 77% me 4000 80% me 10000 4000 htan kanoniko
# augment_F = augment_train_df.loc[augment_train_df.iloc[:,-1] == 3].sample(7000)# 86% me 8000 83% me 4500
#
#
# train_df=rows_N.append(rows_V)
# train_df=train_df.append(rows_Q)
# train_df=train_df.append(augment_S)
# train_df=train_df.append(augment_F)
# train_df=train_df.append(rows_F)
# train_df=train_df.append(rows_S)
#
# ### #metaferw merika dedomena sto test gia na exw arketa gia kathe klash
# rows_S_test = train_data.loc[train_data.iloc[:,-1] == 1].tail(500)
# rows_F_test = train_data.loc[train_data.iloc[:,-1] == 3].tail(41)
# test_data = test_data.append(rows_S_test).append(rows_F_test)




print(train_df[187].value_counts())






def TMs2(img, order):
    F=img
    N = len(F)
    # print(N)
    Tn = Tchebichef_bar_poly(order, N)
    T = np.zeros([order])

    for n in range(order):
            xyz1 = Tn[n, :]
            xyz2 = F[0:N]
            T[n] = np.sum(xyz1 * xyz2)

    Moms = T
    return Moms, Tn


def Tchebichef_bar_poly(nmax, N):
    x = np.arange(N, dtype=np.float_)
    w = 2 * x - N + 1
    w1 = np.sqrt((N * N - 1) / 3)
    if nmax == 0:
        Tk = np.ones([1, len(x)]) / np.sqrt(N)
    elif nmax == 1:
        Tk = np.ones([2, len(x)])
        Tk[0, :] = Tk[0, :] * 0
        Tk[1, :] = (w / w1) * Tk[1, :] / np.sqrt(N)

    else:
        Tk = np.zeros([int(nmax), len(x)], dtype=np.float_)
        Tk[0, :] = np.ones([1, len(x)]) / np.sqrt(N)
        Tk[1, :] = (w / w1) * Tk[0, :]
        for m in range(2, nmax):
            ni = m
            w2_A = N ** 2 - ni ** 2
            w2_A = float(w2_A)
            w2_B = (2 * ni + 1) * (2 * ni - 1);
            w2_B = float(w2_B)

            w2 = ni * np.sqrt(w2_A / w2_B)

            Tk[m, :] = w / w2 * Tk[m - 1, :] - w1 / w2 * Tk[m -2, :]

            w1 =deepcopy(w2)
            T = deepcopy(Tk[m, :])
            for k in range(ni + 1):
                Tk[m, :] = Tk[m, :] - (np.sum(T * Tk[k, :])) * Tk[k, :]
            h = np.sqrt(np.sum(Tk[m, :] ** 2))

            Tk[m, :] = np.divide(Tk[m, :] ,h)
    return Tk


#### moments #####
trainn = train_df
trainX= trainn.iloc[0:, 0:185]


# trainx_table = []
train_table = []

for a in range(len(trainX)):

    train_label = trainn.iloc[a, 187]
    v= trainX.iloc[a, 0:185]
    coefficient,Tn =TMs2(v,186)

    coefficient = np.hstack((coefficient,train_label))
    train_table.append(coefficient)
    print("train",a)
    # trainx_table.append(coefficient)


print("train done")
# trainY = trainn.iloc[0:, 187]


testt = test_data
testX = testt.iloc[0:, 0:185]

# testx_table = []
test_table = []
for z in range(len(testX)):
    test_label = testt.iloc[z, 187]

    array = testt.iloc[z, 0:185]
    coefficient, Tn = TMs2(array, 186)

    coefficient = np.hstack((coefficient,test_label))
    print("test", z)
    test_table.append(coefficient)
    # testx_table.append(coefficient)

# testY = testt.loc[0:, 187]
print("test done")

######### end moments ##########



### save dataset for normal

with open("moments_train_normal_data.csv","w", newline='') as mom_csv:
    csvWriter = csv.writer(mom_csv,delimiter=",")
    # csvWriter.writerows(trainx_table)
    csvWriter.writerows(train_table)

with open("moments_test_normal_data.csv","w", newline='') as testmom_csv:
    csvWriter = csv.writer(testmom_csv,delimiter=",")
    # csvWriter.writerows(testx_table)
    csvWriter.writerows(test_table)




### save dataset for augmented (edw dimiourgw ta epauksimena dedomena basi tou arxikou dataset kai ta apothikeuw se ena neo arxeio)####

# with open("moments_train.csv","w", newline='') as mom_csv:
#     csvWriter = csv.writer(mom_csv,delimiter=",")
#     # csvWriter.writerows(trainx_table)
#     csvWriter.writerows(train_table)
#
# with open("moments_test.csv","w", newline='') as testmom_csv:
#     csvWriter = csv.writer(testmom_csv,delimiter=",")
#     # csvWriter.writerows(testx_table)
#     csvWriter.writerows(test_table)

print("writing done")
# times xwris fft dct
# trainx_table = train_df.iloc[0:, 0:186].values
# testx_table = test_data.iloc[0:, 0:186].values
# trainY = train_df.iloc[0:, 187].values
# testY = test_data.iloc[0:, 187].values

######### load moments ###
# trainx_table= pd.read_csv("moments_train.csv", header=None)
# trainY = train_df.iloc[:, 187].values
# testx_table = pd.read_csv("moments_test.csv", header=None)
# testY = test_data.iloc[0:, 187].values


# load moments augmented ####
# train = pd.read_csv("moments_train.csv", header=None)
# trainx = train.iloc[:, 0:185].values
# trainY = train.iloc[:, 186].values
# test = pd.read_csv("moments_test.csv", header=None)
# testx = test.iloc[:, 0:185].values
# testY = test.iloc[:, 186].values


### load normal data

train = pd.read_csv("moments_train_normal_data.csv", header=None)
trainx = train.iloc[:, 0:185].values
trainY = train.iloc[:, 186].values
test = pd.read_csv("moments_test_normal_data.csv", header=None)
testx = test.iloc[:, 0:185].values
testY = test.iloc[:, 186].values

def Random_Forest(train_data,trainy,test_data,testy):


    train = train_data
    X = train


    train2 = trainy
    y = train2


    test = test_data
    X2 = test

    test2 = testy
    y2 = test2




    X_train = X
    y_train = y
    X_test = X2
    y_test = y2


    classifier = RandomForestClassifier(n_estimators=100,random_state=0) #max_samples=0.3 dokimazw me 10 alla kai 100-200 einai kala
    classifier.fit(X_train,y_train)
    classifier_pred = classifier.predict(X_test)


    from sklearn.metrics import multilabel_confusion_matrix,confusion_matrix
    CMN,CMS,CMV,CMF,CMQ = multilabel_confusion_matrix(y_test,classifier_pred)
    TN1 = CMN[0][0]
    FP1 = CMN[0][1]
    FN1 = CMN[1][0]
    TP1 = CMN[1][1]
    #print(CMF)
    TN2 = CMS[0][0]
    FP2 = CMS[0][1]
    FN2 = CMS[1][0]
    TP2 = CMS[1][1]
    #print(CMN)
    TN3 = CMV[0][0]
    FP3 = CMV[0][1]
    FN3 = CMV[1][0]
    TP3 = CMV[1][1]
    #print(CMS)
    TN4 = CMF[0][0]
    FP4 = CMF[0][1]
    FN4 = CMF[1][0]
    TP4 = CMF[1][1]
   # print(CMV)
    TN5 = CMQ[0][0]
    FP5 = CMQ[0][1]
    FN5 = CMQ[1][0]
    TP5 = CMQ[1][1]
    #print(CMQ)
    sensitivityN = TP1/(TP1+FN1)
    sensitivityS = TP2/(TP2+FN2)
    sensitivityV = TP3/(TP3+FN3)
    sensitivityF = TP4/(TP4+FN4)
    sensitivityQ = TP5/(TP5+FN5)
    print("sensitivity tou N:",sensitivityN*100," ","sensitivity tou S:",sensitivityS*100," ","sensitivity tou V:",sensitivityV*100," ","sensitivity tou F:",sensitivityF*100,"sensitivity tou Q:",sensitivityQ*100)


    acc = accuracy_score(y_test,classifier_pred)
    print("To oliko accuracy einai :",acc)



    classi = classification_report(y_test,classifier_pred)
    print(classi)


    CM = confusion_matrix(y_test, classifier_pred)
    print(CM)



# Random_Forest(trainx_table,trainY,testx_table,testY)
Random_Forest(trainx,trainY,testx,testY)