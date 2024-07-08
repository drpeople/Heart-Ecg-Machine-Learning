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


def import_file_signal(file_name):
    df = pd.read_csv(file_name)
    data = df[["'MLII'"]]
    #data = df[["'V1'"]]
    mlii = np.array(data)
    return mlii


def bandPassFilterTest(signal):
    fs= 360
    lowcut = 1
    highcut = 20
    nyq = 0.5 *fs
    low = lowcut / nyq
    high = highcut / nyq
    order = 1
    b,a = scipy.signal.butter(order,[low,high],'bandpass',analog=False)
    y = scipy.signal.lfilter(b, a, signal, axis=0)
    return (y)

# load dataset kai augmented
train_df = pd.read_csv("trainnewww.csv", header=None)
print(train_df[187].value_counts())
df2 = pd.read_csv("tess.csv", header=None)
from numpy import loadtxt
dtt = loadtxt('augmented_data.csv', delimiter=',')
dtt = pd.DataFrame(data=dtt)



# kanw ena dataset me ta arxika mou dedomena meion ta polla N
rows_N = train_df.loc[train_df.iloc[:,-1] == 0].sample(7000) ## kanw sample gia to N
rows_V = train_df.loc[train_df.iloc[:,-1] == 2] ## kanw sample gia to V
rows_Q = train_df.loc[train_df.iloc[:,-1] == 4] ## kanw sample gia to Q
rows_F = train_df.loc[train_df.iloc[:,-1] == 3].head(600) ## kanw sample gia to F
rows_S = train_df.loc[train_df.iloc[:,-1] == 1].head(1723) ## kanw sample gia to S
#Kanw sample apo ta augmented dedomena
augment_train_df=train_df.append(dtt)
augment_S = augment_train_df.loc[augment_train_df.iloc[:,-1] == 1].sample(8000)# 77% me 4000 80% me 10000
augment_F = augment_train_df.loc[augment_train_df.iloc[:,-1] == 3].sample(7000)# 86% me 8000 83% me 4500


# metaferw merika dedomena sto test gia na exw arketa gia kathe klash
rows_S_test = train_df.loc[train_df.iloc[:,-1] == 1].tail(500)
rows_F_test = train_df.loc[train_df.iloc[:,-1] == 3].tail(41)
df2 = df2.append(rows_S_test).append(rows_F_test)

# ftiaxnw to dataset pou tha xrhsimopoihsw gia train
# train_df=rows_N.append(rows_V)
# train_df=train_df.append(rows_Q)
# train_df=train_df.append(augment_S)
# train_df=train_df.append(augment_F)
# train_df=train_df.append(rows_F)
# train_df=train_df.append(rows_S)
tra=rows_N.append(rows_V)
train_aug= tra.append(rows_Q).append(augment_S).append(augment_F).append(rows_F).append(rows_S)
print("train dedomena",train_aug[187].value_counts())



#### plot analogia dedomenwn prin agument
label_names = {0 : 'N',
              1: 'S',
              2: 'V',
              3: 'F',
              4 : 'Q'}
# label=train_df[187].value_counts()
# label.rename(label_names).plot.bar()
# plt.show()


#### plot analogia dedomenwn meta agument
# label=train_aug[187].value_counts()
# label.rename(label_names).plot.bar()
# plt.show()

### plot test dedomena
# label=df2[187].value_counts()
# label.rename(label_names).plot.bar()
# plt.show()
print("Test dedomena",df2[187].value_counts())


# metatroph train dadaset se fft dwt dct (ta 2 apo ta 3 einai commented kai exw energo mono to dwt)
trainX = train_aug.iloc[0:, 0:186].values
trainx_table = []

for a in trainX:
    a = bandPassFilterTest(a)

    ####### dwt #####################
    c1 = pywt.wavedec(a, 'db4', level=4)
    coefficient = np.concatenate((c1[1],c1[2],c1[3],c1[4]))

    ########## fft #####################
    # coefficient = scipy.fftpack.rfft(a)
    # coefficient = coefficient[1:186]

    ######## dct ####################
    # coefficient = scipy.fftpack.dct(a)
    # coefficient = coefficient[1:186]
    trainx_table.append(coefficient)

trainY = train_aug.iloc[:, 187].values


# metatroph test dadaset se fft dwt dct (ta 2 apo ta 3 einai commented kai exw energo mono to dwt)
testt = df2
testX = testt.iloc[0:, 0:186].values
testx_table = []
for z in testX:
    z = bandPassFilterTest(z)


    ####### dwt #####################
    c1 = pywt.wavedec(z, 'db4', level=4)
    coefficient = np.concatenate((c1[1], c1[2], c1[3], c1[4]))


    ############### fft ###################
    # coefficient = scipy.fftpack.rfft(z)
    # coefficient = coefficient[1:186] # isws pernei kalutera apotelesmata an den kopso to 1

    ######## dct ####################
    # coefficient = scipy.fftpack.dct(z)
    # coefficient = coefficient[1:186]

    testx_table.append(coefficient)

testY = testt.iloc[:, 187].values


# ekpaideusi dedomenwn me Random Forest kai emnfanish apotelemsatwn
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

    names = ['N', 'S', 'V','F','Q']
    values = [sensitivityN*100,sensitivityS*100,sensitivityV*100,sensitivityF*100,sensitivityQ*100]
    plt.bar(names,values)
    plt.yticks(np.arange(0, 100, 5))
    plt.title('Random Forest 1000 Trees')
    plt.ylabel('Percentage',fontweight ='bold', fontsize = 12)
    plt.xlabel('Categories', fontweight='bold', fontsize=12)
    plt.show()

Random_Forest(trainx_table,trainY,testx_table,testY)


