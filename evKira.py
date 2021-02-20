import numpy as np
import pandas as pd
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn import linear_model
from sklearn.metrics.regression import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
data = read_csv("\\list.csv")

array = data.values
X1 = array[:,0:1] # Metre Kare
X2 = array[:,1:2] # Oda Sayısı
X3 = array[:,2:3] # Bina Yaşı
X4 = array[:,3:4] # Bulunduğu Kat
X5 = array[:,4:5] # Site İçinde
Y = array[:,5:6]  # Kira Bedeli

X1_train, X1_validation, Y1_train, Y1_validation = train_test_split(X1, Y, test_size=0.30)#trainkodları
X2_train, X2_validation, Y2_train, Y2_validation = train_test_split(X2, Y, test_size=0.30)#trainkodları
X3_train, X3_validation, Y3_train, Y3_validation = train_test_split(X3, Y, test_size=0.30)#trainkodları
X4_train, X4_validation, Y4_train, Y4_validation = train_test_split(X4, Y, test_size=0.30)#trainkodları
X5_train, X5_validation, Y5_train, Y5_validation = train_test_split(X5, Y, test_size=0.30)#trainkodları

print(data.info()) # Veri Setini İnceler
print(data.describe()) # Verinin İstatistiklerini Verir
print(data.isnull().sum()) # Sütunlardaki Boş Değerlerin Sayısını Verir
data.corr() # Korelasyon Değerlerini Verir

data.hist()#Histogram
#print(data.groupby("MetreKare").size())# Hangi veriden kaç tane olduğunu gösterir.

######################################################################
######################################################################
#############################scatter grafik###########################

#data.plot(x="MetreKare", y="Kira", kind='scatter', subplots=True, layout=(3,3), sharex=False, sharey=False)
#data.plot(x="Oda", y="Kira", kind='scatter', subplots=True, layout=(3,3), sharex=False, sharey=False)
#data.plot(x="BinaYasi", y="Kira", kind='scatter', subplots=True, layout=(3,3), sharex=False, sharey=False)
#data.plot(x="BulunduguKat", y="Kira", kind='scatter', subplots=True, layout=(3,3), sharex=False, sharey=False)
#data.plot(x="SiteIcinde", y="Kira", kind='scatter', subplots=True, layout=(3,3), sharex=False, sharey=False)
#pyplot.show()

######################################################################
### Metre Kare
cokluRegresyon = LinearRegression()
cokluRegresyon.fit(X1_train, Y1_train)

print(cokluRegresyon.coef_.round(2))
print(cokluRegresyon.intercept_.round(2))

Y1_predicted = cokluRegresyon.predict(X1_validation)
print(mean_squared_error(Y1_validation, Y1_predicted))
print(r2_score(Y1_validation, Y1_predicted))

print("\n\n")

pyplot.plot(Y1_validation, label = "GERÇEK")
pyplot.plot(Y1_predicted, label = "TAHMİN")
pyplot.legend()
pyplot.show()

######################################################################
### Oda Sayısı
cokluRegresyon = LinearRegression()
cokluRegresyon.fit(X2_train, Y2_train)

print(cokluRegresyon.coef_.round(2))
print(cokluRegresyon.intercept_.round(2))

Y2_predicted = cokluRegresyon.predict(X2_validation)
print(mean_squared_error(Y2_validation, Y2_predicted))
print(r2_score(Y2_validation, Y2_predicted))

print("\n\n")

pyplot.plot(Y2_validation, label = "GERÇEK")
pyplot.plot(Y2_predicted, label = "TAHMİN")
pyplot.legend()
pyplot.show()

######################################################################
### Bina Yaşı
cokluRegresyon = LinearRegression()
cokluRegresyon.fit(X3_train, Y3_train)

print(cokluRegresyon.coef_.round(2))
print(cokluRegresyon.intercept_.round(2))

Y3_predicted = cokluRegresyon.predict(X3_validation)
print(mean_squared_error(Y3_validation, Y3_predicted))
print(r2_score(Y3_validation, Y3_predicted))

print("\n\n")

pyplot.plot(Y3_validation, label = "GERÇEK")
pyplot.plot(Y3_predicted, label = "TAHMİN")
pyplot.legend()
pyplot.show()

######################################################################
### Bulunduğu Kat
cokluRegresyon = LinearRegression()
cokluRegresyon.fit(X4_train, Y4_train)

print(cokluRegresyon.coef_.round(2))
print(cokluRegresyon.intercept_.round(2))

Y4_predicted = cokluRegresyon.predict(X4_validation)
print(mean_squared_error(Y4_validation, Y4_predicted))
print(r2_score(Y4_validation, Y4_predicted))

print("\n\n")

pyplot.plot(Y4_validation, label = "GERÇEK")
pyplot.plot(Y4_predicted, label = "TAHMİN")
pyplot.legend()
pyplot.show()

######################################################################
### Site İçinde
cokluRegresyon = LinearRegression()
cokluRegresyon.fit(X5_train, Y5_train)

print(cokluRegresyon.coef_.round(2))
print(cokluRegresyon.intercept_.round(2))

Y5_predicted = cokluRegresyon.predict(X5_validation)
print(mean_squared_error(Y5_validation, Y5_predicted))
print(r2_score(Y5_validation, Y5_predicted))

print("\n\n")

pyplot.plot(Y5_validation, label = "GERÇEK")
pyplot.plot(Y5_predicted, label = "TAHMİN")
pyplot.legend()
pyplot.show()

######################################################################
######################################################################
######################################################################
###########################LeastSq Met.###############################
### Metre Kare
mean_X1 = np.mean(X1)
mean_Y = np.mean(Y)

n1 = len(X1)

numer = 0
denom = 0
for i in range(n1):
    numer += (X1[i] - mean_X1) * (Y[i] - mean_Y)
    denom += (X1[i] - mean_X1) ** 2
m = numer / denom
c = mean_Y - (m * mean_X1)

#Printing coefficients
print("MetreKare")
print(m, c)

######################################################################
### Oda Sayısı
mean_X2 = np.mean(X2)
mean_Y = np.mean(Y)

n2 = len(X2)

numer = 0
denom = 0
for i in range(n2):
    numer += (X2[i] - mean_X2) * (Y[i] - mean_Y)
    denom += (X2[i] - mean_X2) ** 2
m = numer / denom
c = mean_Y - (m * mean_X2)

#Printing coefficients
print("Oda Sayısı")
print(m, c)

######################################################################
### Bina Yaşı
mean_X3 = np.mean(X3)
mean_Y = np.mean(Y)

n3 = len(X3)

numer = 0
denom = 0
for i in range(n3):
    numer += (X3[i] - mean_X3) * (Y[i] - mean_Y)
    denom += (X3[i] - mean_X3) ** 2
m = numer / denom
c = mean_Y - (m * mean_X3)

#Printing coefficients
print("Bina Yaşı")
print(m, c)

######################################################################
### Bulunduğu Kat
mean_X4 = np.mean(X4)
mean_Y = np.mean(Y)

n4 = len(X4)

numer = 0
denom = 0
for i in range(n4):
    numer += (X4[i] - mean_X4) * (Y[i] - mean_Y)
    denom += (X4[i] - mean_X4) ** 2
m = numer / denom
c = mean_Y - (m * mean_X4)

#Printing coefficients
print("Bulunduğu Kat")
print(m, c)

######################################################################
### Metre Kare
mean_X5 = np.mean(X5)
mean_Y = np.mean(Y)

n5 = len(X5)

numer = 0
denom = 0
for i in range(n5):
    numer += (X5[i] - mean_X5) * (Y[i] - mean_Y)
    denom += (X5[i] - mean_X5) ** 2
m = numer / denom
c = mean_Y - (m * mean_X5)

#Printing coefficients
print("Site İçinde")
print(m, c)
