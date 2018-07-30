import pandas as pd
import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df= quandl.get('WIKI/GOOGL', authtoken='UuZzKaKtdzCR6nKVshYg')
df= df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT']=(df['Adj. High'] - df['Adj. Low']) / df['Adj. Low']*100.0
df['PCT_change']=(df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']*100.0

df=df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]


forecast_col = 'Adj. Close'
df.fillna(-9999, inplace=True)
forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)
df['label']=df[forecast_col].shift(-forecast_out)


X= np.array(df.drop(['label'],1))

X= preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out:]
df.dropna(inplace=True)
Y= np.array(df['label'])
#X= X[:-forecast_out+1]
#df.dropna(inplace=True)
Y=np.array(df['label'])
#print(len(X), len(Y))

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)

clf = LinearRegression() #Using Regression Algorithm
#clf = svm.SVR() #Using Support Vector Machine
#clf = svm.SVR(kernel='poly') #Using Support Vector Machine and Kernels Defined in ML
#In this paricular program we will use Regression because it gives better resuts or is more accurate
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)
#print(accuracy)
forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan
#To show the date along with the predictions

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
print(df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
                         
