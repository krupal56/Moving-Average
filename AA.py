import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
data = pd.read_csv('dow_jones_index.data')

mask = (data['stock'] == 'AA')
data = data.loc[mask]
data = data.ix[:,:7]
data['date'] = pd.to_datetime(data.date)
data.index = data['date']

#print data

data1 = pd.DataFrame(index=range(0,len(data)), columns=['date','close'])

for i in range(0,len(data)):
	data1['date'][i] = data['date'][i]
	data1['close'][i] = data['close'][i]

print data1

train = data1.loc[:12]
valid = data1.loc[12:]

## Moving Average

preds = [ ]
for i in range(0,13):
	a = train['close'][len(train)-12+i : ].sum() + sum(preds)
	b = a/12
	preds.append(b)

#print preds

valid['prediction'] = 0
valid['prediction'] = preds

rms = np.sqrt(np.mean(np.power((np.array(valid['close'])-preds),2)))
print rms

plt.plot(train['close'])
plt.plot(valid[['close','prediction']])
plt.show()





