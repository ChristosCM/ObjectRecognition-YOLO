import pandas as pd 
import numpy as np

data = pd.read_csv("new.csv")

print (data.minDis.value_counts())
data = data.replace(np.inf,0.0)

tm = data.time.sum(axis=0,skipna=True)
print ("Avg Time:{}".format(tm/100))

avgmin = data.minDis.sum(axis=0,skipna=True)
print ("Avg Min: {}".format(avgmin/100))

obj = data.objects.sum(axis=0,skipna=True)
print ("No of objects: {}".format(obj))