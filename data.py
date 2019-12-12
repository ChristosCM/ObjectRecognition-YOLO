import pandas as pd 
import numpy as np

data = pd.read_csv("150Smooth1.csv")


dis = data['minDis'].replace({"inf":0.0})
print (dis)
dis = dis.replace(0.0,np.nan)
print (dis.value_counts())

dis = dis.sum(axis=0,skipna=True)
print (dis/100)