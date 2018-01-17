import pandas as pd
import numpy as np

data = pd.read_csv('newsub.csv')
data['Survived'][data['Survived']>0.5]=True
data['Survived'][data['Survived']<0.5]=False
data['Survived'] = data['Survived'].astype(int)
print(data)
data.to_csv('sub.csv',index=False)
