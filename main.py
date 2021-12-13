import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

data1 = pd.read_csv("Admission_Predict.csv")
data1 = data1.rename(columns={'GRE Score': 'GRE', 'TOEFL Score': 'TOEFL', 'LOR ': 'LOR', 'Chance of Admit ': 'CoA'})

x = 0
j = 0
for i in data1["CoA"]:
    if 0.8 <= i <= 0.99:
        j = 1
    elif 0.65 <= i <= 0.79:
        j = 2
    elif 0.5 <= i <= 0.64:
        j = 3
    elif 0.5 <= i <= 0.64:
        j = 4
    elif i <= 0.3:
        j = 5
    data1['CoA'][x] = j
    x = x + 1

data1['CoA'] = data1['CoA'].astype(int)
data1['CoA'] = data1['CoA'].astype(str)

y1 = data1["CoA"]
x1 = data1.drop("CoA", axis="columns")
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(x1_train, y1_train)
print(knn.predict([[331, 111, 3, 3, 8, 1]]))
pickle.dump(knn, open('GradSchool.pkl', 'wb'))
