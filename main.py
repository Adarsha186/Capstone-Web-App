import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# from sklearnex.svm import SVC
from sklearn.svm import SVC

data1 = pd.read_csv("Admission_Predict.csv")
data1 = data1.rename(columns={'GRE Score': 'GRE', 'TOEFL Score': 'TOEFL', 'LOR ': 'LOR', 'Chance of Admit ': 'CoA'})

print("---------------------Description of the data sets---------------------\n")
print(data1.describe)
print(data1.isnull().sum())
plt.figure(figsize=(5, 5))
plt.bar(data1["Research"].apply(str), data1["CoA"])
plt.title("Impact of research activity on admission chances")
plt.xlabel('Research Work')
plt.ylabel('Admission chances')
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(x="CGPA", y="GRE", data=data1)
plt.title("Comparison between Cgpa and Gre Scores")
plt.xlabel('CGPA')
plt.ylabel('GRE')
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(x="CoA", y="GRE", data=data1)
plt.title("Comparison between Admit Chances and Gre Scores")
plt.xlabel('Admission Chances')
plt.ylabel('GRE')
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(x="CoA", y="TOEFL", data=data1)
plt.title("Comparison between Admit Chances and TOEFL Scores")
plt.xlabel('Admission Chances')
plt.ylabel('TOEFL')
plt.show()

print(data1.corr(method='kendall'))

print(data1.corr(method='pearson'))

corr_matrix = data1.corr()
print(corr_matrix["CGPA"])

print(corr_matrix["GRE"])

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

linear_regression = LinearRegression()
linear_regression = linear_regression.fit(x1_train, y1_train)
model = LinearRegression(normalize=True)
model.fit(x1_test, y1_test)
linearScore = model.score(x1_test, y1_test)
# print("Accuracy of linear regression model : ",linearScore*100,"%")

SVM_classifier = SVC(kernel='linear')
SVM_classifier.fit(x1_train, y1_train)
SVM_Score = SVM_classifier.score(x1_test, y1_test)
# print("Accuracy of SVM model : ",SVM_Score*100,"%")

knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(x1_train, y1_train)
print(SVM_classifier.predict([[331, 111, 3, 3, 8, 1]]))

random_forest = RandomForestRegressor(max_depth=1, n_estimators=100, random_state=1)
random_forest.fit(x1_train, y1_train)
root_mean_squared_error = np.sqrt(mean_squared_error(y1_train, random_forest.predict(x1_train)))
random_forest_score = random_forest.score(x1_test, y1_test)
print("Accuracy of Random Forest model : ",random_forest_score*100,"%")
print("RMSE of random forest model : ",root_mean_squared_error)

root_mean_squared_error_knn = np.sqrt(mean_squared_error(y1_train, knn.predict(x1_train)))
knn_score = knn.score(x1_test, y1_test)
print("Accuracy of KNN model : ",knn_score*100,"%")
print("RMSE value of KNN : ",root_mean_squared_error_knn)

root_mean_squared_error_svm = np.sqrt(mean_squared_error(y1_train, SVM_classifier.predict(x1_train)))
SVM_Score = SVM_classifier.score(x1_test, y1_test)
print("Accuracy of SVM model : ",SVM_Score*100,"%")
print("RMSE of SVM model : ",root_mean_squared_error_svm)

pickle.dump(SVM_classifier, open('GradSchool.pkl', 'wb'))

print("\nPrediction Model dumped successfully")
