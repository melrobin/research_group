import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
fname='bezdekIris.data'
features=[]
labels=[]
with open(fname) as f:
    for line in f:
        data=line.split(',')
        features.append(data[:-1])
        labels.append(data[-1].strip('\n'))
features=np.array(features,dtype=float)

le = preprocessing.LabelEncoder()
le.fit(labels)
classes=list(le.classes_)
numerical_labels=le.transform(labels)

clf=svm.SVC()
clf.fit(features,numerical_labels)
predicted_values=clf.predict(features)
matrix = confusion_matrix(numerical_labels, predicted_values)
print(matrix)

forest = RandomForestClassifier(n_estimators = 3)
forest.fit(features,numerical_labels)
predicted_values=forest.predict(features)
matrix = confusion_matrix(numerical_labels, predicted_values)
print(matrix)
#array([0, 0, 1, 2]...)
#>>> le.inverse_transform([0, 0, 1, 2])
#array([1, 1, 2, 6])
