# train classifier

import pickle 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

category = input("Enter category to train: ")

data_file = './pickle_jar/' + category + '.pickle'
data_dict = pickle.load(open(data_file, 'rb'))

X = []
y = []

for key, values in data_dict.items():
    for value in values:
        y.append(key)
        X.append(value)

data = np.asarray(X)
labels = np.asarray(y)


x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle = True, stratify=labels)

model = RandomForestClassifier()
model.fit(x_train, y_train)
y_predict=model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score*100))

model_name = 'pickle_jar/' + category + '_model.p'

f = open(model_name, 'wb')
pickle.dump({'model': model}, f)
f.close()




