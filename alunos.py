import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
from sklearn import linear_model
from sklearn.utils import shuffle
data = pd.read_csv("student-mat.csv", sep=";") #DataFrame Pandas

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]  #Limpando a data para atributos uteils
predict = "G3"

X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)


def train_best_model(times):
    best = 0
    for _ in range(times):

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

        model = linear_model.LinearRegression()

        model.fit(x_train, y_train)

        acc = model.score(x_test, y_test)
        print(acc)
        if acc > best:
            best = acc
            with open("studednt_model.pickle", "wb") as f:
                pickle.dump(model, f)
# train_best_model(50)

pickle_in = open("studednt_model.pickle", "rb")
model = pickle.load(pickle_in)

print("Cos:\n", model.coef_)
print("Intercept:\n", model.intercept_)

preds = model.predict(x_test)

for i in range(len(preds)):
    print(preds[i], x_test[i], y_test[i])

p = 'G1'
style.use("ggplot")
plt.scatter(data[p], data['G3'])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()
