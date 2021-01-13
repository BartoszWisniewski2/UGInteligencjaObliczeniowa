import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

np.random.seed(7)

dataForPlot = []
df = pd.read_csv('breast-cancer-wisconsin.csv')

# Get names of indexes for which column value = '?'
indexNames = df[df['Bare-Nuclei'] == '?'].index
# Delete these row indexes from dataFrame
df.drop(indexNames, inplace=True)

# a) Podziel w losowy sposób bazę danych na zbiór treningowy (60%) i testowy (40%).

all_inputs = df[['Clump-Thickness',	'Uniformity-of-Cell-Size', 'Uniformity-of-Cell-Shape', 'Marginal-Adhesion',	'Single-Epithelial-Cell-Size',	'Bare-Nuclei',	'Bland-Chromatin',	'Normal-Nucleoli',	'Mitoses']].values
all_classes = df['Class'].values


(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.6,
                                                                            random_state=1)
for i in range(len(train_inputs)):
    print(train_inputs[i], train_classes[i])
for i in range(len(test_inputs)):
    print(test_inputs[i], test_classes[i])

# b) Uruchom każdy z klasyfikatorów wykorzystując paczki i dokonaj ewaluacji ma zbiorze
# testowym wyświetlając procentową dokładność i macierz błędu.

# GAUSSIAN
gnb = GaussianNB()
gnb = gnb.fit(train_inputs, train_classes)
print("Poprawność Gaussian: %f %%" % (gnb.score(test_inputs, test_classes) * 100))
dataForPlot.append((gnb.score(test_inputs, test_classes) * 100))

matrix1 = plot_confusion_matrix(gnb, test_inputs, test_classes)
plt.title('Macierz błędu Gaussian')

# DRZEWA
dtc = DecisionTreeClassifier()
dtc = dtc.fit(train_inputs, train_classes)
print("Poprawność Drzewa: %f %%" % (dtc.score(test_inputs, test_classes) * 100))
dataForPlot.append((dtc.score(test_inputs, test_classes) * 100))

matrix2 = plot_confusion_matrix(dtc, test_inputs, test_classes)
plt.title('Macierz błędu Drzewa')

# k-NN, k=3
knn3 = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn3.fit(train_inputs, train_classes)
print("Poprawność k-NN, k= 3 : %f %%" % (knn3.score(test_inputs, test_classes) * 100))
dataForPlot.append((knn3.score(test_inputs, test_classes) * 100))
matrix3 = plot_confusion_matrix(knn3, test_inputs, test_classes)
plt.title('Macierz błędu k-NN, k=3')

# k-NN, k=5
knn5 = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn5.fit(train_inputs, train_classes)
print("Poprawność k-NN, k= 5 : %f %%" % (knn5.score(test_inputs, test_classes) * 100))
matrix5 = plot_confusion_matrix(knn5, test_inputs, test_classes)
dataForPlot.append((knn5.score(test_inputs, test_classes) * 100))
plt.title('Macierz błędu k-NN, k=5')

# k-NN, k=11
knn11 = KNeighborsClassifier(n_neighbors=11, metric='euclidean')
knn11.fit(train_inputs, train_classes)
print("Poprawność k-NN, k= 11 : %f %%" % (knn11.score(test_inputs, test_classes) * 100))
matrix11 = plot_confusion_matrix(knn11, test_inputs, test_classes)
dataForPlot.append((knn11.score(test_inputs, test_classes) * 100))
plt.title('Macierz błędu k-NN, k=11')

#Neural network

# a) scaling the data

scaler = StandardScaler()

# we fit the train data
scaler.fit(train_inputs)


# scaling the train data
train_data = scaler.transform(train_inputs)
test_data = scaler.transform(test_inputs)

# Training the Model
# creating an classifier from the model:
mlp = MLPClassifier(hidden_layer_sizes=(8, 8, 8), max_iter=500)

# let's fit the training data to our model
mlp.fit(train_data, train_classes)

predictions_train = mlp.predict(train_data)
print(accuracy_score(predictions_train, train_classes))
predictions_test = mlp.predict(test_data)
print(accuracy_score(predictions_test, test_classes))


print(confusion_matrix(predictions_train, train_classes))
print(confusion_matrix(predictions_test, test_classes))
print(classification_report(predictions_test, test_classes))

matrixxd = plot_confusion_matrix(mlp, test_data, test_classes)
dataForPlot.append((mlp.score(test_data, test_classes) * 100))
plt.title('Macierz błędu sieci neuronowych')

plt.show()

bars = ('Gaussian', 'Drzewa', 'k-NN, k=3', 'k-NN, k=5', 'k-NN, k=11', 'Sieci neuronowe')
plt.barh(bars, dataForPlot)

for index, value in enumerate(dataForPlot):
    plt.text(30, index, str(value))
plt.title('Poprawność przewidzianych wyników')

plt.show()

