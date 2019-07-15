"""
Created on Wed Feb 20 02:00:31 2019

@author: tanbin 
"""
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

names = ['id', 'clump_thickness', 'uniformity_of_cell_size', 'uniformity_of_cell_shape', 'marginal_adhesion', 'single_epithial_cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses', 'benign_malignant', '', '', '']
# names =  ['id', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', '', '', '']
dataset = pd.read_csv("Breast-Cancer-Preprocessed.csv", header = None, names = names)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




################################


class_names = [ 'B', 'M']

# Import dataset
import pandas as pd
names = ['id', 'clump_thickness', 'uniformity_of_cell_size', 'uniformity_of_cell_shape', 'marginal_adhesion', 'single_epithial_cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses', 'benign_malignant', '', '', '']
# names =  ['id', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', '', '', '']
dataset = pd.read_csv("Breast-Cancer-Preprocessed.csv", header = None, names = names)

# Split test and train data
import numpy as np
from sklearn.model_selection import train_test_split
#X = np.array(dataset.ix[:, 1:10])
#y = np.array(dataset['benign_malignant'])
X = dataset.iloc[0:700:, 1:10].values
y = dataset.iloc[0:700, 10].values
#####################################

# shape of data is 150
cv = model_selection.KFold(n_splits=4,shuffle=False,random_state=None)
# print(cv)
for train_index, test_index in cv.split(X,y):

    X_tr, X_tes = X[train_index], X[test_index]
    y_tr, y_tes = y[train_index],y[test_index]
    #clf = svm.SVC(kernel='rbf', C=1).fit(X_tr, y_tr) 
    clf = RandomForestClassifier(n_estimators=50,n_jobs=-1)
    classifier = AdaBoostClassifier(base_estimator=clf,n_estimators=clf.n_estimators)
    clf = classifier.fit(X_tr, y_tr)
    y_pred=clf.predict(X_tr)
    y_pred1 = clf.predict(X_tes)
    cnf_matrix = confusion_matrix(y_tr, y_pred)
    cnf_matrix1 = confusion_matrix(y_tes, y_pred1)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Training Phase')
    plt.rc('font', size=18) 
    
    plt.figure()
    plot_confusion_matrix(cnf_matrix1, classes=class_names,
                      title='Testing Phase')
    plt.rc('font', size=18) 
    # Plot normalized confusion matrix
    #plt.figure()
    #plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
    #                  title='Normalized confusion matrix')

    plt.show()
    
clfr = RandomForestClassifier(n_estimators=50,n_jobs=-1)
classifier = AdaBoostClassifier(base_estimator=clf,n_estimators=clf.n_estimators)
clf = classifier.fit(X_tr, y_tr)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = clf, X = X, y = y, cv = 4)
print(accuracies.mean())
