
# coding: utf-8

# In[ ]:

#CSC570R - Advanced Validation Assignment-Chris DiBello
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.grid_search import GridSearchCV
from sklearn import linear_model
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score
from __future__ import print_function



from sklearn import datasets

from sklearn.metrics import classification_report
from sklearn.svm import SVC

from sklearn import datasets, svm

from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
#Load breast Cancer Model
df = pd.read_csv("breast_cancer.csv")
X = pd.DataFrame()
X['id_number'] = df['id_number']
X['clump_thickness'] = df['clump_thickness']
X['uniformity_of_cell_size'] = df['uniformity_of_cell_size']

X['uniformity_of_cell_shape'] = df['uniformity_of_cell_shape']
X['marginal_adhesion'] = df['marginal_adhesion']
X['epithelial_cell_size'] = df['epithelial_cell_size']
X['bland_chromatin'] = df['bland_chromatin']
X['normal_nucleoli'] = df['normal_nucleoli']
X['mitoses'] = df['mitoses']
X['malignant'] = df['malignant']
y = X.pop("malignant")
get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u'pylab inline')
#Describe data and look for missing values
print ("    ")
print ("    ")
print ("Describe Data")
print (X.describe())
print ("    ")
print ("    ")
print ("    List of Missing Variables")
print(X.apply(lambda x: sum(x.isnull()),axis=0))
print ("    ")
print ("    ")


print(__doc__)
#use train_test_split to split data
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
#np.random.seed(100)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
n_estimators = [300,400,500]
max_features = ['auto', 'sqrt','log2']
min_samples_split = [3,5,7]

#Use GridSearch with Random Forest Classifier
rfc = RandomForestClassifier(n_jobs=1)
#Parameters of pipelines can be set using ‘__’ separated parameter names:
estimator = GridSearchCV(rfc,
                         dict(n_estimators=n_estimators,
                              max_features=max_features,
                              min_samples_split=min_samples_split
                              ), cv=None, n_jobs=-1)

estimator.fit(X_train, y_train)
estimator.best_estimator_
best_rfc = estimator.best_estimator_
from sklearn import cross_validation
scores = cross_validation.cross_val_score(best_rfc, X, y, cv=10)
for score in scores:
    #Display gridsearch scores using precision and recall
    print("# Tuning hyper-parameters for %s" % score)
    print()

    print("Best parameters set found on development set:")
    print()
    print(estimator.best_params_)
    print()
    

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, estimator.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
#Show Base and AUC scores
def base_rate_model(X):
    y = np.zeros(X.shape[0])
    return y
#Show acuracy and AUC score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Z=X
X= scaler.fit_transform(X)
#build test and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_base_rate = base_rate_model(X_test)
from sklearn.metrics import accuracy_score
print ("Base rate accuracy is %2.2f" % accuracy_score(y_test, y_base_rate))
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty='l2', C=1)
model.fit(X_train, y_train)
print ("Logistic accuracy is %2.2f" % accuracy_score(y_test,model.predict(X_test)))
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
print ("---Base Model---")
#base rate AUC
base_roc_auc = roc_auc_score(y_test, base_rate_model(X_test))
print ("Base Rate AUC = %2.2f" % base_roc_auc)
print (classification_report(y_test,base_rate_model(X_test) ))
print ("\n\n---Logistic Model---")
#logistic AUC
logit_roc_auc = roc_auc_score(y_test, model.predict(X_test))
print ("Logistic AUC = %2.2f" % logit_roc_auc)
print (classification_report(y_test, model.predict(X_test) ))
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 2], [0, 2], 'k--')
plt.xlim([0.0, 2.0])
plt.ylim([0.0, 2.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
#Compute and display K-Fold values
print ('K-Fold Score')
mean_score = scores.mean()
std_dev = scores.std()
std_error = scores.std() / math.sqrt(scores.shape[0])
ci =  2.262 * std_error
lower_bound = mean_score - ci
upper_bound = mean_score + ci

print ("Score is %f +/-  %f" % (mean_score, ci))
print ("95 percent probability that if this experiment were")
print ("repeated over and over the average score would be between %f and %f" % (lower_bound, upper_bound))


get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u'pylab inline')

print ("    ")
print ("    ")
#Calculate Linear Regression 
np.random.seed(100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

model = LinearRegression()
model.fit(X_train, y_train)
print ("Linear Regression:", model.score(X_test, y_test).round(2))
print("    ")
print ("    ")
#Calulate DecisionTreRegressor
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
print ("Decision Tree:", model.score(X_test, y_test).round(2))

print ("    ")
print ("    ")
#Calculate Random Forrest Regressor Score
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
print ("Random Forest:", model.score(X_test, y_test).round(2))
print ("    ")
print ("    ")
print ("Visualize Random Forest:")

model.feature_importances_
# Simple version that shows all of the variables
feature_importances = pd.Series(model.feature_importances_, index=Z.columns)
feature_importances.sort()
feature_importances.plot(kind="barh", figsize=(7,6));
print ("    ")
print ("    ")
#
print ("    ")
print ("    ")


# In[ ]:



