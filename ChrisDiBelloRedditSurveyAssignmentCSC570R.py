
# coding: utf-8

# In[ ]:



# In[ ]:


# coding: utf-8

# In[ ]:
#Chris DiBello CSC570R - Reddit Survey Assgnment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from string import letters
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
#rom sklearn.model_selection import GridSearchCV
from sklearn import cross_validation
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris


get_ipython().magic(u'pylab inline')
get_ipython().magic(u'matplotlib inline')
#Chris DiBello EDA Lab CSC570R


print '    '
print '    '
print '    '
#Read train file-Plot 
#df = pd.read_csv("RedditShortDemoSurvey-1-Cleaned.csv") #
#df = pd.read_csv("test.csv") #
df2 = pd.read_csv("Countries-Continents-csv.csv") #
#df.describe()


a = pd.read_csv("test.csv")
b = pd.read_csv("Countries-Continents-csv.csv")
b = b.dropna(axis=1)
merged = a.merge(b, on='Country')
merged = merged.groupby('Id').max()
#aggregate countried to continents
print '           '
print '           '

print '    List of Missing Variables'
print merged.apply(lambda x: sum(x.isnull()),axis=0)
print '           '
print '           '
print 'Clean data and Replace missing SubredditData Creating Dummy Variables With Most Common Value askreddit'
merged = merged.replace(np.nan,'askreddit', regex=True)
#merged = merged['SubredditData'].replace(np.nan, 'askreddit', regex=True)
print merged.apply(lambda x: sum(x.isnull()),axis=0)
print '           '
print '           '

stopset = set(stopwords.words('english'))

vectorizer = TfidfVectorizer(stop_words=stopset,
                                 use_idf=True, ngram_range=(1, 3))
stopset.update(['and','game', 'nntp', 'go', 'and', 'the', 'com','After', 'Although', 'As', 'As', 'If', 'As', 'Long', 'As',
'Because', 'Before', 'Even', 'If', 'Even', 'Though', 'If','Once', 'Provided', 'Since', 'So', 'That', 'That',
'Though', 'Till', 'Unless', 'Until', 'What','When', 'Whenever', 'Wherever', 'Whether', 'While', 'And',
'Or', 'But', 'Nor', 'So', 'For', 'Yet','Accordingly','Also', 'Anyway', 'Besides', 'Consequently',
'Finally','For', 'Example', 'For', 'Instance', 'Further', 'Furthermore','Hence','However', 'Incidentally',
'Indeed', 'In', 'Fact','Instead','Likewise', 'Meanwhile', 'Moreover', 'Namely','Now','Of', 'Course',
'On', 'the','Contrary','On', 'the', 'Other','Hand','Otherwise','Nevertheless', 'Next', 'Nonetheless', 'Similarly',
'So', 'Far','Until','Now','Still', 'Then', 'Therefore', 'Thus'   ])

#set vectorizor shape
X = vectorizer.fit_transform(a)



X = pd.DataFrame()
#X['id'] = merged['Id']
X['continent'] = merged['Continent']
X['country'] = merged['Country']
X['age'] = merged['Age']

X['gender'] = merged['Gender']
X['maritalstatus'] = merged['MaritalStatus']
X['militaryservice'] = merged['MilitaryService']
X['employmentstatus'] = merged['EmploymentStatus']
X['havechildrenunder18'] = merged['HaveChildrenUnder18']
X['educationlevel'] = merged['EducationLevel']
X['income'] = merged['Income']
X['subredditdata'] = merged['SubredditData']
X['dogorcat'] = merged['DogOrCat']
X['typeofCheese'] = merged['TypeOfCheese']

print '           '
print '           '
print 'Aggregate Countries To Continents:'
merged= merged.groupby(['Country', 'Continent'])['Country'].count() 
print merged

X.drop_duplicates()


print '           '
print '           '
print 'Print Raw Data Without USStates:'
print X

print '           '
print '           '
print 'Visualization Of Data:'
print X.describe()

print '           '
print '           '
get_ipython().magic(u'pylab inline')
get_ipython().magic(u'matplotlib inline')
print '           '
print '           '
#sns.barplot(x="employmentstatus", y="gender", data=X);
print '           '
print '           '
f, ax = plt.subplots(figsize=(7, 3))
sns.countplot(y="age", data=X, color="c");
print '           '
print '           '
f, ax = plt.subplots(figsize=(7, 3))
sns.countplot(y="employmentstatus", data=X, color="c");
print '           '
print '           '
f, ax = plt.subplots(figsize=(7, 3))
sns.countplot(y="gender", data=X, color="c");
print '           '
print '           '
f, ax = plt.subplots(figsize=(7, 3))
sns.countplot(y="employmentstatus", data=X, color="c");
print '           '
print '           '
f, ax = plt.subplots(figsize=(7, 3))
sns.countplot(y="maritalstatus", data=X, color="c");
print '           '
print '           '
f, ax = plt.subplots(figsize=(7, 3))
sns.countplot(y="militaryservice", data=X, color="c");
print '           '
print '           '
f, ax = plt.subplots(figsize=(7, 3))
sns.countplot(y="havechildrenunder18", data=X, color="c");
print '           '
print '           '
f, ax = plt.subplots(figsize=(7, 3))
sns.countplot(y="employmentstatus", data=X, color="c");
print '           '
print '           '
f, ax = plt.subplots(figsize=(7, 3))
sns.countplot(y="educationlevel", data=X, color="c");
print '           '
print '           '
f, ax = plt.subplots(figsize=(7, 3))
sns.countplot(y="income", data=X, color="c");
print '           '
print '           '
f, ax = plt.subplots(figsize=(7, 3))
sns.countplot(y="dogorcat", data=X, color="c");
print '           '
print '           '
f, ax = plt.subplots(figsize=(7, 3))
sns.countplot(y="typeofCheese", data=X, color="c");
print '           '
print '           '

sns.set(style="white")
print 'Graph Results of Pearsons Correlation Coefficient'
# Generate a large random dataset
#rs = np.random.RandomState(33)
#d = pd.DataFrame(data=rs.normal(size=(100, 26)),
#                 columns=list(letters[:26]))
#rs = np.random.RandomState(33)
#d = pd.DataFrame(data=rs.normal(size=(100, 26)),
# columns=list(letters[:26]))
#   
# Compute the correlation matrix
#rs = a
print X.head(26)
rs = np.random.RandomState(33)
d = pd.DataFrame(data=rs.normal(size=(100, 26)),
                 columns=X.head(26))

# Compute the correlation matrix
corr = d.corr()


# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
#sns.countplot(y=corr, data=X, color="c");



# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
            square=True, xticklabels=5, yticklabels=5,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

print '       '
print '       '
print 'Use Random Forest To Predict Employment'
print '       '
print '     '
print 'Convert String Data to Numeric'
print '       '
print '     '
y = X.pop("employmentstatus")
from sklearn.preprocessing import LabelEncoder
#convert string data to integer-i don't see how to run Random Forest on string data



X.fillna("Missing", inplace=True)
enc = LabelEncoder()

enc.fit(X['continent'])
X['continent'] = enc.transform(X['continent'])
enc.fit(X['country'])
X['country'] = enc.transform(X['country'])
enc.fit(X['age'])
X['age'] = enc.transform(X['age'])
enc.fit(X['gender'])
X['gender'] = enc.transform(X['gender'])
enc.fit(X['maritalstatus'])
X['maritalstatus'] = enc.transform(X['maritalstatus'])

enc.fit(X['havechildrenunder18'])
X['havechildrenunder18'] = enc.transform(X['havechildrenunder18'])
enc.fit(X['educationlevel'])
X['educationlevel'] = enc.transform(X['educationlevel'])
enc.fit(X['income'])
X['income'] = enc.transform(X['income'])
enc.fit(X['subredditdata'])
#nc.fit(X['employmentstatus'])  #employmentstatus not working use income
X['employmentstatus'] = X['income']
X['subredditdata']  = enc.transform(X['subredditdata'])
enc.fit(X['dogorcat'])
X['dogorcat']  = enc.transform(X['dogorcat'])
enc.fit(X['typeofCheese'])
X['typeofCheese']  = enc.transform(X['typeofCheese'])
enc.fit(X['militaryservice'])
X['militaryservice']  = enc.transform(X['militaryservice'])
print(X)
y = X.pop("employmentstatus")

rf_model = RandomForestClassifier(n_estimators=1000, # Number of trees
                                  max_features=7,    # Num features considered
                                  oob_score=True)    # Use OOB scoring*

#features = ['income','subredditdata','subredditdata','dogorcat','typeofCheese','militaryservice','employmentstatus']

# Train the model
rf_model.fit(X,y)
X.head(26)
print("OOB accuracy: ")
print(rf_model.oob_score_)
plt.figure()
plt.plot(rf_model.oob_score_, label='Prediction Score')
plt.plot(y, label='Prediction')
plt.plot(X.head(26), '--', label='Test Data Set')
plt.legend(loc='best')
plt.title("Random Forest Prediction ")
plt.show()
 

 

get_ipython().magic(u'pylab inline')
get_ipython().magic(u'matplotlib inline')


show()
y = label_binarize(y, classes=[0, 1, 2, 3])
clf_SVM = OneVsRestClassifier(LinearSVC())
params = {
      'estimator__C': [0.5, 1.0, 1.5],
      'estimator__tol': [1e-3, 1e-4, 1e-5],
      }
gs = GridSearchCV(clf_SVM, params, cv=5, scoring='roc_auc')
gs.fit(X, y)


X.describe()
# Impute Age with mean
#["Age"].fillna(X.Age.mean(), inplace=True)

# Confirm the code is correct
X.describe()
numeric_variables = list(X.dtypes[X.dtypes != "object"].index)
model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)

# I only use numeric_variables because I have yet to dummy out the categorical variables

model.fit(X[numeric_variables], y)

model.oob_score_
y_oob = model.oob_prediction_

print "c-stat: ", roc_auc_score(y, y_oob)
# Here is a simple function to show descriptive stats on the categorical variables
def describe_categorical(X):
    """
    Just like .describe(), but returns the results for
    categorical variables only.
    """
    from IPython.display import display, HTML
    display(HTML(X[X.columns[X.dtypes == "object"]].describe().to_html()))
#.drop(["Name", "Ticket", "PassengerId"], axis=1, inplace=True)
# Change the Cabin variable to be only the first letter or None
def clean_cabin(x):
    try:
        return x[0]
    except TypeError:
        return "None"

#["Cabin"] = X.Cabin.apply(clean_cabin)
#ategorical_variables = ['Sex', 'Cabin', 'Embarked']

#for variable in categorical_variables:
    # Fill missing data with the word "Missing"
    #[variable].fillna("Missing", inplace=True)
    # Create array of dummies
#    dummies = pd.get_dummies(X[variable], prefix=variable)
    # Update X to include dummies and drop the main variable
#    X = pd.concat([X, dummies], axis=1)
#    X.drop([variable], axis=1, inplace=True)
# Look at all the columns in the dataset
def printall(X, max_rows=10):
    from IPython.display import display, HTML
    display(HTML(X.to_html(max_rows=max_rows)))
    
printall(X)
model = RandomForestRegressor(100, oob_score=True, n_jobs=-1, random_state=42)
model.fit(X, y)

print "C-stat: ", roc_auc_score(y, model.oob_prediction_)
model.feature_importances_
get_ipython().magic(u'pylab inline')
get_ipython().magic(u'matplotlib inline')
# Simple version that shows all of the variables
print 'model'
print model.feature_importances_
print 'index'
print X.columns
printall(X)
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.sort_values(inplace=True)
feature_importances.plot(kind="barh", figsize=(7,6));