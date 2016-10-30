
# coding: utf-8

# In[ ]:

#CSC470R Chris DiBello Week 9 Assignment
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
df= pd.read_csv("SMSSpamCollection",sep='\t', names=['spam', 'txt'])
df.head()
from sklearn.feature_extraction import DictVectorizer
dvec = DictVectorizer(sparse=False)
#print pd.get_dummies(df)
#X = dvec.fit_transform(df.transpose().to_dict().values())
df_spam = pd.get_dummies(df['spam'])
#print df_spam
df_new = pd.concat([df, df_spam], axis=1)
print 'Show Binary Indicator Variables'
print df_new
print df_new.describe()
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(df_new)
X_train_counts.shape
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape
print 'Show TFIDF Vectors'
print X_train_tf
X = pd.DataFrame()
y = pd.DataFrame()
X['txt'] = df_new['txt']
X['ham'] = df_new['ham']

#y['txt'] = df_new['txt']
#y['ham'] = df_new['ham']
enc = LabelEncoder()
#binarize string variables
enc.fit(X['txt'])
X['txt'] = enc.transform(X['txt'])
enc.fit(X['ham'])
X['ham'] = enc.transform(X['ham'])
#enc.fit(y['txt'])
#y['txt'] = enc.transform(y['txt'])
#enc.fit(y['ham'])
#y['ham'] = enc.transform(y['ham'])
y = X.pop("txt")
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, y)
GaussianNB(priors=None)
print 'Show Naive Bayes Classifier'
#print(clf.predict([[-0.8, -1]]))

clf_pf = GaussianNB()
clf_pf.partial_fit(X,y, np.unique(y))
GaussianNB(priors=None)
print(clf_pf.predict([[-0.8, -1]]))

get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u'pylab inline')
probs = clf_pf.predict_proba(X)[:, 1]
print(roc_auc_score(X, probs))
logit_roc_auc = roc_auc_score(X, probs)
print 'Show roc_auc_score score'
print "Logistic AUC = %2.2f" % logit_roc_auc
#print classification_report(X, probs)
from sklearn.metrics import roc_curve
print 'Visualize AUC score'
fpr, tpr, thresholds = roc_curve(X, probs)
# Plot of a ROC curve 
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[ ]:



