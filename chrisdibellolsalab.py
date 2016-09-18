
# coding: utf-8

# In[ ]:


#Lsa Lab CSC570R-Chris DiBello
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
#Retrieve newsgroup from Sklearn
from sklearn.datasets import fetch_20newsgroups
categories = ['rec.sport.baseball']
dataset = fetch_20newsgroups(subset='all',shuffle=True, random_state=42, categories=categories)
corpus = dataset.data
#set stop words to english and additional updates
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
X = vectorizer.fit_transform(corpus)
X[0]
print X[0]
X.shape
lsa = TruncatedSVD(n_components=20, n_iter=100)
lsa.fit(X)
#This is the first row for V
lsa.components_[0]
import sys
#print (sys.version)
terms = vectorizer.get_feature_names()
#print concepts
for i, comp in enumerate(lsa.components_): 
    termsInComp = zip (terms,comp)
    sortedTerms =  sorted(termsInComp, key=lambda x: x[1], reverse=True) [:10]
    print "Concept %d:" % i
    for term in sortedTerms:
        print term[0]
    print " "
lsa.components_