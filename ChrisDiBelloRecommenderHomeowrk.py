
# coding: utf-8

# In[ ]:

#Chris DiBello CSC570R Recommender Homework
import pandas as pd
import numpy as np
M = pd.read_csv('Movie Reviews.csv', index_col='Name')
#M.drop(['timestamp'], axis=1, inplace=True)
print 'Describe Data'
print M.describe()
def pearson(s1, s2):
#    """Take two pd.Series objects and return a pearson correlation."""
    s1_c = s1 - s1.mean()
    s2_c = s2 - s2.mean()
    return np.sum(s1_c * s2_c) / np.sqrt(np.sum(s1_c ** 2) * np.sum(s2_c ** 2))
print '    '
#Find movie most similar to the fault in our stars:
print 'Movies With Similar Ranking to The Fault In Our Stars:'
#print 'The Hunger Games: Mockingjay - Part 1'
countP =  pearson(M['The Fault in Our Stars'], M['The Hunger Games: Mockingjay - Part 1'])

#print 'American Sniper'
countM = pearson(M['The Fault in Our Stars'], M['American Sniper'])
if countP > countM:
    countM = countP
    save = 'American Sniper'
#print 'Guardians of the Galaxy'
countP = pearson(M['The Fault in Our Stars'], M['Guardians of the Galaxy'])
if countP > countM:
    countM = countP
    save = 'Guardians of the Galaxy'
#print 'The Lego Movie'
countP = pearson(M['The Fault in Our Stars'], M['The Lego Movie'])
if countP > countM:
    countM = countP
    save = 'The Lego Movie'
#print 'The Hobbit'
countP = pearson(M['The Fault in Our Stars'], M['The Hobbit'])
if countP > countM:
    countM = countP
    save = 'The Hobbit'
#print 'Transformers'
countP = pearson(M['The Fault in Our Stars'], M['Transformers'])
if countP > countM:
    countM = countP
    save = 'Transformers'
#print 'Malificent'
countP = pearson(M['The Fault in Our Stars'], M['Malificent'])
if countP > countM:
    countM = countP
    save = 'Malificent'
#print 'Big Hero 6'
countP = pearson(M['The Fault in Our Stars'], M['Big Hero 6'])
if countP > countM:
    countM = countP
    save = 'Big Hero 6'
#print 'Godzilla'
countP = pearson(M['The Fault in Our Stars'], M['Godzilla'])
if countP > countM:
    countM = countP
    save = 'Godzilla'
#print 'Interstellar'
countP = pearson(M['The Fault in Our Stars'], M['Interstellar'])
if countP > countM:
    countM = countP
    save = 'Interstellar'
#print 'How to Train your Dragon 2'
countP = pearson(M['The Fault in Our Stars'], M['How to Train your Dragon 2'])
if countP > countM:
    countM = countP
    save = 'How to Train your Dragon 2'
#print 'Gone Girl'
countP = pearson(M['The Fault in Our Stars'], M['Gone Girl'])

if countP > countM:
    countM = countP
    save = 'Gone Girl'
#print 'Divergent'
countP = pearson(M['The Fault in Our Stars'], M['Divergent'])
if countP > countM:
    countM = countP
    save = 'Divergent'
#print 'Unbroken'
countP = pearson(M['The Fault in Our Stars'], M['Unbroken'])
if countP > countM:
    countM = countP
    save = 'Unbroken'
#print '300: Rise of an Empire'

countP = pearson(M['The Fault in Our Stars'], M['300: Rise of an Empire'])
if countP > countM:
    countM = countP
    save = '300: Rise of an Empire'
print '    '
print 'The movie most similar to The Fault In Our Stars is: '  + save
print  countM 
print '    '
#Find movie with the highest ranking.
#print M['The Fault in Our Stars'].value_counts()
count =  len(M.groupby(['The Hunger Games: Mockingjay - Part 1']).groups[5])
countnext =  len(M.groupby(['American Sniper']).groups[5])
if countnext > count:
    count = countnext
    save = 'American Sniper'
countnext =  len(M.groupby(['Guardians of the Galaxy']).groups[5])
if countnext > count:
    count = countnext
    save = 'Guardians of the Galaxy'
count =  len(M.groupby(['The Lego Movie']).groups[5])
if countnext > count:
    count = countnext
    save = 'The Lego Movie'
count =  len(M.groupby(['The Hobbit']).groups[5])
if countnext > count:
    count = countnext
    save = 'The Hobbit'
count =  len(M.groupby(['Transformers']).groups[5])
if countnext > count:
    count = countnext
    save = 'Transformers'
count =  len(M.groupby(['Malificent']).groups[5])
if countnext > count:
    count = countnext
    save = 'Malificent'
count =  len(M.groupby(['Big Hero 6']).groups[5])
if countnext > count:
    count = countnext
    save = 'Big Hero 6'
count =  len(M.groupby(['Godzilla']).groups[5])
if countnext > count:
    count = countnext
    save = 'Godzilla'
count =  len(M.groupby(['Interstellar']).groups[5])
if countnext > count:
    count = countnext
    save = 'Interstellar'
count =  len(M.groupby(['Gone Girl']).groups[5])
if countnext > count:
    count = countnext
    save = 'Gone Girl'
count =  len(M.groupby(['Divergent']).groups[5])
if countnext > count:
    count = countnext
    save = 'Divergent'
count =  len(M.groupby(['Unbroken']).groups[5])
if countnext > count:
    count = countnext
    save = 'Unbroken'
count =  len(M.groupby(['300: Rise of an Empire']).groups[5])
if countnext > count:
    count = countnext
    save = '300: Rise of an Empire'
print '    '
print '    '
print 'The movie I would like to see based on the highest number of 5 rankings is:'
print save
print 'Total Category 5 Rankings: ' 
print count
def get_recs(movie_name, M, num):

    import numpy as np
    reviews = []
    for title in M.columns:
        if title == movie_name:
            continue
        cor = pearson(M[movie_name], M[title])
        if np.isnan(cor):
            continue
        else:
            reviews.append((title, cor))
    
    reviews.sort(key=lambda tup: tup[1], reverse=True)
#    print 'xxx'
#    print reviews[:num]
    return reviews[:num]
recs = get_recs('The Fault in Our Stars', M, 10)
#print 'yyy'
#print recs[:10]
anti_recs = get_recs('The Fault in Our Stars', M, 8551)
#print 'zzz'
#print anti_recs[-10:]


# In[ ]:



