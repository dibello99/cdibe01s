
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
get_ipython().magic(u'pylab inline')
get_ipython().magic(u'matplotlib inline')
#Chris DiBello EDA Lab CSC570R

print '    List of Missing Variables in Train.csv'
print '    Results from -df.apply(lambda x: sum(x.isnull()),axis=0) '
print '    '
print '    Age-Missing 177 Values'
print '    '
print '    Cabin-Missing 687 Values'
print '    '
print '    Embarked-Missing 2 Values'
print '    '
print '    Continuous variables: '
print '    PassengerId, Age, Fare,  '
print '    '
print '    Categorical variables: '
print '    Survived, Pclass, Sex, SibSp, Ticket, Parch, Cabin, Embarked    '       

show()
print '    '
print '    '
print '    '
#Read train file-Plot 
df = pd.read_csv("/train/train.csv") #

#Read train file-Plot Fare


titanic_plot = df['Fare'].hist(bins=10)
titanic_plot.set_title("Fare Histogram")
titanic_plot.set_xlabel("Total Fare")
titanic_plot.set_ylabel("Number of Passengers")
plt.show()
#plot Age

titanic_plot = df['Age'].hist(bins=5)
titanic_plot.set_title("Age Histogram")
titanic_plot.set_xlabel("Age")
titanic_plot.set_ylabel("Number of Passengers")
plt.show()
#plot Pclass

titanic_plot = df['Pclass'].hist(bins=5)
titanic_plot.set_title("Pclass Histogram")
titanic_plot.set_xlabel("Pclass")
titanic_plot.set_ylabel("Number of Passengers")
plt.show()
#plot survived

titanic_plot = df['Survived'].hist(bins=5)
titanic_plot.set_title("Survived Histogram")
titanic_plot.set_xlabel("Number Survived")
titanic_plot.set_ylabel("Number of Passengers")
plt.show()
#plot SibSb

titanic_plot = df['SibSp'].hist(bins=5)
titanic_plot.set_title("SibSp Histogram")
titanic_plot.set_xlabel("SibSb")
titanic_plot.set_ylabel("Number of Passengers")
plt.show()
#plot Parch

titanic_plot = df['Parch'].hist(bins=5)
titanic_plot.set_title("Parch Histogram")
titanic_plot.set_xlabel("Parch")
titanic_plot.set_ylabel("Number of Passengers")
plt.show()

# Plot passengerid
titanic_plot = df['PassengerId'].plot(kind='bar')
titanic_plot.set_title("Passengerid Bar Graph")
titanic_plot.set_xlabel("Passengerid")
titanic_plot.set_ylabel("Number of Passengers")
plt.show()
#Show distribution of Tickets
print '    '
print '    Distribution of Tickets'

print df['Ticket'].value_counts(ascending=True)
plt.show()
print '    '
#Show distribution by Sex
print '    Distribution by sex'

print df['Sex'].value_counts(ascending=True)
plt.show()

print '   '
print '   '

print '    Distribution of Cabin'

print df['Cabin'].value_counts(ascending=True)
plt.show()

print '   '
print '   '
print '    Distribution of Embarked'

print df['Embarked'].value_counts(ascending=True)
plt.show()

#account for missing values of Age

print '    List of Missing Variables in Train.csv'
print df.apply(lambda x: sum(x.isnull()),axis=0)
plt.show()
print '   '
print '   '
#plot Age
get_ipython().magic(u'pylab inline')
get_ipython().magic(u'matplotlib inline')

print '    Age Histogram-Recalulate with average age replacing missing age values with mean age'

average = df.Age.mean()
df.Age=df.Age.fillna(value=average)
titanic_plot = df['Age'].hist(bins=5)
titanic_plot.set_title("Age Histogram")
titanic_plot.set_xlabel("Age")
titanic_plot.set_ylabel("Number of Passengers")
plt.show()
print '   '
print '   '
#Show Min, Max, Mean, and Standard Deviation
print '    Min, Max, Mean, and Standard Deviation of the numerical and continuous variables'


df.describe()

