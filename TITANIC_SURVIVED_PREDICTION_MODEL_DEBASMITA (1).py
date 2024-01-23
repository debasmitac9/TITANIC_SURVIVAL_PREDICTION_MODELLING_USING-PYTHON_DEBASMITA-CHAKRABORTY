#!/usr/bin/env python
# coding: utf-8

# In[93]:


#importing modules
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

### For data analysis and wrangling
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random as rnd

# For visualization
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib.inline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC , LinearSVC
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[94]:


#Reading Training dataset
# Load the train and Display the First 5 Rows of Titanic Dataset
train = pd.read_csv("train.csv")
train.head()


# In[95]:


#Reading Test dataset
# Load the test and Display the First 5 Rows of Titanic Dataset
test = pd.read_csv("test.csv")
test.head()


# In[96]:


#Combine training and testing data
combine = [train,test]


# In[97]:


#Number of Rows and column in training dataset
print('Number of Rows and column in training dataset',train.shape)
print('Number of Rows and column in testing dataset',test.shape)


# In[98]:


#Analyze by describing data
#Which features are available in the dataset?
print('Training columns : ')
print(train.columns.values)
print('Testing columns : ')
print(test.columns.values)


# In[99]:


#Which is the target Column ?
target_variable = list(set(train) - set(test))
print('Target Column is :',target_variable)


# In[100]:


#Feature Data Type:
#1. Numerical:
    #a. Continuos
    #b. Discrete
#2. Categorical:
    #a. Nominal
    #b. Ordinal


# In[101]:


train.info()


# In[102]:


#Inference : By seeing the above feature information we can infer we have 2 float Features , Integer Feature 5 and Categorical feature 5.
Missing Values: We can also see the count of Age(714) , Cabin(204) and Embarked(889) column is less as compared to other columns. that means it contain some missing value


# In[103]:


# select_dtypes : Return a subset of the DataFrame’s columns based on the column dtypes.

#it will return column which contain discrete (integer) value  
discrete_data = train.select_dtypes(include=['int64'])
#it will return column which contain continous (float) value
continous_data = train.select_dtypes(include=['float64'])
#it will return column which contain categorical (object) value
categorical_data = train.select_dtypes(include=['object'])


# In[104]:


print('Discrete Features : ',discrete_data.columns.values)
print('Continous Features : ',continous_data.columns.values)
print('Categorical Features : ',categorical_data.columns.values)


# In[105]:


categorical_data.head()


# In[106]:


#Cabin and Ticket Contain Mixed datatype (numerical + textual) information


# In[107]:


discrete_data.tail()


# In[108]:


continous_data.head()


# In[109]:


#Which Feature may contain errors
train.head()


# In[110]:


train.tail()


# In[111]:


#Which Feature Contain Blank values ?
#Training dataset
sns.heatmap(train.isnull(),yticklabels=False,cmap='bwr')


# In[112]:


train.info()


# In[113]:


#Cabin > Age > Embarked features contain a number of null values in that order for the training dataset.


# In[114]:


#Testing dataset
sns.heatmap(test.isnull(),yticklabels=False,cmap='bwr')


# In[115]:


train.info()
print('--'*40)
test.info()


# In[116]:


#Cabin > Age >Fare Contain number of null values in that order for testing dataset


# In[117]:


#Describing the Data
#Complete training data


# In[118]:


train.describe()


# In[119]:


#This helps us determine, among other early insights, how representative is the training dataset of the actual problem domain.

#Most Passenger (>75%) did not travel with parents or children
#Survived is a categorical feature with 0 or 1 values.
#Around 38% samples survived representative of the actual survival rate at 32%.
#Fare vary significatly for few passengers paying as high as $512


# In[120]:


#Categorical Description
categorical_data.describe()


# In[121]:


#What is the distribution of categorical features?

#Names are unique across the dataset
#Sex variable has two possible values with 65% male
#Embarked takes three possible values. S port used by most passengers (top=S) i.e. Most Passenger has started their jouney from SouthHampton
#Cabin contain several unique values also it several passenger has shared the cabin
#Ticket has high ratio of duplicate values


# In[122]:


#Discreate description
discrete_data.describe()


# In[123]:


continous_data.describe()


# In[124]:


from numpy import percentile
from numpy.random import rand

print('Sibling Ratio : ',percentile(train['SibSp'],[10, 20, 30, 40, 50, 60,65,68, 70, 80, 90]))
print('Fare Charge : ',percentile(train['Fare'],[10, 20, 30, 40, 50, 60,70, 80, 90,95,99]))


# In[125]:


#Nearly 32% of the passenger had the spouse and siblings
#Less than 2% of the passenger had paid more fares


# In[126]:


#Analyze by pivoting features
train['Pclass'].value_counts()


# In[127]:


train[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[128]:


#Pclass We observe significant correlation (>0.5) among Pclass=1 and Survived (classifying #3). We decide to include this feature in our model.


# In[129]:


train[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[130]:


train[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[131]:


#This Features (Parch and SibSp) have zero correlation for certain values. it may be best to derive a new feature from a feature or set of feature from these individual features


# In[132]:


train[['Sex','Survived']].groupby('Sex',as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[133]:


#Sex We confirm the observation during problem definition that Sex=female had very high survival rate at 74% (classifying #1).


# In[134]:


#Analyze by visualizing the data
grid1 =sns.FacetGrid(train,col='Survived')
grid1.map(plt.hist, 'Age', bins=20)


# In[135]:


#Observation:¶
#Infrant(Age<=4) had high survival rate
#Oldest Passenger 80 is Survived.
#Large Number of of people between 16-28 did not Survived
#Most Passenger are in th range of 15-3%5
#Decision:
#This simple analysis confirms our assumptions as decisions for subsequent workflow stages.

#We should consider the Age Feature in our model training
#Complete the Age Feature for null values. (Completion 1)
#we should band age groups (Creation 3)


# In[136]:


#Correlating numerical and ordinal features
grid2 = sns.FacetGrid(train,col='Survived',row='Pclass')
grid2.map(plt.hist,'Age',alpha=0.5,bins=20)
grid2.add_legend()


# In[137]:


#Observations:
#Pclass=3 Had most passenger,however they didn't survived.Confirm Classifying Assumption
#Infrant Passenger in Pclass=2 and Pclass=3. mostly survived.Further qualifies for classification.
#Most Passenger in Pclass=1 has Survived. Confirm Classifying assumption 3
#Pclass varies in terms of Age Distribution
#Decision:
#Consider Pclass for Modelling


# In[138]:


#Correlating categorical features
grid2 = sns.FacetGrid(train,row='Embarked' , aspect=1.6)
grid2.map(sns.pointplot,'Pclass','Survived','Sex', palette='deep')
grid2.add_legend()


# In[139]:


grid2 = sns.FacetGrid(train,row='Embarked',col='Survived',aspect=2.0)
grid2.map(sns.barplot,'Sex','Fare',alpha=0.5,ci=None)
grid2.add_legend()


# In[140]:


#Observations.
#Female passengers had much better survival rate than males. Confirms classifying (#1).
#Exception in Embarked=C where males had higher survival rate. This could be a correlation between Pclass and Embarked and in turn Pclass and Survived, not necessarily direct correlation between Embarked and Survived.
#Males had better survival rate in Pclass=3 when compared with Pclass=2 for C and Q ports. Completing (#2).
#Ports of embarkation have varying survival rates for Pclass=3 and among male passengers. Correlating (#1).
#Decisions.
#Add Sex feature to model training. Complete and add Embarked feature to model training.


# In[141]:


#Data Wrangling
print('Before : ',train.shape,test.shape)
columns = ['Cabin','Ticket']
new_train = train.drop(columns,axis=1)
new_test = test.drop(columns,axis=1) 
combine = [new_train,new_test]
print('After : ',new_train.shape,new_test.shape)


# In[142]:


new_train.head()


# In[143]:


for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
dataset['Title'].head()


# In[144]:


pd.crosstab(new_train['Title'],new_train['Sex'])


# In[145]:


new_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[146]:


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Capt','Col','Countess','Don','Dr','Jonkheer','Major','Rev','Sir','Dona','Lady'],'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
new_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[147]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

new_train.head()


# In[148]:


new_train = new_train.drop(['Name', 'PassengerId'], axis=1)
new_test = new_test.drop(['Name'], axis=1)
combine = [new_train, new_test]
new_train.shape, new_test.shape


# In[149]:


for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({"male" : 0 , "female" : 1}).astype(int)

new_train.head()


# In[150]:


new_train.info()


# In[151]:


grid = sns.FacetGrid(new_train, row='Pclass', col='Sex', aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# In[152]:


guess_ages = np.zeros((2,3))
guess_ages

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess =  guess_df.median()
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = guess_ages[i,j]
    dataset['Age'] = dataset['Age'].astype(int)
new_train.head()


# In[153]:


new_train.info()


# In[154]:


new_train['AgeBand'] = pd.cut(new_train['Age'], 5)
new_train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# In[155]:


new_train.head()


# In[156]:


for dataset in combine:
    dataset.loc[dataset['Age'] <= 16,'Age'] = 0
    dataset.loc[(dataset['Age']>16) & (dataset['Age'] <=32),'Age'] = 1
    dataset.loc[(dataset['Age']>32) & (dataset['Age'] <=48), 'Age'] = 2
    dataset.loc[(dataset['Age']>38) & (dataset['Age'] <=64),'Age'] = 3
    dataset.loc[(dataset['Age']>64), 'Age'] = 4


# In[157]:


new_train.head()


# In[158]:


new_test.head()


# In[159]:


for dataset in combine:
    dataset['Familysize'] = dataset['Parch'] + dataset['SibSp'] + 1
new_train[['Familysize', 'Survived']].groupby(['Familysize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[160]:


for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['Familysize'] == 1,'IsAlone'] = 1


# In[161]:


new_train[['IsAlone','Survived']].groupby(['IsAlone'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[162]:


new_train = new_train.drop(['Parch', 'SibSp', 'Familysize'], axis=1)
new_test = new_test.drop(['Parch', 'SibSp', 'Familysize'], axis=1)


# In[163]:


combine = [new_train, new_test]

new_train.head()


# In[164]:


for dataset in combine:
    dataset['Age*class'] = dataset.Age * dataset.Pclass
new_train.loc[:,['Age*class','Age','Pclass']].head()


# In[165]:


#Completing a categorical feature
freq_port = new_train.Embarked.dropna().mode()[0]
freq_port


# In[166]:


dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
new_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[167]:


#Converting Categorical Feature to numerical
for dataset in combine:
    print(dataset.info())
    print(dataset.dropna(how='any',inplace=True))
    print(dataset.info())


# In[168]:


from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 


# In[169]:


for dataset in combine:
    dataset['Embarked'] = label_encoder.fit_transform(dataset['Embarked'])


# In[170]:


dataset['Embarked'].unique()


# In[171]:


new_train.head()


# In[172]:


#Quick completing and converting a numeric feature
new_test['Fare'].fillna(new_test['Fare'].dropna().median(), inplace=True)
new_test.head()


# In[173]:


new_train['FareBand'] = pd.qcut(new_train['Fare'], 4,duplicates='drop')
new_train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# In[174]:


for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

new_train = new_train.drop(['FareBand','AgeBand'], axis=1)
combine = [new_train, new_test]
    
new_train.head(10)


# In[175]:


#Training a model and requiring solution
X_train = new_train.drop("Survived", axis=1)
Y_train = new_train["Survived"]
X_test  = new_test.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# In[176]:


#1.Logistic Regression
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()
logistic_model.fit(X_train,Y_train)
### Check the training accuracy
training_accuracy = logistic_model.score(X_train,Y_train)
y_pred = logistic_model.predict(X_test)
acc_log = round((training_accuracy)*100,2)
print('Training Accuracy For the Logistic Regression Model is ',acc_log)


# In[177]:


#Coeffecient of each feature
coeff_df = pd.DataFrame(new_train.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logistic_model.coef_[0])
coeff_df.sort_values(by='Correlation', ascending=False)


# In[178]:


#2.Support Vector Machine
from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train,Y_train)
acc_svc = round(svc_model.score(X_train,Y_train) * 100 ,2)
print('Training Accuracy For the SVC Model ',acc_svc)
y_pred = svc_model.predict(X_test)


# In[179]:


#3.K-Nearest Neighbor(KNN) Classification
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print('Training Accuracy For the KNN Neighbour is ',acc_knn)


# In[180]:


#4.Naive Bayes
from sklearn.naive_bayes import GaussianNB
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train,Y_train)
acc_gaussian = round(naive_bayes_model.score(X_train,Y_train)*100,2)
print('Training Accuracy For the Navie Bayes Model is : ',acc_gaussian)
y_pred = naive_bayes_model.predict(X_test)


# In[181]:


#5.Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print('Training Accuracy For the Decision Tree Classifier Model is :',acc_decision_tree)


# In[182]:


#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier()
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print('Training Accuracy For the RandomForestClassifier Model is :',acc_random_forest)


# In[183]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
               'Naive Bayes',  
              'Decision Tree','Random Forest'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_gaussian, acc_decision_tree,acc_random_forest]})
models.sort_values(by='Score', ascending=False)


# In[ ]:




