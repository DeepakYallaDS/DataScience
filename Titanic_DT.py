# =============================================================================
# Titanic survival prediction -- Kaggle 
# Decision Tree
# 15th Nov, 2018 
# =============================================================================

#importing pandas for data frames
import pandas as pd

#import tree from sklearn
from sklearn import tree

#Use train data
titanic_train=pd.read_csv("D:\\Kaggle\\train.csv")

titanic_train.shape

titanic_train.info()

titanic_train.describe()

##Sex,Ticket,Name,Cabin,Embarked -- not able to convert into float
##Fare -- not having all values in test
##Age -- having NaN null values in train

#Creating model
#x_titanic_train=titanic_train[['Pclass','SibSp','Parch']] ## 0.68421
#x_titanic_train=titanic_train[['SibSp','Parch','Pclass']]  ## 0.68421
#x_titanic_train=titanic_train[['Parch','Pclass','SibSp']]  ## 0.6842144

#x_titanic_train=titanic_train[['Pclass','Parch','SibSp']]## 0.6842144
x_titanic_train=titanic_train[['Parch','SibSp']]
#x_titanic_train=titanic_train[['Pclass','SibSp']]#0.63636
#x_titanic_train=titanic_train[['Pclass','Parch']]##0.67942
y_titanic_train=titanic_train['Survived']

dt=tree.DecisionTreeClassifier()
dt.fit(x_titanic_train,y_titanic_train)

#Using test data
titanic_test=pd.read_csv("D:\\Kaggle\\test.csv")

titanic_test.shape

titanic_test.info()

titanic_test.describe()

#Predict the test data
#x_titanic_test=titanic_test[['Pclass','SibSp','Parch']]
x_titanic_test=titanic_test[['Parch','SibSp']]

titanic_test['Survived']=dt.predict(x_titanic_test)

titanic_test.to_csv("D:\\Kaggle\\submission_titanic.csv", columns=['PassengerId','Survived'], index=False)

