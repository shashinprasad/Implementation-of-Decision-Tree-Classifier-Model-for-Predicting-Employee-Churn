# EX-06 Implementation of Decision Tree Classifier Model for Predicting Employee Churn
### AIM:
To write a program to implement the Decision Tree Classifier &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;: <br>
Model for Predicting Employee Churn.
### Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
### Algorithm
1. Import pandas and read the csv file.
2. Import Decision tree classifier.
3. Fit the data in the model.
4. Find the accuracy score.
### Program:
Developed By: Shashin prasad.S
Register No: 212222230144
```Python
import pandas as pd
df=pd.read_csv("CSVs/Employee.csv")
df.head()
df.info()
df.isnull().sum()
df['left'].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["salary"]=le.fit_transform(df['salary'])
df.head()
x=df[['satisfaction_level','last_evaluation','number_project','average_montly_hours',
      'time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()
y=df['left']
from sklearn.model_selection import train_test_split as tts
Xtrain,Xtest,Ytrain,Ytest=tts(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(Xtrain,Ytrain)
Ypred=dt.predict(Xtest)
from sklearn import metrics
accuracy=metrics.accuracy_score(Ytest,Ypred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
### Output:
**df.head()** <br>
<img src="https://github.com/ROHITJAIND/EX-06-Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707073/70c2b03d-9c32-4044-8847-08fb925602ee">
<br>
<br>
**df.info()**&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**df.isnull().sum()**&emsp;&emsp;&emsp;&emsp;&emsp;**df['left'].value_counts()** <br>
<img valign=top src="https://github.com/ROHITJAIND/EX-06-Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707073/1218f5aa-3253-42ac-a008-453d6ab1a0fb">&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
<img valign=top src="https://github.com/ROHITJAIND/EX-06-Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707073/a7a3e9c9-cdbd-44eb-8b09-9207c76a3738">&emsp;&emsp;&emsp;
<img valign=top src="https://github.com/ROHITJAIND/EX-06-Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707073/cff84f6e-783c-4353-a3c0-924faefecf1a">
<br>
<br>
**Label Encoding for String values**<br>
<img valign=top src="https://github.com/ROHITJAIND/EX-06-Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707073/ca70ed9a-721a-4a11-bd90-618dcc2070dd">
<br>
<br>
**x.head()**<br>
<img valign=top src="https://github.com/ROHITJAIND/EX-06-Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707073/1b0dbd7d-ddee-420a-9cb6-fa04fdaabe09">
<br>
<br>
**Accuracy:** &emsp;&emsp;&emsp;&emsp;**Prediction:**<br>
<img valign=top src="https://github.com/ROHITJAIND/EX-06-Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707073/050a0166-65c0-43fe-89eb-4d1baf4c127b">&emsp;&emsp;&emsp;&emsp;
<img valign=top src="https://github.com/ROHITJAIND/EX-06-Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707073/45d731a0-e4f8-43c0-812a-b99e7cf305da">














### Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
