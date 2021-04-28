import joblib
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score


## stopwords are words used very frequently, but dont contribute much to the meaning of a sentence.
#print(stopwords.words('english')[0:15])

data = pd.read_csv("SMSSpamCollection",header=None,sep='\t')
data.columns=['label','body']
print(data.head())
print("Dataset has {} rows and {} col".format(len(data),len(data.columns)))
print("{} spam and {} ham".format(len(data[data['label']=='spam']),
                                  len(data[data['label']=='ham'])))
print("nulls in label are {}".format(data['label'].isnull().sum()))
print("nulls in body are {}".format(data['body'].isnull().sum()))

df = data;
df['len']=df['body'].apply(len)
print(df.head())
##Stemming is used to reduce redundancy
##as most of the time the word stem and their inflected/derived words mean the same.
ps = PorterStemmer()
message = []
for i in range(0, df.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', df['body'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    message.append(review)

df['clean_msg']=np.empty((len(message),1))
for i in range(len(message)):
    df['clean_msg'][i]=message[i]
df['clean_msg_len']=df['clean_msg'].apply(len)
print(df.head())

df['body'].describe()
df['clean_msg'].describe()


df=pd.concat([df, pd.get_dummies(df['label'])], axis=1)
df.drop(['label'],axis=1,inplace=True)
df.drop(['spam'],axis=1,inplace=True)
df.rename(columns={'ham':'label'},inplace=True)
print(df.head())
##1 - Ham, 0 - Spam

#vectorization

X=df['clean_msg']
cv = CountVectorizer(max_features=2500) ##feature selection
X = cv.fit_transform(message).toarray()
Y=np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.10, random_state = 0)
print("X_train shape: {}\n X_test shape: {}\nY_train shape: {}\nY_test shape: {}".format(X_train.shape,X_test.shape,y_train.shape,y_test.shape))
joblib.dump(X_test,'x_test_nlp1.joblib')
joblib.dump(y_test,'y_test_nlp1.joblib')

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("SVM's Accuracy:{0}".format(accuracy_score(y_test, y_pred)))

joblib.dump(clf, "spamhamSVM.joblib")





'''
##data cleaning
rawdata = open("SMSSpamCollection").read()
#print(rawdata[0:500])
##replace every \t by \n
parsedData = rawdata.replace('\t', '\n').split('\n')
labelList = parsedData[0::2] ##position 0 to end
textList = parsedData[1::2]

#print(labelList[0:5])
#print(textList[0:5])
#print(len(labelList)," ", len(textList))
#print(labelList[-5:]) #print last 5 items
##ignore extra last element from labellist
datafinal = pd.DataFrame({'label':labelList[:-1],
                           'body_list': textList});
print(datafinal.head())
'''