import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
df_train = pd.read_csv("E:\dataset\self end to end\Corona_NLP_train.csv")
df2_test = pd.read_csv("E:\dataset\self end to end\Corona_NLP_test.csv")

df = pd.concat([df_train , df2_test])
#df.head()
#df.columns

df.isnull().sum() #Location columns has all Nun values

#### Droping unnecessary columns

df = df.iloc[:,4:]

plt.figure(figsize = (10,8))
sns.countplot(y = df['Sentiment'])
plt.show()

## Text processing ##

def clean_text(text):
    text = re.sub('@[A-Z-a-z0-9]+' , '' , text)  # + is for 1 and more usernames
    text = re.sub('#' ,'', text)  #remove hashtags
    text = re.sub('RT[\s]+' , '' , text) #removing RT
    text = re.sub('https?:\/\/\S+' , '' , text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in set(stopwords.words('english'))]
    words = [ps.stem(word) for word in words]
    words = ' '.join(words)
    
    
    return words

df['OriginalTweet']  = df['OriginalTweet'].apply(clean_text)





#############################################################################
                  #Classify with Keras and tensorflow
#############################################################################

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

                        #MODEL BUILFING#
                             #
corpus = all_words.split(' ')
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
tfidf.fit(df['OriginalTweet'])
X = tfidf.transform(df['OriginalTweet'])
tfidf.get_feature_names()

y = pd.get_dummies(df['Sentiment'] ,drop_first =True)

import pickle
pickle.dump(tfidf, open('tfidf-transform.pkl', 'wb'))



############################################################################### 

                            #MODEL BUILFING#
                                  #
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB(alpha=0.3)
classifier.fit(X_train, y_train)

filename = 'corona sentiment-mnb-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))

#print("Train Accuracy:",classifier.predict(X_train, y_train))
y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred) 

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test , y_pred )

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred  = lr.predict(X_test)
accuracy_score(y_test, lr_pred) #logistic is giving better results

cf_matrix = confusion_matrix(y_test , lr_pred)

labels = ['True Neg','False Pos','False Neg','True Pos']
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix , annot=labels , fmt = '' , cmap = 'Blues')

filename2 = 'corona sentiment-logreg-model.pkl'
pickle.dump(lr, open(filename2, 'wb'))


