# SVM and Random Forest
# Team Project

import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import svm
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import scikitplot as skplt
import matplotlib.pyplot as plt

def char_flag(l):
    if l <= 40:
        return 1
    elif l <= 60:
        return 2
    elif l <= 80:
        return 3
    elif l <= 120:
        return 4
    elif l <= 160:
        return 5
    else:
        return 6

df = pd.read_csv("spam.csv", encoding='latin-1')
df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])

df_data = df['v2']

token_1 = []
token_2 = []
token_3 = []
token_4 = []
token_5 = []
token_6 = []

for d in df_data:
    token_1.append(re.sub('[^A-Za-z]+', '', d.lower()))
    token_2.append(d.count('$'))
    x = re.sub('[^0-9]+', '', d.lower())
    token_3.append(len(x))
    token_4.append(len(d))
    token_5.append(char_flag(len(d)))
    y = re.sub(r'[^://@]', '', d.lower())
    token_6.append(y != '')


data = [np.array([token_2[i], token_3[i], token_4[i], token_5[i], token_6[i]]) for i in range(len(df_data))]
df_labels = df['v1']
trainset, testset, trainlabel, testlabel = train_test_split(data, df_labels, test_size=0.33, random_state=42)
print(len(trainset))
print(len(trainlabel))
print(len(testset))
print(testlabel)
#countvect= CountVectorizer()
#x_counts = countvect.fit(trainset.v2)
#preparing for training set
#x_train_df =countvect.transform(trainset.v2)
#a = x_train_df.toarray()
#print(x_train_df.toarray())
#print(a[1][257])
#print(x_train_df.shape)
#preparing for test set
#x_test_df = countvect.transform(testset.v2)

        
SVM = svm.SVC()
SVM.fit(trainset,trainlabel)
predicted_values_svm=SVM.predict(testset)
print(predicted_values_svm)
acurracy_SVM = accuracy_score(testlabel,predicted_values_svm)
print("acurracy_SVM " +str(acurracy_SVM))
confusion_matrix_SVM = confusion_matrix(testlabel,predicted_values_svm,labels=["spam", "ham"])
print(confusion_matrix_SVM)
skplt.metrics.plot_confusion_matrix(testlabel,predicted_values_svm, normalize=True)
plt.show()
