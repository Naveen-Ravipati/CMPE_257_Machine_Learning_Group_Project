# SVM and Random Forest
# Team Project

from sklearn_pandas import DataFrameMapper
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import scikitplot as skplt
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

stemmer = PorterStemmer()
emotions_list = [":)", ":(", ":p", ":D", "-_-", ":o"]
import wordcloud


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


def preprocessing_text():
    for d, ln in zip(messages_data, token_4):
        token_1.append(d)
        token_2.append(d.count('$'))
        x = re.sub('[^0-9 ]+', '', d.lower())
        token_3.append(len(x))
        token_5.append(char_flag(ln))
        if (re.sub(r'[^://@]', '', d.lower())) is not '':
            token_6.append(1)
        else:
            token_6.append(0)
        token_7.append(len(x.split()))
        for emoji in emotions_list:
            if re.search(re.escape(emoji), d):
                token_8.append(1)
            else:
                token_8.append(0)
    return np.array(
        [np.array([token_1[i], token_2[i], token_3[i], token_4[i], token_5[i], token_6[i], token_7[i], token_8[i]],
                  dtype=object) for i in
         range(len(messages_data))])


def text_process(mess):
    no_punct = re.sub('[^A-Za-z ]+', '', mess.lower())
    return np.array([stemmer.stem(word) for word in no_punct.split() if word not in stopwords.words('english')])


def svm_fit():
    SVM = svm.SVC(gamma='scalar')
    SVM.fit(trainset, trainlabel)
    predicted_values_svm = SVM.predict(testset)
    print(predicted_values_svm)
    acurracy_SVM = accuracy_score(testlabel, predicted_values_svm)
    print("acurracy_SVM " + str(acurracy_SVM))
    confusion_matrix_SVM = confusion_matrix(testlabel, predicted_values_svm, labels=["ham", "spam"])
    print(confusion_matrix_SVM)
    skplt.metrics.plot_confusion_matrix(testlabel, predicted_values_svm, normalize=True)
    plt.show()


if __name__ == "__main__":
    messages = pd.read_csv("spam.csv", encoding='latin-1')
    messages = messages.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
    emotions_list = [':)', ':(', ':p', ':D', '-_-', ':o']
    print(messages.head(5))
    messages_labels = messages['v1']
    messages['length'] = messages['v2'].apply(len)
    messages_data = messages['v2']

    token_1 = []
    token_2 = []
    token_3 = []
    token_4 = messages['length']
    token_5 = []
    token_6 = []
    token_7 = []
    token_8 = []

    data = preprocessing_text()
    labels = ['message', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7']
    df = pd.DataFrame.from_records(data, columns=labels)
    mapper = DataFrameMapper([
        (['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7'], None),
        ('message', CountVectorizer(analyzer=text_process, ngram_range=(2, 2)))])
    X = mapper.fit_transform(df)
    trainset, testset, trainlabel, testlabel = train_test_split(X, messages_labels, test_size=0.33, random_state=42)

    svm_fit()

    clf = MultinomialNB()
    clf.fit(trainset, trainlabel)
    predicted_values_NB = clf.predict(testset)
    print(predicted_values_NB)
    acurracy_NB = accuracy_score(testlabel, predicted_values_NB)
    print("acurracy_NB " + str(acurracy_NB))
    confusion_matrix_SVM = confusion_matrix(testlabel,predicted_values_NB,labels=["ham","spam"] )
    print(confusion_matrix_SVM)
    skplt.metrics.plot_confusion_matrix(testlabel,predicted_values_NB, normalize=False)
    plt.show()



    spam_words = ' '.join(list(messages[messages['v1'] == 'spam']['v2']))
    spam_wc = WordCloud(width=512, height=512).generate(spam_words)
    plt.figure(figsize=(10, 8), facecolor='k')
    plt.imshow(spam_wc)
    plt.axis('off')
    plt.show()

    ham = ' '.join(list(messages[messages['v1'] == 'ham']['v2']))
    ham_wc = WordCloud(width = 512,height = 512).generate(spam_words)
    plt.figure(figsize = (10,8),facecolor = 'k')
    plt.imshow(ham_wc)
    plt.axis('off')
    plt.show()