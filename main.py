# SVM and Random Forest
# Team Project

import numpy as np
import pandas as pd
import re


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
print(df.head(5))

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
    token_3.append(re.sub('[^0-9]+', '', d.lower()))
    token_4.append(len(d))
    token_5.append(char_flag(len(d)))
    token_6.append(re.sub(r'[^://@]', '', d.lower()))


data = [np.array([token_1[i], token_2[i], token_3[i], token_4[i], token_5[i], token_6[i]]) for i in range(len(df_data))]

