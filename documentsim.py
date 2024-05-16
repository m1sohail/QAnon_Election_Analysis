# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 12:36:15 2021

"""

import csv
import pandas as pd
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.tokenize import PunktSentenceTokenizer,RegexpTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from nltk.stem import WordNetLemmatizer
import warnings
from nltk.tokenize import sent_tokenize
import gensim
from string import digits
import inflect

#to ignore deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

data = pd.read_csv(r"C:\Users\Maria\OneDrive - McGill University\Desktop\McGill Notes\Text Analytics\Data\Similarity.csv")

#replacing numbers in a string
data.replace(' \d+', '', regex=True, inplace=True)

#removing special characters from text
spec_chars = ['>>','>','@']

for char in spec_chars:
    data['Quanon'] = data['Quanon'].str.replace(char, ' ')
    
for char in spec_chars:
    data['Trump'] = data['Trump'].str.replace(char, ' ')

# tokenising the data
data["tokenized_Q"] = [word_tokenize(i) for i in data["Quanon"]]
data["tokenized_T"] = [word_tokenize(i) for i in data["Trump"]]

#lowercasing the data
def to_lowercase(words):
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

data['lowered_Q'] = data.apply(lambda row: to_lowercase(row['tokenized_Q']), axis=1)
data['lowered_T'] = data.apply(lambda row: to_lowercase(row['tokenized_T']), axis=1)

#removing stop words
def remove_stopwords(words):
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

data['No_SW_Q'] = data.apply(lambda row: remove_stopwords(row['lowered_Q']), axis=1)
data['No_SW_T'] = data.apply(lambda row: remove_stopwords(row['lowered_T']), axis=1)

def lemmatize_verbs(words):
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


data['normalized_Q'] = data.apply(lambda row: lemmatize_verbs(row['No_SW_Q']), axis=1)
data['normalized_T'] = data.apply(lambda row: lemmatize_verbs(row['No_SW_T']), axis=1)


#data.to_csv(r"C:\Users\Maria\OneDrive - McGill University\Desktop\McGill Notes\Text Analytics\Data\tokenized.csv")

#building bag of words using frequency
vec_words = CountVectorizer()

#detokenizing
data["strg_Q"] = [TreebankWordDetokenizer().detokenize(i) for i in data["No_SW_Q"]]
data["strg_T"] = [TreebankWordDetokenizer().detokenize(i) for i in data["No_SW_T"]]

#data.to_csv(r"C:\Users\Maria\OneDrive - McGill University\Desktop\McGill Notes\Text Analytics\Data\combined.csv")

combined = pd.read_csv(r"C:\Users\Maria\OneDrive - McGill University\Desktop\McGill Notes\Text Analytics\Data\combined_new.csv")

#creating vocab of words
total_features_words1 = vec_words.fit_transform(combined["combined"])
print(total_features_words1.shape)

#Calculating pairwise cosine similarity
feat = pd.DataFrame(total_features_words1.toarray(), columns=vec_words.get_feature_names())
total_features_Q= feat.iloc[1:499,:]
total_features_T= feat.iloc[500:,:]
print(total_features_Q.shape)
print(total_features_T.shape)
similarity=1-pairwise_distances(total_features_Q,total_features_T, metric='cosine')

#Assigning the similarity score to dataframe
similarity=pd.DataFrame(similarity)

#took threshold of greater 0.45
visua = similarity[similarity > 0.45]
sims = visua[visua.notnull()].stack().index


#data.to_csv(r"C:\Users\Maria\OneDrive - McGill University\Desktop\McGill Notes\Text Analytics\Data\data_new.csv")


######################################################################################################################################
#Date Range from March to May 2020

data2 = pd.read_csv(r"C:\Users\Maria\OneDrive - McGill University\Desktop\McGill Notes\Text Analytics\Data\Date1.csv")

#replacing numbers in a string
data2.replace('\d+', '', regex=True, inplace=True)

#removing special characters from text
spec_chars = ['>>','>']

for char in spec_chars:
    data2['Q'] = data2['Q'].str.replace(char, ' ')
    
for char in spec_chars:
    data2['T'] = data2['T'].str.replace(char, ' ')

# tokenising the data
data2["tokenized_Q"] = [word_tokenize(i) for i in data2["Q"]]
data2["tokenized_T"] = [word_tokenize(i) for i in data2["T"]]

def to_lowercase(words):
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

data2['lowered_Q'] = data2.apply(lambda row: to_lowercase(row['tokenized_Q']), axis=1)
data2['lowered_T'] = data2.apply(lambda row: to_lowercase(row['tokenized_T']), axis=1)

def remove_stopwords(words):
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

data2['No_SW_Q'] = data2.apply(lambda row: remove_stopwords(row['lowered_Q']), axis=1)
data2['No_SW_T'] = data2.apply(lambda row: remove_stopwords(row['lowered_T']), axis=1)


def lemmatize_verbs(words):
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


data2['normalized_Q'] = data2.apply(lambda row: lemmatize_verbs(row['No_SW_Q']), axis=1)
data2['normalized_T'] = data2.apply(lambda row: lemmatize_verbs(row['No_SW_T']), axis=1)

#data2.to_csv(r"C:\Users\Maria\OneDrive - McGill University\Desktop\McGill Notes\Text Analytics\Data\tokenized.csv")

#building bag of words using frequency
vec_words2 = CountVectorizer()

#detokenizingtext
data2["strg_Q"] = [TreebankWordDetokenizer().detokenize(i) for i in data2["No_SW_Q"]]
data2["strg_T"] = [TreebankWordDetokenizer().detokenize(i) for i in data2["No_SW_T"]]

#data2.to_csv(r"C:\Users\Maria\OneDrive - McGill University\Desktop\McGill Notes\Text Analytics\Data\combined2.csv")

combined2 = pd.read_csv(r"C:\Users\Maria\OneDrive - McGill University\Desktop\McGill Notes\Text Analytics\Data\new_data.csv")

#creating vocab of words
total_features_words2 = vec_words.fit_transform(combined2["combined"])
print(total_features_words2.shape)

#Calculating pairwise cosine similarity
feat2 = pd.DataFrame(total_features_words2.toarray(), columns=vec_words.get_feature_names())
total_feat_Q= feat2.iloc[1:481,:]
total_feat_T= feat2.iloc[482:,:]
print(total_feat_Q.shape)
print(total_feat_T.shape)
similarity2=1-pairwise_distances(total_feat_Q,total_feat_T, metric='cosine')

#Assigning the similarity score to dataframe
similarity2=pd.DataFrame(similarity2)

#took similarity threshold of greater than 0.40 to get more number of documents for comparison
visua2 = similarity2[similarity2 > 0.40]
sims2 = visua2[visua2.notnull()].stack().index


#data2.to_csv(r"C:\Users\Maria\OneDrive - McGill University\Desktop\McGill Notes\Text Analytics\Data\data2.csv")


######################################################################################################################################
#Date Range from December 2019 to March 2020

data3 = pd.read_csv(r"C:\Users\Maria\OneDrive - McGill University\Desktop\McGill Notes\Text Analytics\Data\Date_Dec_Mar20.csv")

#removing digits and numbers from a string
data3.replace(' \d+', '', regex=True, inplace=True)

#removing special characters
spec_chars = ['>>','>']

for char in spec_chars:
    data3['Q'] = data3['Q'].str.replace(char, ' ')
    
for char in spec_chars:
    data3['T'] = data3['T'].str.replace(char, ' ')

#tokenizing the text
data3["tokenized_Q"] = [word_tokenize(i) for i in data3["Q"]]
data3["tokenized_T"] = [word_tokenize(i) for i in data3["T"]]

def to_lowercase(words):
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

data3['lowered_Q'] = data3.apply(lambda row: to_lowercase(row['tokenized_Q']), axis=1)
data3['lowered_T'] = data3.apply(lambda row: to_lowercase(row['tokenized_T']), axis=1)

def remove_stopwords(words):
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

data3['No_SW_Q'] = data3.apply(lambda row: remove_stopwords(row['lowered_Q']), axis=1)
data3['No_SW_T'] = data3.apply(lambda row: remove_stopwords(row['lowered_T']), axis=1)


def lemmatize_verbs(words):
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


data3['normalized_Q'] = data3.apply(lambda row: lemmatize_verbs(row['No_SW_Q']), axis=1)
data3['normalized_T'] = data3.apply(lambda row: lemmatize_verbs(row['No_SW_T']), axis=1)


#building bag of words using frequency
vec_words3 = CountVectorizer()

#detokenizing the text
data3["strg_Q"] = [TreebankWordDetokenizer().detokenize(i) for i in data3["No_SW_Q"]]
data3["strg_T"] = [TreebankWordDetokenizer().detokenize(i) for i in data3["No_SW_T"]]

#data3.to_csv(r"C:\Users\Maria\OneDrive - McGill University\Desktop\McGill Notes\Text Analytics\Data\combined3.csv")

combined3 = pd.read_csv(r"C:\Users\Maria\OneDrive - McGill University\Desktop\McGill Notes\Text Analytics\Data\new_data3.csv")

#creating vocab of words
total_features_words3 = vec_words.fit_transform(combined3["combined"])
print(total_features_words3.shape)

#Calculating pairwise cosine similarity
feat3 = pd.DataFrame(total_features_words3.toarray(), columns=vec_words.get_feature_names())
total_feat_Q2= feat3.iloc[1:349,:]
total_feat_T2= feat3.iloc[350:,:]
print(total_feat_Q2.shape)
print(total_feat_T2.shape)
similarity3=1-pairwise_distances(total_feat_Q2,total_feat_T2, metric='cosine')

#Assigning the similarity score to dataframe
similarity3=pd.DataFrame(similarity3)

#took similarity threshold of 0.30
visua3 = similarity3[similarity3 > 0.30]
sims3 = visua3[visua3.notnull()].stack().index


#data3.to_csv(r"C:\Users\Maria\OneDrive - McGill University\Desktop\McGill Notes\Text Analytics\Data\data3.csv")


############################################################################################################################
#Nov-March high toxic scores

fd = pd.read_excel(r"C:\Users\Maria\OneDrive - McGill University\Desktop\McGill Notes\Text Analytics\Data\new.xlsx")

#removing special characters
spec_chars = ['>>','>']

for char in spec_chars:
    fd['Quanon'] = fd['Quanon'].str.replace(char, ' ')
    
for char in spec_chars:
    fd['Trump'] = fd['Trump'].str.replace(char, ' ')

#removing digits and numbers from a string
fd["Quanon"].replace(' \d+', '', regex=True, inplace=True)

fd["Trump"].replace(' \d+', '', regex=True, inplace=True)

#removing trailing spaces from the text
fd["Quanon"] = fd["Quanon"].str.strip()
fd["Trump"] = fd["Trump"].str.strip()

#tokenising the text
fd["tokenized_Q"] = [word_tokenize(i) for i in fd["Quanon"]]
fd["tokenized_T"] = [word_tokenize(i) for i in fd["Trump"]]


def to_lowercase(words):
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

fd['lowered_Q'] = fd.apply(lambda row: to_lowercase(row['tokenized_Q']), axis=1)
fd['lowered_T'] = fd.apply(lambda row: to_lowercase(row['tokenized_T']), axis=1)

def remove_stopwords(words):
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

fd['No_SW_Q'] = fd.apply(lambda row: remove_stopwords(row['lowered_Q']), axis=1)
fd['No_SW_T'] = fd.apply(lambda row: remove_stopwords(row['lowered_T']), axis=1)

def lemmatize_verbs(words):
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


fd['normalized_Q'] = fd.apply(lambda row: lemmatize_verbs(row['No_SW_Q']), axis=1)
fd['normalized_T'] = fd.apply(lambda row: lemmatize_verbs(row['No_SW_T']), axis=1)


#building bag of words using frequency
vec_words = CountVectorizer()

#detokenising text
fd["strg_Q"] = [TreebankWordDetokenizer().detokenize(i) for i in fd["normalized_Q"]]
fd["strg_T"] = [TreebankWordDetokenizer().detokenize(i) for i in fd["normalized_T"]]

#fd.to_csv(r"C:\Users\Maria\OneDrive - McGill University\Desktop\McGill Notes\Text Analytics\Data\final_new.csv")

combined4 = pd.read_excel(r"C:\Users\Nupur\OneDrive - McGill University\Desktop\McGill Notes\Text Analytics\Data\mergedd.xlsx")

#creating bag of words
total_features_words4 = vec_words.fit_transform(combined4["combined"])
print(total_features_words4.shape)

#Calculating pairwise cosine similarity
feat4 = pd.DataFrame(total_features_words4.toarray(), columns=vec_words.get_feature_names())
total_feat_Q4= feat4.iloc[1:499,:]
total_feat_T4= feat4.iloc[500:,:]
print(total_feat_Q4.shape)
print(total_feat_T4.shape)
similarity4=1-pairwise_distances(total_feat_Q4,total_feat_T4, metric='cosine')

#Assigning the similarity score to dataframe
similarity4=pd.DataFrame(similarity4)

#taking threshold of greater than 0.40
visua4 = similarity4[similarity4 > 0.40]
sims4 = visua4[visua4.notnull()].stack().index

#fd.to_excel(r"C:\Users\Maria\OneDrive - McGill University\Desktop\McGill Notes\Text Analytics\Data\finalv2.xlsx")

###################################################################################################################################
#analysing sentiment in the similar documents
from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

sent_Q = pd.read_excel(r"C:\Users\Maria\OneDrive - McGill University\Desktop\McGill Notes\Text Analytics\Data\Q_sentiment.xlsx")

sent_Q['compound'] = [analyzer.polarity_scores(x)['compound'] for x in sent_Q['Quanon']]
sent_Q['neg'] = [analyzer.polarity_scores(x)['neg'] for x in sent_Q['Quanon']]
sent_Q['neu'] = [analyzer.polarity_scores(x)['neu'] for x in sent_Q['Quanon']]
sent_Q['pos'] = [analyzer.polarity_scores(x)['pos'] for x in sent_Q['Quanon']]


sent_T = pd.read_excel(r"C:\Users\Maria\OneDrive - McGill University\Desktop\McGill Notes\Text Analytics\Data\T_sentiment.xlsx")

sent_T['compound'] = [analyzer.polarity_scores(x)['compound'] for x in sent_T['Trump']]
sent_T['neg'] = [analyzer.polarity_scores(x)['neg'] for x in sent_T['Trump']]
sent_T['neu'] = [analyzer.polarity_scores(x)['neu'] for x in sent_T['Trump']]
sent_T['pos'] = [analyzer.polarity_scores(x)['pos'] for x in sent_T['Trump']]












