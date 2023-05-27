###########################################################
### step  1 - loading dataset, preprocessing dataset
# 
# Add description for this code.
###########################################################

# libraries for this project
import pandas as pd
import sys
import itertools
from datetime import datetime
from collections import Counter
import numpy as np
import gc # Garbage Collector interface
import time
from icecream import ic # Never use print() to debug again, ic a high level for debug
import snoop # print the lines of code being executed in a function/ great feature very useful to debug :)

import toolkit as tool # see the file toolkit.py for more info
import feather
import requests as re
import re # for regex
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
#nltk.download('all')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score
import pickle
from transformers import BertTokenizer, BertForSequenceClassification
import string
from collections import Counter

# Get start time 
start_time = time.time()

# Here we are going to implement some functions

print('**********************************************')
print('Cleaning the dataset...')
print('**********************************************')

now = datetime.now()
 
print("date..............:", now)

#### https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

# loading the dataset
df_reviews = pd.read_feather('data-feather/Reviews.ftr') # full dataset

# dataset shape
print(f"Shape dataset Full:.........observations/rows: {df_reviews.shape[0]} and columns: {df_reviews.shape[1]}")

# reduce memory usage
df_reviews = tool.reduce_mem_usage(df_reviews)

################################################ FIXME - remove
######## parte do dataset
df_reviews_copy = df_reviews.copy()
df_reviews_sample = df_reviews_copy.sample(28000) # FIXME remover no final

# release memory RAM - dataframe
tool.release_memory(df_reviews_copy)
tool.release_memory(df_reviews)
#df_reviews = df_reviews_sample[['Score', 'Text']] 
df_reviews = df_reviews_sample
tool.release_memory(df_reviews_sample)
############################################

# Lista de valores faltantes
df_reviews.isna().sum()

# dataset - tamanho
df_reviews.shape

# retirar linha com valores faltantes
df_reviews = df_reviews.dropna()

# checar numero de linha faltantes
df_reviews.isna().sum()

# retirar os neutros.
df_reviews = df_reviews[df_reviews['Score'] != 3]

################### cleaning ################### 
def clean(text):
    cleaned = re.compile(r'<.*?>') # remove tags html
    return re.sub(cleaned,'',text)

# remover caracteres especiais
def is_special(text):
    rem = ''
    for i in text:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
    return rem

# Converter - lowercase
def to_lower(text):
    return text.lower()

nltk.download('stopwords')
nltk.download('punkt')

def rem_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [w for w in words if w not in stop_words]

#No review tenha palavras de outro idioma
def stem_txt(text):
    ss = SnowballStemmer('english')
    return " ".join([ss.stem(w) for w in text])

#################################################

print('Applying the functions - cleaning...')
# applying the functions to clean and adjust the dataset
df_reviews['Text'] = df_reviews['Text'].apply(clean)
print('#1 Done, Next function...')
df_reviews['Text'] = df_reviews['Text'].apply(is_special)
print('#2 Done, Next function...')
df_reviews['Text'] = df_reviews['Text'].apply(to_lower)
print('#3 Done, Next function...')
df_reviews['Text'] = df_reviews['Text'].apply(rem_stopwords)
print('#4 Done, Next function...')
df_reviews['Text'] = df_reviews['Text'].apply(stem_txt)
print('Next...')

print('create preprocess_text function...')

# create preprocess_text function
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]


    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text


df_reviews['Text'] = df_reviews['Text'].apply(preprocess_text)

print(f"Dataset Full after clean:.....rows: {df_reviews.shape[0]} and columns: {df_reviews.shape[1]}")
print("saving the file format feather...")

# this is important to do before save in feather format.
df_reviews = df_reviews.reset_index(drop=True)

# saving in the feather format
df_reviews.to_feather('data-feather/cleaned.ftr')

# time of execution in minutes
time_exec_min = round( (time.time() - start_time)/60, 4)

print(f'time of execution (preprocessing): {time_exec_min} minutes')
print("the cleaning is done.")
print("The next step is to do the feature engineering.")
print("All Done.")