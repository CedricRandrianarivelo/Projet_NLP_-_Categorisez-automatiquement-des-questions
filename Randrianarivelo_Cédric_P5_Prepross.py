# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:44:37 2023

@author: cedri
"""

import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd 
##from nlpk_module import normalize_corpus, remove_stopwords,tok
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
import string
import time

import lxml
import html5lib
from bs4 import BeautifulSoup
pd.set_option('display.max_colwidth', 100)
import re
import string
import nltk
import warnings
##import spacy
warnings.filterwarnings("ignore")
import tensorflow as tf

from wordcloud import WordCloud, STOPWORDS
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

import nltk

nltk.download('omw-1.4')
nltk.download('punkt')

nltk.download('stopwords')
stopword = nltk.corpus.stopwords.words('english')
stopword.extend(re.split('\W+', "https"))
numbers = np.arange(9999)
number_add = ["01","02","03","04","05","06","07","08","09"]
for i in numbers :
    stopword.extend(re.split('\W+', str(i)))

for i in number_add :
    stopword.extend(re.split('\W+', str(i)))
    
nltk.download('wordnet')
wn = nltk.WordNetLemmatizer()

def creation_dataframe(data):
    
    df = pd.DataFrame(data)
    
    return df

def clean_html(text):
    """
    Remove HTML from a text.
    
    Args:
        text(String): Row text with html 
             
    Returns:
        cleaned String
    """
    import lxml
    import html5lib
    from bs4 import BeautifulSoup
 
    soup = BeautifulSoup(text, "html5lib")

    for sent in soup(['style', 'script']):
            sent.decompose()
   
        
    return ' '.join(soup.stripped_strings)

def apply_html (df):
    
    colonne_str = list(df.columns)
    
    for col in colonne_str :
        df [col] = [clean_html(text) for text in df [col]]
    
    return df

def text_cleaning(text):
    """
    Remove figures, punctuation, words shorter than two letters (excepted C or R) in a lowered text. 
    
    Args:
        text(String): Row text to clean
        
    Returns:
       res(string): Cleaned text
    """
    import re
    
    pattern = re.compile(r'[^\w]|[\d_]')
    
    try: 
        res = re.sub(pattern," ", text).lower()
    except TypeError:
        return text
    
    res = res.split(" ")
    res = list(filter(lambda x: len(x)>3 , res))
    res = " ".join(res)
    return res

def apply_data_cleaning(df) :
    
    colonne_str = list(df.columns)

    for col in colonne_str :
        df [col] = [text_cleaning(text) for text in  df [col]]
        
    
    return df

def tokenize(text):
    """
    Tokenize words of a text.
    
    Args:
    
        text(String): Row text
        
    Returns
    
        res(list): Tokenized string.
    """
    
    try:
        res = word_tokenize(text, language='english')
    except TypeError:
        return text
    
    res = [token for token in res if token not in stopword]
    return res

def apply_tokenize(df) :
    
    colonne_str = list(df.columns)

    for col in colonne_str :
        df[col] = [tokenize(text) for text in df[col].to_list()]
        
    
    return df

def size_word(tokenize_list):
    
    text = [word for word in tokenize_list if (len(word)>2 and len(word) < 3000)]
    clean_text = text.copy()
    
    return clean_text

def lemmatize(tokenize_word):
    text = [wn.lemmatize(word) for word in tokenize_word]
    return text

def apply_lem(df) : 
    
    colonne_str = list(df.columns)
        
    for col in colonne_str :
        df[col] = df[col].apply(lambda x : lemmatize(x))
        
    return df

def token_to_normal(text):
    """
    Token to Normal
    """
    new_text = ""
    
    for word in text :
        new_text = new_text + " " + word

    return new_text

def apply_t2N (df) : 
    
    df["Body T2N"] = df["Body"].apply(lambda x : token_to_normal(x))
    df["Title T2N"] = df["Title"].apply(lambda x : token_to_normal(x))
        
    return df

def data_load_reshape (data):
    

    data["Title + Body"] = data["Title T2N"] + data["Body T2N"] 
    data["Title + Body"] = data["Title + Body"].str.split()
    data["Title + Body T2N"] = data["Title T2N"].astype(str) + " " + data["Body T2N"].astype(str) 
    docs = data["Title T2N"].astype(str) + " " +data["Body T2N"].astype(str)
    
    return data

model_path =r"C:\Users\cedri\OneDrive\Documents\Cours\OpenClassroom\Cours\Projet_5"

import pickle
with open("Model_Tf_Id.pkl", "rb") as f:
    TF_ID = pickle.load(f)

def vector_to_TF_id (data) :
    
    data_tf_id = TF_ID.transform(data["Title + Body T2N"]).toarray()
    feature_names = TF_ID.get_feature_names_out()
    df_finale = pd.DataFrame(data = data_tf_id, columns=feature_names)
    
    return df_finale



def main (data):
    
    data = creation_dataframe(data)
    data = apply_html(data)
    data = apply_data_cleaning(data)
    data = apply_tokenize(data)
    data = data.apply(lambda text : size_word(text))
    data = apply_lem(data)
    data = apply_t2N (data)
    data = data_load_reshape(data)
    data = vector_to_TF_id (data)
    
    return data
    





##vectorizer 