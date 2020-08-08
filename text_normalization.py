import pandas as pd
import numpy as np
import re
import streamlit as st
from num2words import num2words
import string
import unidecode
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.snowball import FrenchStemmer
from nltk.stem import SnowballStemmer

def convert_lower_case(data):
    return np.char.lower(data)
    

def remove_stop_words(data , lenguage):
    if lenguage == "French":
        stop_words = stopwords.words('french')
    if lenguage == "English":
        stop_words = stopwords.words('english')
    if lenguage == "Spanish":
        stop_words = stopwords.words('spanish')

    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text


def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data

def remove_apostrophe(data):
    return np.char.replace(data, "'", "")


def stemming(data ,lenguage):
    if lenguage == "English" : 
         stemmer= PorterStemmer()
    if lenguage == "French" : 
        stemmer = FrenchStemmer()
    if lenguage == "Spanish" : 
        stemmer =  SnowballStemmer('spanish')

    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text

def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text


def remove_accents(data):
    data_str = str(data)
    unaccented_string = unidecode.unidecode(data_str)
    return np.array(unaccented_string)

def preprocess(data,lenguage):
    data = convert_lower_case(data)
    data = remove_accents(data)
    data = remove_punctuation(data) #remove comma seperately
    data = remove_apostrophe(data)
    data = remove_stop_words(data,lenguage)
    data = convert_numbers(data)
    data = stemming(data,lenguage)
    data = remove_punctuation(data)
    data = convert_numbers(data)
    data = stemming(data,lenguage) #needed again as we need to stem the words
    data = remove_punctuation(data) #needed again as num2word is giving few hypens and commas fourty-one
    data = remove_stop_words(data,lenguage) #needed again as num2word is giving stop words 101 - one hundred and one
    return data