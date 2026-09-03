import numpy as np
from functools import lru_cache
from num2words import num2words
import snowballstemmer
from stop_words import get_stop_words
import unidecode

def convert_lower_case(data):
    return np.char.lower(data)
    

LANGUAGES = {
    "English": ("english", "english"),
    "French": ("french", "french"),
    "Spanish": ("spanish", "spanish"),
}


@lru_cache(maxsize=len(LANGUAGES))
def _stop_words(lenguage):
    try:
        language, _ = LANGUAGES[lenguage]
    except KeyError as error:
        raise ValueError(f"Unsupported language: {lenguage!r}") from error
    return frozenset(get_stop_words(language))


def remove_stop_words(data, lenguage):
    stop_words = _stop_words(lenguage)

    words = str(data).split()
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text


def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data

def remove_apostrophe(data):
    return np.char.replace(data, "'", "")


def stemming(data, lenguage):
    try:
        _, stemmer_language = LANGUAGES[lenguage]
    except KeyError as error:
        raise ValueError(f"Unsupported language: {lenguage!r}") from error
    stemmer = snowballstemmer.stemmer(stemmer_language)

    tokens = str(data).split()
    return " " + " ".join(stemmer.stemWords(tokens)) if tokens else ""

def convert_numbers(data):
    tokens = str(data).split()
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except (TypeError, ValueError):
            pass
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
