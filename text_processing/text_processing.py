from pydantic import BaseModel
from fastapi import APIRouter, Body
import nltk
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from typing import List  
import string

router = APIRouter()



class Text(BaseModel):
    text: str




def tokneizing(text: str):
    return text.split()
    

@router.post("/text_processing/tokenize")
def tokenizer(text: Text = Body(...)):  
    return tokneizing(text.text)






def Remove_punctuation(text: str):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stopwords(text: str):
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def stemmeing(text: str):
    stemmer = PorterStemmer()
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text



@router.post("/text_processing/preprocessor")
def preprocessor(text: Text = Body(...)):
    text = text.text.lower()
    text = Remove_punctuation(text)
    text = remove_stopwords(text)
    text = stemmeing(text)
    return text


def correct_sentence_spelling(sentence: str):
    spell = SpellChecker()
    tokens = word_tokenize(sentence)
    misspelled = spell.unknown(tokens)
    for i, token in enumerate(tokens):
        if token in misspelled:
            corrected = spell.correction(token)
            if corrected is not None:
                tokens[i] = corrected
    return ' '.join(tokens)

@router.post("/text_processing/spell")
def spell(text: Text = Body(...)):
    return correct_sentence_spelling(text.text)


    