from fastapi import APIRouter, Body, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple
from io import BytesIO
import numpy as np
import pandas as pd

import requests
import json
import pickle
import os
import base64




import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string


router = APIRouter()

class Text(BaseModel):
    text: str

class Corpus(BaseModel):
    corpus: str


def tokenizer(text: str):
    url = "http://127.0.0.1:8000/text_processing/tokenize"
    data = {"text": text}
    headers = {'Content-type': 'application/json'}
    response = requests.post(url, json=data, headers=headers)
    return response.json()

# def tokneizing(text: str):
#     return text.split()
    

# def tokenizer(text: str):
#     try:  
#         return tokneizing(text)
#     except:
#         print(text)
#         print('-----------------')


def preprocessor(text: str):
    url = "http://127.0.0.1:8000/text_processing/preprocessor"
    data = {"text": text}
    headers = {'Content-type': 'application/json'}
    response = requests.post(url, json=data, headers=headers)
    return response.json()

# def Remove_punctuation(text: str):
#     return text.translate(str.maketrans('', '', string.punctuation))

# def remove_stopwords(text: str):
#     stop_words = set(stopwords.words('english'))
#     text = ' '.join([word for word in text.split() if word not in stop_words])
#     return text

# def stemmeing(text: str):
#     stemmer = PorterStemmer()
#     text = ' '.join([stemmer.stem(word) for word in text.split()])
#     return text


# def preprocessor(text: str):
#     try :
#         test = text
#         text = text.lower()
#         text =  Remove_punctuation(text)
#         text = remove_stopwords(text)
#         text = stemmeing(text)
#         return text
#     except: 
#         print(test)
#         print('-----------------')


@router.post("/indexing/build/{dataset}")
def build_index(dataset: str, corpus: Corpus = Body(...)):
    df = pd.read_csv(f'./search/datasets/{dataset}/{dataset}_docs.csv')
    corpus = df['text'].tolist()

    vectorizer = TfidfVectorizer(preprocessor=preprocessor, tokenizer=tokenizer)
    # corpus.corpus = base64.b64decode(corpus.corpus)
    # corpus.corpus = pickle.loads(corpus.corpus)
    # speras_matrix = vectorizer.fit_transform(corpus.corpus)
    speras_matrix = vectorizer.fit_transform(corpus)

    # Convert the sparse matrix to a tuple of (data, indices, indptr, shape)
    data = speras_matrix.data
    indices = speras_matrix.indices
    indptr = speras_matrix.indptr
    shape = speras_matrix.shape

    # Serialize the matrix data and the vectorizer
    speras_matrix_data = pickle.dumps((data, indices, indptr, shape))
    vectorizer_data = pickle.dumps(vectorizer)

     # Encode the serialized data
    speras_matrix_encoded = base64.b64encode(speras_matrix_data).decode('utf-8')
    vectorizer_encoded = base64.b64encode(vectorizer_data).decode('utf-8')

    with open(f'./search/{dataset}_speras_matrix.pkl', 'wb') as f:
        pickle.dump(speras_matrix, f)

    with open(f'./search/{dataset}_vectorizer.pkl','wb') as f:
        pickle.dump(vectorizer, f)
    

    # Return the encoded data
    return {"speras_matrix": speras_matrix_encoded, "vectorizer": vectorizer_encoded}

