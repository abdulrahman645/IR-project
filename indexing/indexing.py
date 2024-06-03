from fastapi import APIRouter, Body, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from typing import Tuple
from io import BytesIO
import numpy as np
import pandas as pd

import requests
import json


from storge.storge import load_binary_file, save_binary_file, load_df

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



def preprocessor(text: str):
    url = "http://127.0.0.1:8000/text_processing/preprocessor"
    data = {"text": text}
    headers = {'Content-type': 'application/json'}
    response = requests.post(url, json=data, headers=headers)
    return response.json()



@router.post("/indexing/build/{dataset}/{var_type}")
def build_index(dataset: str,var_type: str):
    df = load_df(dataset, var_type)
    corpus = df['text'].tolist()

    vectorizer = TfidfVectorizer(preprocessor=preprocessor, tokenizer=tokenizer)
    speras_matrix = vectorizer.fit_transform(corpus)

    save_binary_file(dataset,f'speras_matrix_{var_type}',speras_matrix)
    save_binary_file(dataset,f'vectorizer_{var_type}',vectorizer)
    # Return the encoded data
    return {"status": "successs"}


@router.post("/clustring/{dataset}")
def make_cluster(dataset: str):

    speras_matrix = load_binary_file(dataset,'speras_matrix_docs')

    k = 25  # or however many clusters you want
    model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
    model.fit(speras_matrix)


    save_binary_file(dataset,'model',model)
    
    labels = model.predict(speras_matrix)
    cluster_lists = {}
    for i in range(0, len(labels)):
        if labels[i] not in cluster_lists:
            cluster_lists[labels[i]] = []
        cluster_lists[labels[i]].append(i)

    save_binary_file(dataset,'cluster_lists',cluster_lists)
    return {"status": "successs"}