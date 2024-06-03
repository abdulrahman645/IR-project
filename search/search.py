from fastapi import APIRouter, Body, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

import random
import numpy as np
import pandas as pd
import requests
import json

from storge.storge import load_binary_file, save_binary_file, load_df

router = APIRouter()

class Text(BaseModel):
    text: str

class Corpus(BaseModel):
    corpus: list[str]

class Query(BaseModel):
    query: str


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


def spell(text: str):
    url = "http://127.0.0.1:8000/text_processing/spell"
    data = {"text": text}
    headers = {'Content-type': 'application/json'}
    response = requests.post(url, json=data, headers=headers)
    return response.json()


    

@router.get("/search/{dataset}")
def search(dataset: str,query: str):

    speras_matrix = load_binary_file(dataset,'speras_matrix_docs')
    vectorizer = load_binary_file(dataset,'vectorizer_docs')
        
    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, speras_matrix).flatten()

    similar_indices = np.argsort(cosine_similarities)

    matching_indices = []

    for idx in similar_indices[::-1]:
        similarity_score = cosine_similarities[idx]
        
        similarity_threshold = 0.5
        if similarity_score >= similarity_threshold:
            matching_indices.append(idx)


    df = load_df(dataset,'docs')
    corpus = df['text'].tolist()
    doc_id = df['doc_id'].tolist()
    retrieved = []
    index = []
    for i in range(min(len(matching_indices), 30)):
        retrieved.append(corpus[matching_indices[i]])
        index.append(doc_id[matching_indices[i]])



    model = load_binary_file(dataset,'model')
    cluster_lists = load_binary_file(dataset,'cluster_lists')

    labels = model.predict(speras_matrix)

    clusters = [0] * 25
    for i in range(min(len(matching_indices), 30)):
        clusters[labels[matching_indices[i]]] += 1
    max_value = max(clusters)
    max_cluster = clusters.index(max_value)

    elements = random.sample(cluster_lists[max_cluster], 5)
    cluster_doc = []
    for i in elements:
        cluster_doc.append(corpus[i])
    


    return {"retrieved": retrieved,"ID": index, "cluster_doc": cluster_doc}




@router.get("/search/realtime_suggestions/{dataset}")
def realtime_suggestions(dataset: str, query: str):

    query = spell(query)

    speras_matrix = load_binary_file(dataset,'speras_matrix_queries')
    vectorizer = load_binary_file(dataset,'vectorizer_queries')



    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, speras_matrix).flatten()


    similar_indices = np.argsort(cosine_similarities)

    matching_indices = []

    for idx in similar_indices[::-1]:
        similarity_score = cosine_similarities[idx]
        similarity_threshold = 0.5  
        if similarity_score >= similarity_threshold:
            matching_indices.append(idx)



    df = load_df(dataset,'queries')
    corpus = df['text'].tolist()
    retrieved = []
    for i in matching_indices:
        retrieved.append(corpus[i])

    return {"suggestions": retrieved}