from fastapi import APIRouter, Body, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
import random
import numpy as np
import pandas as pd
import requests
import json
import pickle
import io
import base64
import ir_datasets
import os


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




def store_index(dataset: str, value: int):
    if value == 1:
        df = pd.read_csv(f'./search/datasets/{dataset}/{dataset}_docs.csv')
        corpus = base64.b64encode(pickle.dumps(df['text'].tolist())).decode('utf-8')
    else:
        df = pd.read_csv(f'./search/datasets/{dataset}/{dataset}_queries.csv')
        corpus = base64.b64encode(pickle.dumps(df['text'].tolist())).decode('utf-8')

    url = f"http://127.0.0.1:8000/indexing/build/{dataset}"
    data = {"corpus": corpus}
    headers = {'Content-type': 'application/json'}
    response = requests.post(url, json=data, headers=headers)
        
    speras_matrix_data = base64.b64decode(response.json()["speras_matrix"])
    vectorizer_data = base64.b64decode(response.json()["vectorizer"])


    # Reconstruct the speras_matrix and vectorizer
    data, indices, indptr, shape = pickle.loads(speras_matrix_data)
    speras_matrix = csr_matrix((data, indices, indptr), shape=shape)
    vectorizer = pickle.loads(vectorizer_data)
    vectorizer.preprocessor = preprocessor
    vectorizer.tokenizer = tokenizer


    # Save the TF-IDF matrix to a binary file
    

    if value == 1:
        with open(f'./search/{dataset}_speras_matrix.pkl', 'wb') as f:
            pickle.dump(speras_matrix, f)

        with open(f'./search/{dataset}_vectorizer.pkl','wb') as f:
            pickle.dump(vectorizer, f)
    else:
        with open(f'./search/{dataset}_speras_matrix_queries.pkl', 'wb') as f:
            pickle.dump(speras_matrix, f)

        with open(f'./search/{dataset}_vectorizer_queries.pkl','wb') as f:
            pickle.dump(vectorizer, f)

    
    return # corrected the typo here

def make_cluster(dataset: str):
    with open(f'./search/{dataset}_speras_matrix.pkl', 'rb') as f:
            speras_matrix = pickle.load(f)


    k = 25  # or however many clusters you want
    model = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
    model.fit(speras_matrix)


    with open(f'./search/{dataset}_model.pkl', 'wb') as f:
        pickle.dump(model, f)


    
    labels = model.predict(speras_matrix)
    cluster_lists = {}
    for i in range(0, len(labels)):
        if labels[i] not in cluster_lists:
            cluster_lists[labels[i]] = []
        cluster_lists[labels[i]].append(i)

    with open(f'./search/{dataset}_cluster_lists.pkl', 'wb') as f:
        pickle.dump(cluster_lists, f)
    

@router.get("/search/{dataset}")
def search(dataset: str,query: str):
    if not os.path.exists(f'./search/{dataset}_speras_matrix.pkl') or not os.path.exists(f'./search/{dataset}_vectorizer.pkl'):
        store_index(dataset,1)
    if not os.path.exists(f'./search/{dataset}_model.pkl') or not os.path.exists(f'./search/{dataset}_cluster_lists.pkl'):
        make_cluster(dataset)

    with open(f'./search/{dataset}_speras_matrix.pkl', 'rb') as f:
        speras_matrix = pickle.load(f)
    with open(f'./search/{dataset}_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open(f'./search/{dataset}_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(f'./search/{dataset}_cluster_lists.pkl', 'rb') as f:
        cluster_lists = pickle.load(f)

    labels = model.predict(speras_matrix)
        

    vectorizer.preprocessor = preprocessor
    vectorizer.tokenizer = tokenizer


    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, speras_matrix).flatten()
    best_match_idx = np.argsort(cosine_similarities)[-10:]

    clusters = [0] * 25
    for i in best_match_idx:
        clusters[labels[i]] += 1
    max_value = max(clusters)
    max_cluster = clusters.index(max_value)

    df = pd.read_csv(f'./search/datasets/{dataset}/{dataset}_docs.csv')
    corpus = df['text'].tolist()
    doc_id = df['doc_id'].tolist()
    retrieved = []
    index = []
    for i in best_match_idx:                      
        retrieved.append(corpus[i])
        index.append(doc_id[i])
    retrieved.reverse()
    index.reverse()

    elements = random.sample(cluster_lists[max_cluster], 5)
    cluster_doc = []
    for i in elements:
        cluster_doc.append(corpus[i])
    


    return {"retrieved": retrieved, "ID": index, "cluster_doc": cluster_doc}



# @router.get("/search/{dataset}")
# def search(dataset: str,query: str):
#     if not os.path.exists(f'./search/{dataset}_speras_matrix.pkl') or not os.path.exists(f'./search/{dataset}_vectorizer.pkl'):
#         store_index(dataset,1)
#     # if not os.path.exists(f'./search/{dataset}_model.pkl') or not os.path.exists(f'./serach/{dataset}_cluster_lists.pkl'):
#     #     make_cluster(dataset)

#     with open(f'./search/{dataset}_speras_matrix.pkl', 'rb') as f:
#         speras_matrix = pickle.load(f)
#     with open(f'./search/{dataset}_vectorizer.pkl', 'rb') as f:
#         vectorizer = pickle.load(f)
#     # with open(f'./search/{dataset}_model.pkl', 'rb') as f:
#     #     model = pickle.load(f)
#     # with open(f'./serach/{dataset}_cluster_lists.pkl', 'rb') as f:
#     #     cluster_lists = pickle.load(f)

#     # labels = model.predict(speras_matrix)
        

#     vectorizer.preprocessor = preprocessor
#     vectorizer.tokenizer = tokenizer


#     query_vector = vectorizer.transform([query])
#     cosine_similarities = cosine_similarity(query_vector, speras_matrix).flatten()
#     best_match_idx = np.argsort(cosine_similarities)[-10:]

#     # clusters = [0] * 25
#     # for i in best_match_idx:
#     #     clusters[labels[i]] += 1
#     # max_cluster = p.Series(clusters).idxmax()

#     df = pd.read_csv(f'./search/datasets/{dataset}/{dataset}_docs.csv')
#     corpus = df['text'].tolist()
#     doc_id = df['doc_id'].tolist()
#     retrieved = []
#     index = []
#     for i in best_match_idx:
#         retrieved.append(corpus[i])
#         index.append(doc_id[i])
#     retrieved.reverse()
#     index.reverse()


#     return {"retrieved": retrieved, "ID": index}


@router.get("/search/suggest/{dataset}")
def suggest(dataset: str, query: str):
    if not os.path.exists(f'./search/{dataset}_speras_matrix_queries.pkl') or not os.path.exists(f'./search/{dataset}_vectorizer_queries.pkl'):
        store_index(dataset,2)

    with open(f'./search/{dataset}_speras_matrix_queries.pkl', 'rb') as f:
        speras_matrix = pickle.load(f)
    with open(f'./search/{dataset}_vectorizer_queries.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    vectorizer.preprocessor = preprocessor
    vectorizer.tokenizer = tokenizer


    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, speras_matrix).flatten()
    best_match_idx = np.argsort(cosine_similarities)[-10:]


    df = pd.read_csv(f'./search/datasets/{dataset}/{dataset}_docs.csv')
    corpus = df['text'].tolist()
    doc_id = df['doc_id'].tolist()
    retrieved = []
    index = []
    for i in best_match_idx:
        retrieved.append(corpus[i])
        index.append(doc_id[i])
    retrieved.reverse()
    index.reverse()


    return {"retrieved": retrieved, "ID": index}
