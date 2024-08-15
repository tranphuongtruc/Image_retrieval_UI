import faiss
import numpy as np


def build_faiss_index(vectors):
    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.array(vectors))
    return index


def search_index(index, query_vector, k=5):
    distances, indices = index.search(query_vector, k)
    return indices, distances
