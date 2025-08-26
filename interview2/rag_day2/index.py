import faiss, numpy as np

def build_flat_index(vecs: np.ndarray):
    d = vecs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(vecs)
    return index

def search(index, qvec: np.ndarray, k: int = 5):
    D, I  = index.search(qvec, k)
    return D[0].tolist(), I[0].tolist()