from sentence_transformers import SentenceTransformer
import faiss
import re
import numpy as np

# chunkers
def chunk_stride(text, size=200, stride=50):
    s = 0
    chunks = []
    while s < len(text):
        toappend = text[s:s+size].strip()
        if toappend:
            chunks.append(toappend)
        s += stride
        
    return chunks

def chunk_semantic(text):
    s = 0
    return re.split('[.?!]', text)

chunks = chunk_semantic("Hi. My name is Ayan. What is your name? I am from India. I am feeling great! There is a mountain next to me. We can go painting some time.")
print(chunks)

# Embedding
class Embedder():
    def __init__(self, name = "all-MiniLM-L6-V2"):
        self.model = SentenceTransformer(name)
    
    def encode(self, texts):
        output = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(output)
    
    
# indexing
def build_index(vecs, name="flat"):
    d = vecs.shape[1]
    if name == "flat":
        index = faiss.IndexFlatIP(d)
    elif name == "hnsw":
        index = faiss.IndexHNSW(d, 32)
    elif name == "ivf":
        index = faiss.IndexIVFFlat(d)
    
    return index

def search(query_vec, index, top_k = 2):
    D, I = index.search(query_vec, top_k)
    return D[0], I[0]


def evaluate():
    pass