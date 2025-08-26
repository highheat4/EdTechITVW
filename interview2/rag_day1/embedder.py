from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    def __init__(self, name="all-MiniLM-L6-v2"):
       self.model = SentenceTransformer(name)
    
    def encode(self, texts):
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(embeddings, dtype="float32")