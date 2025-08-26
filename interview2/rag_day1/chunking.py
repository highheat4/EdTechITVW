import os
import re

def chunk_fixed(text, size = 600, overlap = 100):
    s = 0
    while s < max(1, len(text)):
        yield text[s:s+size]
        if s + size >= len(text): break
        s += size - overlap
        
def chunk_sliding(text, size=300, stride=200):
    s = 0
    while s < max(1, len(text)):
        yield text[s:s + size]
        if s + size >= len(text): break
        s += stride

def chunk_semantic(text, max_len=500):
    sol = re.split(r'(?<=[.!?])\s+|\n+', text)
    for idx in range(len(sol)):
        if len(sol[idx]) > max_len:
            for chunk in chunk_fixed(sol[idx], max_len, 0):
                sol.append(chunk)
    for item in sol:
        yield item

