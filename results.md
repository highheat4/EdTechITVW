| Chunker(config) | Embedder | k | Index | p@k | lat ms p50 | docs | chunks |
|---|---|---|---|---|---|---|---|
| fixed(size=600,overlap=100) | all-MiniLM-L6-v2 | 3 | FLAT | 0.600 | 0.0 | 11 | 18 |
| fixed(size=600,overlap=100) | all-MiniLM-L6-v2 | 5 | FLAT | 0.600 | 0.0 | 11 | 18 |
| fixed(size=600,overlap=100) | all-MiniLM-L6-v2 | 8 | FLAT | 0.600 | 0.0 | 11 | 18 |
| fixed(size=600,overlap=100) | gte-base | 3 | FLAT | 0.600 | 0.1 | 11 | 18 |
| fixed(size=600,overlap=100) | gte-base | 5 | FLAT | 0.600 | 0.1 | 11 | 18 |
| fixed(size=600,overlap=100) | gte-base | 8 | FLAT | 0.600 | 0.1 | 11 | 18 |
| sliding(size=300,stride=200) | all-MiniLM-L6-v2 | 3 | FLAT | 0.600 | 0.0 | 11 | 37 |
| sliding(size=300,stride=200) | all-MiniLM-L6-v2 | 5 | FLAT | 0.600 | 0.1 | 11 | 37 |
| sliding(size=300,stride=200) | all-MiniLM-L6-v2 | 8 | FLAT | 0.600 | 0.0 | 11 | 37 |
| sliding(size=300,stride=200) | gte-base | 3 | FLAT | 0.600 | 0.1 | 11 | 37 |
| sliding(size=300,stride=200) | gte-base | 5 | FLAT | 0.600 | 0.1 | 11 | 37 |
| sliding(size=300,stride=200) | gte-base | 8 | FLAT | 0.600 | 0.1 | 11 | 37 |
| semantic(max_len=500) | all-MiniLM-L6-v2 | 3 | FLAT | 0.600 | 0.0 | 11 | 21 |
| semantic(max_len=500) | all-MiniLM-L6-v2 | 5 | FLAT | 0.600 | 0.0 | 11 | 21 |
| semantic(max_len=500) | all-MiniLM-L6-v2 | 8 | FLAT | 0.600 | 0.0 | 11 | 21 |
| semantic(max_len=500) | gte-base | 3 | FLAT | 0.560 | 0.1 | 11 | 21 |
| semantic(max_len=500) | gte-base | 5 | FLAT | 0.600 | 0.1 | 11 | 21 |
| semantic(max_len=500) | gte-base | 8 | FLAT | 0.600 | 0.1 | 11 | 21 |