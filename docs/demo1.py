import torch
import faiss
from activeft.sift import Retriever

# Before Test-Time
index = faiss.IndexFlatIP(dataset_embeddings.size(1))
index.add(dataset_embeddings)
retriever = Retriever(index)

# At Test-Time, given query
indices = retriever.search(query_embeddings, N=10)
model.step(dataset[indices])
