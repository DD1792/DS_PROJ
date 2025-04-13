from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
import os

# Force fresh download and get local path
local_model_path = snapshot_download("sentence-transformers/all-MiniLM-L6-v2")

# Load from local path
model = SentenceTransformer(local_model_path)

sentences = ["This is an example sentence.", "This is another example."]
embeddings = model.encode(sentences)
print(embeddings)
