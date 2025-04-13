#GENERATING SBERT EMBEDDINGS FOR TEXT
from sentence_transformers import SentenceTransformer

# Initialize the Sentence-BERT model
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

def generate_embeddings(texts):
    embeddings = model.encode(texts)
    return embeddings