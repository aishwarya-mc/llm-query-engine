from sentence_transformers import SentenceTransformer

# Load model once when this module is imported
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Takes a list of strings and returns their 384-dim embeddings.
    """
    embeddings = model.encode(texts, show_progress_bar=True)
    return [emb.tolist() for emb in embeddings]
