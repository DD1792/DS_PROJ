#CALCULATING COSINE SIMILARITY
from sklearn.metrics.pairwise import cosine_similarity

def get_movie_similarity(embeddings):
    movie_similarity = {}

    for movie1, embedding1 in embeddings.items():
        similarities = []
        for movie2, embedding2 in embeddings.items():
            if movie1 != movie2:
                similarity = cosine_similarity([embedding1], [embedding2])[0][0]
                similarities.append((movie2, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        movie_similarity[movie1] = similarities[:5]  # Top 5 similar movies

    return movie_similarity
