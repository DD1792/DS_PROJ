import praw
import pandas as pd
import re
from collections import defaultdict
from sentiment import get_sentiment
from embeddings import generate_embeddings
from recommedations import get_movie_similarity

# ----------------------------------------------
# STEP 1: Load Known Movie List
# ----------------------------------------------
with open("movies.txt", "r") as f:
    known_movies = [line.strip().lower() for line in f.readlines()]

# ----------------------------------------------
# STEP 2: Connect to Reddit
# ----------------------------------------------
reddit = praw.Reddit(
    client_id='eQeAeUfhhVuMrzhtlPYRLg',
    client_secret='ELQAhGWhqRaa3m5xtj_iQ9Kaly1m-g',
    user_agent='RECCSYSTEM_RAHH')

# ----------------------------------------------
# STEP 3: Scrape Reddit Posts
# ----------------------------------------------
subreddits = ['movies', 'moviesuggestions']
posts = []

for sub in subreddits:
    subreddit = reddit.subreddit(sub)
    for post in subreddit.hot(limit=300):  # Get hot posts (max 300)
        posts.append({
            'title': post.title,
            'selftext': post.selftext,
            'comments': [comment.body for comment in post.comments if hasattr(comment, 'body')]
        })

# ----------------------------------------------
# STEP 4: Extract Movie Mentions
# ----------------------------------------------
def clean_text(text):
    """Cleans the input text by removing URLs and special characters."""
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphabet characters
    return text.lower()

def extract_movies(text, known_movies):
    """Extracts movie names from the given text."""
    found = []
    for title in known_movies:
        if title in text:
            found.append(title)
    return list(set(found))  # Remove duplicates

movie_discussions = defaultdict(str)

for post in posts:
    full_text = clean_text(post['title'] + ' ' + post['selftext'] + ' ' + ' '.join(post['comments']))
    movies_in_post = extract_movies(full_text, known_movies)
    for movie in movies_in_post:
        movie_discussions[movie] += full_text  # Concatenate the text mentioning the movie

# ----------------------------------------------
# STEP 5: Sentiment Analysis and Embeddings
# ----------------------------------------------
# Apply sentiment analysis to each movie discussion
sentiment_scores = {movie: get_sentiment(discussion) for movie, discussion in movie_discussions.items()}

# Generate embeddings for each movie discussion text
embeddings = {movie: generate_embeddings([discussion])[0] for movie, discussion in movie_discussions.items()}

# Calculate the cosine similarity between all movie discussions
movie_similarity = get_movie_similarity(embeddings)

# ----------------------------------------------
# STEP 6: Recommendation Function
# ----------------------------------------------
def recommend_movie(movie_name):
    """Recommend movies similar to the given movie name."""
    movie_name = movie_name.lower()
    if movie_name in movie_similarity:
        recommendations = movie_similarity[movie_name]
        print(f"\nRecommendations for '{movie_name.title()}':")
        for movie, score in recommendations:
            print(f"- {movie.title()} (Similarity: {score:.3f})")
    else:
        print(f"Sorry, we couldn't find '{movie_name.title()}' in the database.")

# ----------------------------------------------
# STEP 7: User Input
# ----------------------------------------------
def get_user_input():
    """Prompt the user to input a movie name for recommendations."""
    while True:
        movie_name = input("\nEnter a movie name for recommendations (or type 'exit' to quit): ").strip()
        if movie_name.lower() == 'exit':
            print("Goodbye!")
            break
        else:
            recommend_movie(movie_name)

# Run the interactive recommendation system
get_user_input()
