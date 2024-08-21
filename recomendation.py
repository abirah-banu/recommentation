import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# Sample data
movie_data = {
    'title': ['Toy Story', 'Jumanji', 'Grumpier Old Men', 'Waiting to Exhale', 'Father of the Bride Part II'],
    'genre': ['Animation|Adventure|Comedy', 'Adventure|Children|Fantasy', 'Comedy|Romance', 'Comedy|Drama|Romance', 'Comedy']
}

# Create DataFrame
movie_df = pd.DataFrame(movie_data)

# Create a count matrix based on genres
count_vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'))
count_matrix = count_vectorizer.fit_transform(movie_df['genre'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Convert the similarity matrix into a DataFrame
cosine_sim_df = pd.DataFrame(cosine_sim, index=movie_df['title'], columns=movie_df['title'])

# Function to get movie recommendations
def get_content_based_recommendations(movie_title, n_recommendations=2):
    if movie_title not in cosine_sim_df.index:
        raise ValueError(f"Movie title '{movie_title}' not found in the dataset.")
    # Get similarity scores of the given movie with all other movies
    similar_movies = cosine_sim_df[movie_title].sort_values(ascending=False)[1:n_recommendations+1]
    return similar_movies.index.tolist()

# Example usage
content_based_recommended_movies = get_content_based_recommendations('Toy Story')
print("Content-based recommended movies:", content_based_recommended_movies)
