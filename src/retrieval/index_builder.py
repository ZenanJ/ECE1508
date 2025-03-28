import pandas as pd
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import faiss
import torch
from sklearn.preprocessing import MinMaxScaler
import os


# Load CSV files
movies_df = pd.read_csv("movielens_data/raw/movies.csv")  # movieId, title, genres
ratings_df = pd.read_csv("movielens_data/raw/ratings.csv")  # userId, movieId, rating, timestamp
tags_df = pd.read_csv("movielens_data/raw/tags.csv")  # userId, movieId, tag, timestamp

save_path = "movielens_data/processed/processed_movies.csv"
embedding_save_path = "movielens_data/processed/movielens_hybrid.index"


def preprocess_data(movies_df, ratings_df, tags_df, save_path):
    # Aggregate ratings: Calculate average rating and number of ratings per movie
    ratings_agg = ratings_df.groupby("movieId").agg(
        avg_rating=("rating", "mean"),
        num_ratings=("rating", "count")
    ).reset_index()
    print(f"Movies with ratings: {len(ratings_agg)}")  # Check how many movies have ratings

    # Merge with movies dataset
    movies_df = movies_df.merge(ratings_agg, on="movieId", how="left")
    print(f"Movies after ratings merge: {len(movies_df)}")  # Should still be ~80,000

    # Merge tags (to add textual metadata)
    movie_tags = tags_df.groupby("movieId")["tag"].apply(lambda tags: " ".join(tags.astype(str))).reset_index()
    print(f"Movies with tags: {len(movie_tags)}")  # Check how many movies have tags

    movies_df = movies_df.merge(movie_tags, on="movieId", how="left")
    print(f"Movies after tags merge: {len(movies_df)}")  # Should still be ~80,000

    # Fill missing values (movies without ratings or tags)
    movies_df["avg_rating"] = movies_df["avg_rating"].fillna(0)  # No ratings → avg 0
    movies_df["num_ratings"] = movies_df["num_ratings"].fillna(0)  # No ratings → 0 count
    movies_df["tag"] = movies_df["tag"].fillna("")  # No tags → empty string

    # Concatenate text fields for embedding
    movies_df["text"] = movies_df["title"] + " " + movies_df["genres"] + " " + movies_df["tag"]
    print(f"Movies count before save: {len(movies_df)}")
    movies_df.to_csv(save_path, index=False)
    return movies_df


# this method only convert the processed_movies_df["text"] column into embeddings
def generate_text_embeddings(processed_movies_df):
    if torch.cuda.is_available():
        model_device = {'device': 'cuda'}
    else:
        model_device = {'device': 'cpu'}
    print(f"Using device {model_device['device']}")

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Generate text embeddings
    text_embeddings = model.encode(processed_movies_df["text"].tolist(), batch_size=64, show_progress_bar=True)
    return text_embeddings


# this method normalize and append the numerical rating values to the text embeddings
def append_numerical_features(processed_movies_df, text_embeddings):
    print("Currently normalizing the rating values and add to text embeddings")
    # Select numerical columns
    numeric_features = processed_movies_df[["avg_rating", "num_ratings"]].values

    # Apply MinMax scaling (scale between 0 and 1)
    scaler = MinMaxScaler()
    scaled_numeric_features = scaler.fit_transform(numeric_features)

    # Convert to NumPy
    scaled_numeric_features = np.array(scaled_numeric_features).astype("float32")

    # Convert text embeddings to NumPy
    text_embeddings = np.array(text_embeddings).astype("float32")

    # Concatenate text embeddings with scaled numerical features
    hybrid_embeddings = np.hstack([text_embeddings, scaled_numeric_features])
    return hybrid_embeddings


def store_embeddings(hybrid_embeddings, embedding_save_path):
    print("Currently storing the hybrid embeddings")
    # Define FAISS index (L2 distance)
    embedding_dimension = hybrid_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dimension)

    # Add embeddings to FAISS index
    index.add(hybrid_embeddings)

    # Save FAISS index
    faiss.write_index(index, embedding_save_path)
    return index

if __name__ == "__main__":
    # preprocess data and save it (this may take very long time)
    if not os.path.exists(save_path):
        print("It's processing the data currently, and may take very long time dur to large file size")
        preprocess_data(movies_df, ratings_df, tags_df, save_path)

    if not os.path.exists(embedding_save_path):
        processed_movies_df = pd.read_csv(save_path)
        # generate text embeddings based on the "text" column
        text_embedding = generate_text_embeddings(processed_movies_df)
        # print(f"Text embedding examples: {text_embedding}")

        # also add the numerical rating values to embeddings
        hybrid_embedding = append_numerical_features(processed_movies_df, text_embedding)
        # print(f"Hybrid embedding examples: ", {hybrid_embedding})

        # Initialize FAISS index and save locally
        index = store_embeddings(hybrid_embedding, embedding_save_path)
        print("The embedding index successfully save within movielens_data/processed folder")