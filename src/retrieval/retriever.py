import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
import os


class Retriever:
    def __init__(self):
        self.processed_movies_path = "../../movielens_data/processed/processed_movies.csv"
        self.embedding_save_path = "../../movielens_data/processed/movielens_hybrid.index"

        # Load data
        if not os.path.exists(self.processed_movies_path):
            raise FileNotFoundError(f"Processed movies file not found at {self.processed_movies_path}")
        self.processed_movies_df = pd.read_csv(self.processed_movies_path)

        # Load FAISS index
        if not os.path.exists(self.embedding_save_path):
            raise FileNotFoundError(f"FAISS index file not found at {self.embedding_save_path}")
        self.index = faiss.read_index(self.embedding_save_path)

        # Load model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Fit scaler on existing data
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.processed_movies_df[["avg_rating", "num_ratings"]])

        # Store column means for fallback
        self.default_avg_rating = self.processed_movies_df["avg_rating"].mean()
        self.default_num_ratings = self.processed_movies_df["num_ratings"].mean()


    def search_movie(self, query_text, avg_rating=None, num_ratings=None, top_k=50):
        # Use defaults if None
        if avg_rating is None:
            avg_rating = self.default_avg_rating
        if num_ratings is None:
            num_ratings = self.default_num_ratings

        # Generate text embedding
        query_embedding = self.model.encode([query_text])

        # Normalize numerical features
        query_numeric = self.scaler.transform([[avg_rating, num_ratings]]).astype("float32")

        # Combine text and numerical embedding
        hybrid_query = np.hstack([query_embedding, query_numeric])

        # Search FAISS index
        distances, indices = self.index.search(np.array(hybrid_query).astype("float32"), top_k)

        # Return top results
        return self.processed_movies_df.iloc[indices[0]]


if __name__ == "__main__":
    # Optional: Set max column width if needed
    pd.set_option("display.max_columns", None)  # Show all columns
    pd.set_option("display.width", None)  # No wrapping
    pd.set_option("display.max_colwidth", None)  # Show full content in each cell
    pd.set_option("display.max_rows", None)  # Show all rows if needed


    # Example Usage: Find a sci-fi movie with avg rating ≥ 4.0 and > 500 ratings
    retriever = Retriever()
    print("Example 1: Find a sci-fi movie with avg rating ≥ 4.0 and > 500 ratings")
    query_result = retriever.search_movie(
        query_text="sci-fi space adventure",
        avg_rating=4.0,
        num_ratings=500,
        top_k=10
    )

    print(query_result[["movieId", "title", "genres", "avg_rating", "num_ratings"]])

    print("Example 2: Find a romantic movie")
    query_result = retriever.search_movie(
        query_text="Find a romantic love movie",
        top_k=10
    )

    print(query_result[["movieId", "title", "genres", "avg_rating", "num_ratings"]])
