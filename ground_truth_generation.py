import pandas as pd
import json
import os
import random


def load_ratings(file_path):
    return pd.read_csv(file_path)


def filter_active_users(ratings_df, min_interactions=20):
    user_counts = ratings_df['userId'].value_counts()
    active_users = user_counts[user_counts >= min_interactions].index.tolist()
    return active_users


def build_ground_truth_subset(ratings_df, user_ids, min_rating=4.0):

    liked_df = ratings_df[(ratings_df['userId'].isin(user_ids)) & (ratings_df['rating'] >= min_rating)]

    # 可选：按评分排序
    liked_df = liked_df.sort_values(by=['userId', 'rating'], ascending=[True, False])

    ground_truth = (
        liked_df.groupby('userId')['movieId']
        .apply(list)
        .to_dict()
    )
    return ground_truth


def save_ground_truth(ground_truth_dict, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(ground_truth_dict, f)
    print(f"✅ Saved {len(ground_truth_dict)} users to {output_file}")


if __name__ == "__main__":
    ratings_file = r"C:\Users\71850\Desktop\1508\ml-32m\ratings.csv"
    save_path = r"C:\Users\71850\Desktop\1508\ground_truth_subset.json"

    ratings = load_ratings(ratings_file)
    active_users = filter_active_users(ratings, min_interactions=20)

    # 👇 从中随机抽取 200 个用户
    selected_users = random.sample(active_users, 200)

    ground_truth = build_ground_truth_subset(ratings, selected_users)
    save_ground_truth(ground_truth, output_file=save_path)
