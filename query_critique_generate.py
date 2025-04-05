import pandas as pd
import json
import random
import os

# ===== File paths =====
GROUND_TRUTH_PATH = r"C:\Users\71850\Desktop\1508\ground_truth_subset.json"
MOVIES_CSV_PATH = r"C:\Users\71850\Desktop\1508\ml-32m\movies.csv"
OUTPUT_PROMPTS_PATH = r"C:\Users\71850\Desktop\1508\evaluation_prompts.json"

# ===== Load data =====
with open(GROUND_TRUTH_PATH, "r") as f:
    ground_truth = json.load(f)

movies_df = pd.read_csv(MOVIES_CSV_PATH)
movie_id_to_title = dict(zip(movies_df["movieId"], movies_df["title"]))

# ===== Categorized critique pool =====
critique_pool = {
    "emotional": [
        "I prefer happy endings over tragic ones.",
        "I'm not a fan of sad or depressing stories.",
        "I enjoy emotionally powerful and character-driven stories.",
        "I want something that makes me feel inspired.",
        "I prefer romance to action."
    ],
    "genre": [
        "Please no horror or jump scares.",
        "I enjoy historical dramas.",
        "I don’t enjoy fantasy or science fiction.",
        "No musicals, please.",
        "I love movies based on true stories.",
        "I enjoy comedies more than dramas."
    ],
    "pacing": [
        "I prefer fast-paced plots to slow dramas.",
        "Nothing too complicated—keep it simple.",
        "I like movies with clever plot twists.",
        "I prefer movies with minimal dialogue.",
        "Keep it light and entertaining."
    ],
    "format": [
        "I like older classics from the 80s or 90s.",
        "Please avoid black-and-white films.",
        "I’d rather avoid movies that are over 2 hours long.",
        "I enjoy visually stunning films with great cinematography.",
        "I prefer movies with modern visuals and soundtracks."
    ]
}

# ===== Build full prompt list =====
prompts = []
for user_id, liked_movies in ground_truth.items():
    if not liked_movies:
        continue

    # Select a reference movie the user likes
    ref_movie_id = random.choice(liked_movies)
    ref_title = movie_id_to_title.get(ref_movie_id, f"Movie ID {ref_movie_id}")

    query = f"Can you recommend some movies similar to '{ref_title}'?"

    # Randomly choose a critique category and a critique
    critique_type = random.choice(list(critique_pool.keys()))
    critique = random.choice(critique_pool[critique_type])

    # Save prompt
    prompts.append({
        "user_id": int(user_id),
        "query": query,
        "critique": critique,
        "critique_type": critique_type
    })

# ===== Save to file =====
os.makedirs(os.path.dirname(OUTPUT_PROMPTS_PATH), exist_ok=True)
with open(OUTPUT_PROMPTS_PATH, "w", encoding="utf-8") as f:
    json.dump(prompts, f, ensure_ascii=False, indent=2)

print(f"✅ Successfully generated {len(prompts)} prompts with critique types saved to:\n{OUTPUT_PROMPTS_PATH}")









