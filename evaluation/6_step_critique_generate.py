import json
import pandas as pd
import random
import os

# === 修改为你的本地路径 ===
GROUND_TRUTH_PATH = r"C:\Users\71850\Desktop\1508\ground_truth_subset.json"
MOVIES_CSV_PATH = r"C:\Users\71850\Desktop\1508\ml-32m\movies.csv"
OUTPUT_PROMPTS_PATH = r"C:\Users\71850\Desktop\1508\evaluation_prompts_6step.json"

# === 加载数据 ===
with open(GROUND_TRUTH_PATH, "r") as f:
    ground_truth = json.load(f)

movies_df = pd.read_csv(MOVIES_CSV_PATH)
movie_id_to_title = dict(zip(movies_df["movieId"], movies_df["title"]))

# === 更丰富的自然语言 critique 语料（不分类型）===
critique_list = [
    "I prefer happy endings.",
    "Avoid sad or depressing stories.",
    "Please make it more light-hearted.",
    "I enjoy movies based on true stories.",
    "Don't include horror elements.",
    "Avoid anything longer than 2 hours.",
    "I like fast-paced action.",
    "No old-fashioned or black-and-white films.",
    "Make it romantic.",
    "I want more thrill and excitement.",
    "I love visually stunning cinematography.",
    "Keep it emotionally deep.",
    "I enjoy strong female leads.",
    "Avoid war-themed movies.",
    "I like minimal dialogue and atmosphere.",
    "Don’t include musicals.",
    "Avoid open endings — I want closure.",
    "Make it feel nostalgic.",
    "I prefer modern productions after 2000.",
    "Nothing too complex — keep it simple.",
    "Avoid fantasy or science fiction.",
    "Give me more plot twists!",
    "Make it slow and meditative.",
    "Include powerful moral themes.",
    "I want something uplifting and inspiring."
]

# === 构造每个用户的 6-step prompt ===
multi_step_prompts = []

for user_id, liked_movies in ground_truth.items():
    if not liked_movies:
        continue

    ref_movie_id = random.choice(liked_movies)
    ref_title = movie_id_to_title.get(ref_movie_id, f"Movie ID {ref_movie_id}")
    query = f"Can you recommend some movies similar to '{ref_title}'?"

    # 随机选出 5 条 critique（不分类）
    selected_critiques = random.sample(critique_list, 5)

    prompt = {
        "user_id": int(user_id),
        "query": query,
        "steps": []
    }

    # step 0：只有 query
    prompt["steps"].append({
        "step": 0,
        "critiques": []
    })

    # step 1 ~ 5：累计加入 critique
    for step in range(1, 6):
        current_critiques = selected_critiques[:step]
        prompt["steps"].append({
            "step": step,
            "critiques": current_critiques
        })

    multi_step_prompts.append(prompt)

# === 保存为 JSON 文件 ===
with open(OUTPUT_PROMPTS_PATH, "w", encoding="utf-8") as f:
    json.dump(multi_step_prompts, f, ensure_ascii=False, indent=2)

print(f"✅ Successfully generated {len(multi_step_prompts)} 6-step prompts (no category) to:\n{OUTPUT_PROMPTS_PATH}")
