# Experiment 5: Recommendation System
# Collaborative Filtering + Content-Based Filtering

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

print("\n Recommendation System \n - exp5.py:7")

# -----------------------------------
# 1. Sample User-Item Matrix
# Rows = Users, Columns = Items
# -----------------------------------
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [0, 0, 5, 4],
])

# -----------------------------------
# 2. Collaborative Filtering
# -----------------------------------
print("\n Collaborative Filtering \n - exp5.py:23")

# Compute similarity between users
user_similarity = cosine_similarity(ratings)
print("User Similarity Matrix:\n - exp5.py:27", user_similarity)

# Recommend items for User 0
user_index = 0
scores = user_similarity[user_index].dot(ratings)
print("\nRecommendation Scores for User 0:\n - exp5.py:32", scores)

# -----------------------------------
# 3. Content-Based Filtering
# -----------------------------------
print("\n ContentBased Filtering \n - exp5.py:37")

# Item features (example)
item_features = np.array([
    [1, 0, 1],  # Item 1
    [1, 1, 0],  # Item 2
    [0, 1, 1],  # Item 3
    [1, 0, 0],  # Item 4
])

# Compute similarity between items
item_similarity = cosine_similarity(item_features)
print("Item Similarity Matrix:\n - exp5.py:49", item_similarity)

# Recommend items similar to Item 1
item_index = 0
print("\nItems similar to Item 1:\n - exp5.py:53", item_similarity[item_index])