import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.sparse.linalg import svds


class SVDRecommender:
    def __init__(self, n_factors=15):
        self.n_factors = n_factors
        self.user_bias = defaultdict(float)
        self.item_bias = defaultdict(float)
        self.global_mean = 0.0
        self.user_factors = None
        self.item_factors = None

    def fit(self, ratings):
        df = pd.DataFrame(ratings, columns=['user_id', 'item_id', 'rating'])
        df = df.groupby(['user_id', 'item_id'])['rating'].mean().reset_index()

        self.global_mean = df['rating'].mean()
        user_means = df.groupby('user_id')['rating'].mean()
        item_means = df.groupby('item_id')['rating'].mean()

        for user, mean in user_means.items():
            self.user_bias[user] = mean - self.global_mean

        for item, mean in item_means.items():
            self.item_bias[item] = mean - self.global_mean

        df['adjusted_rating'] = df.apply(
            lambda x: x['rating'] - (self.global_mean + self.user_bias[x['user_id']] + self.item_bias[x['item_id']]),
            axis=1
        )

        rating_matrix = df.pivot(index='user_id', columns='item_id', values='adjusted_rating').fillna(0)
        U, sigma, Vt = svds(rating_matrix.values, k=self.n_factors)
        sigma = np.diag(sigma)

        self.user_factors = U
        self.item_factors = Vt.T
        self.user_ids = rating_matrix.index
        self.item_ids = rating_matrix.columns
        self.user_id_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.item_id_to_idx = {i: j for j, i in enumerate(self.item_ids)}

    def predict(self, user, item):
        if user not in self.user_id_to_idx or item not in self.item_id_to_idx:
            return self.global_mean + self.user_bias.get(user, 0) + self.item_bias.get(item, 0)

        user_idx = self.user_id_to_idx[user]
        item_idx = self.item_id_to_idx[item]
        pred = self.global_mean + self.user_bias[user] + self.item_bias[item]
        pred += np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        return np.clip(pred, 1, 5)

    def evaluate(self, test_ratings):
        errors = []
        for user, item, rating in test_ratings:
            pred = self.predict(user, item)
            errors.append((rating - pred) ** 2)
        return np.sqrt(np.mean(errors))

    def visualize_biases(self):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(list(self.user_bias.values()), bins=30, color='royalblue', alpha=0.7)
        plt.title("User Biases Distribution")

        plt.subplot(1, 2, 2)
        plt.hist(list(self.item_bias.values()), bins=30, color='orange', alpha=0.7)
        plt.title("Item Biases Distribution")

        plt.tight_layout()
        plt.show()
