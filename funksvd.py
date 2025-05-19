import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class FunkSVD:
    def __init__(self, n_factors=20, n_epochs=50, lr=0.01, reg=0.015):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg

        # Инициализация с более умным распределением
        self.user_bias = defaultdict(float)
        self.item_bias = defaultdict(float)
        self.user_factors = defaultdict(lambda: np.random.uniform(-0.05, 0.05, n_factors))
        self.item_factors = defaultdict(lambda: np.random.uniform(-0.05, 0.05, n_factors))
        self.global_mean = 0.0

    def fit(self, ratings, verbose=True):
        self.global_mean = np.mean([r for _, _, r in ratings])

        # Шаг обучения с адаптивным LR
        for epoch in range(self.n_epochs):
            total_error = 0
            for user, item, rating in ratings:
                # Предсказание с кешированием
                user_vec = self.user_factors[user]
                item_vec = self.item_factors[item]
                pred = self.global_mean + self.user_bias[user] + self.item_bias[item] + np.dot(user_vec, item_vec)

                error = rating - pred
                total_error += error ** 2

                # Адаптивное обновление
                lr = self.lr * (0.9 ** (epoch // 10))  # Постепенно уменьшаем LR

                # Обновление смещений
                self.user_bias[user] += lr * (error - self.reg * self.user_bias[user])
                self.item_bias[item] += lr * (error - self.reg * self.item_bias[item])

                # Обновление факторов
                delta_u = error * item_vec - self.reg * user_vec
                delta_i = error * user_vec - self.reg * item_vec
                self.user_factors[user] += lr * delta_u
                self.item_factors[item] += lr * delta_i

            if verbose and epoch % 5 == 0:
                print(f"Epoch {epoch}, RMSE: {np.sqrt(total_error / len(ratings)):.4f}")

    def predict(self, user, item):
        try:
            return np.clip(
                self.global_mean +
                self.user_bias.get(user, 0) +
                self.item_bias.get(item, 0) +
                np.dot(self.user_factors[user], self.item_factors[item]),
                1, 5
            )
        except KeyError:
            return self.global_mean

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
