import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.sparse.linalg import svds


class SVDRecommender:
    """
    Реализуем рекомендательноую систему на основе SVD
    с учетом смещений пользователей и товаров
    """

    def __init__(self, n_factors=15):
        """
        Параметры:
        - n_factors: количество факторов
        """
        self.n_factors = n_factors
        self.user_bias = defaultdict(float)
        self.item_bias = defaultdict(float)
        self.global_mean = 0.0
        self.user_factors = None
        self.item_factors = None

    def fit(self, ratings):
        """
        Обучаем модельку на данных оценок

        Параметры:
        - ratings: список кортежей (user_id, item_id, rating)
        """
        # Создаем DataFrame из оценок
        df = pd.DataFrame(ratings, columns=['user_id', 'item_id', 'rating'])

        # Обрабатываем дубликаты - берем среднее значение для одинаковых пар (user_id, item_id)
        df = df.groupby(['user_id', 'item_id'])['rating'].mean().reset_index()

        # Вычисляем средние значения
        self.global_mean = df['rating'].mean()
        user_means = df.groupby('user_id')['rating'].mean()
        item_means = df.groupby('item_id')['rating'].mean()

        # Заполняем смещения
        for user, mean in user_means.items():
            self.user_bias[user] = mean - self.global_mean

        for item, mean in item_means.items():
            self.item_bias[item] = mean - self.global_mean

        # Создаем матрицу оценок с учетом смещений
        df['adjusted_rating'] = df.apply(
            lambda x: x['rating'] - (self.global_mean + self.user_bias[x['user_id']] + self.item_bias[x['item_id']]),
            axis=1
        )

        # Создаем разреженную матрицу пользователь-товар (теперь без дубликатов)
        rating_matrix = df.pivot(index='user_id', columns='item_id', values='adjusted_rating').fillna(0)

        # Выполняем SVD
        U, sigma, Vt = svds(rating_matrix.values, k=self.n_factors)

        # Преобразуем sigma в диагональную матрицу
        sigma = np.diag(sigma)

        # Сохраняем факторы
        self.user_factors = U
        self.item_factors = Vt.T

        # Создаем mapping user_id/item_id к индексам матрицы
        self.user_ids = rating_matrix.index
        self.item_ids = rating_matrix.columns
        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.item_id_to_idx = {item_id: idx for idx, item_id in enumerate(self.item_ids)}

    def predict(self, user, item):
        """
        Предсказываем рейтинг для пары пользователь-товар
        """
        if user not in self.user_id_to_idx or item not in self.item_id_to_idx:
            return self.global_mean + self.user_bias.get(user, 0) + self.item_bias.get(item, 0)

        user_idx = self.user_id_to_idx[user]
        item_idx = self.item_id_to_idx[item]

        pred = (self.global_mean +
                self.user_bias.get(user, 0) +
                self.item_bias.get(item, 0) +
                np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))

        # Ограничиваем предсказание в разумных пределах
        return np.clip(pred, 1, 5)

    def evaluate(self, test_ratings):
        """
        Оцениваем качество модельки на тестовых данных

        Возвращаем RMSE
        """
        squared_errors = []
        for user, item, rating in test_ratings:
            pred = self.predict(user, item)
            squared_errors.append((rating - pred) ** 2)

        return np.sqrt(np.mean(squared_errors))

    def visualize_biases(self):
        """
        Визуализируем распределения смещений пользователей и товаров
        """
        plt.figure(figsize=(12, 5))

        # Распределение смещений пользователей
        plt.subplot(1, 2, 1)
        plt.hist(list(self.user_bias.values()), bins=20, color='darkmagenta')
        plt.title("Распределение User Bias")
        plt.xlabel("Значение смещения")
        plt.ylabel("Количество пользователей")

        # Распределение смещений товаров
        plt.subplot(1, 2, 2)
        plt.hist(list(self.item_bias.values()), bins=20, color='darkmagenta')
        plt.title("Распределение Item Bias")
        plt.xlabel("Значение смещения")
        plt.ylabel("Количество товаров")

        plt.tight_layout()
        plt.show()


def generate_synthetic_data(num_users=10000, num_items=5000, num_ratings=50000):
    np.random.seed(53)
    users = np.random.randint(1, num_users + 1, num_ratings)
    items = np.random.randint(1, num_items + 1, num_ratings)

    # Добавляем смещения для пользователей и товаров:
    user_biases = np.random.normal(0, 1, num_users)
    item_biases = np.random.normal(0, 1, num_items)

    ratings = []
    for _ in range(num_ratings):
        user = users[_]
        item = items[_]
        # Базовый рейтинг + смещение пользователя + смещение товара + шум
        rating = 3 + user_biases[user - 1] + item_biases[item - 1] + np.random.normal(0, 0.5)
        rating = np.clip(rating, 1, 5)
        ratings.append((user, item, int(round(rating))))

    return ratings

def train_test_split(ratings, test_size=0.2):
    """
    Разделяем данные на обучающую и тестовую выборки
    """
    np.random.seed(25)
    np.random.shuffle(ratings)
    split_idx = int(len(ratings) * (1 - test_size))
    return ratings[:split_idx], ratings[split_idx:]


if __name__ == "__main__":
    # Генерация данных
    print("Генерируем искуственные данные...")
    ratings = generate_synthetic_data()
    train_data, test_data = train_test_split(ratings)

    # Инициализация и обучение модели
    print("Обучаем модельку SVD...")
    model = SVDRecommender(n_factors=15)
    model.fit(train_data)

    # Оценка качества
    rmse = model.evaluate(test_data)
    print(f"\nRMSE на тестовых данных: {rmse:.4f}")

    # Примеры предсказаний
    print("\nПримеры предсказаний:")
    test_samples = test_data[:3]  # Первые 3 примера из тестовой выборки
    for user, item, true_rating in test_samples:
        pred = model.predict(user, item)
        print(f"Пользователь {user}, Товар {item}:")
        print(f"  Истинный рейтинг: {true_rating}")
        print(f"  Предсказанный: {pred:.2f}")
        print(f"  User bias: {model.user_bias.get(user, 0):.2f}")
        print(f"  Item bias: {model.item_bias.get(item, 0):.2f}\n")

    # Визуализация распределения смещений
    print("Визуализируем распределения смещений...")
    model.visualize_biases()