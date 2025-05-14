import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

class FunkSVD:
    """
    Здесь мы реализуем алгоритм Funk-SVD с нуля
    с учетом смещений пользователей и товаров
    """

    def __init__(self, n_factors=15, n_epochs=25, lr=0.007, reg=0.03):
        """
        Описание параметров модели:

        - n_factors: количество латентных факторов
        - n_epochs: количество эпох обучения
        - lr: скорость обучения (learning rate)
        - reg: коэффициент регуляризации
        """
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg

        # Параметры модели
        self.user_bias = defaultdict(float)
        self.item_bias = defaultdict(float)
        self.user_factors = defaultdict(lambda: np.random.normal(0, 0.1, n_factors))
        self.item_factors = defaultdict(lambda: np.random.normal(0, 0.1, n_factors))
        self.global_mean = 0.0

    def fit(self, ratings):
        """
        Наша моделька обучается на данных оценок

        Параметры:
        - ratings: список кортежей (user_id, item_id, rating)
        """
        # Вычисляем средний рейтинг
        self.global_mean = np.mean([r for _, _, r in ratings])

        # Основной цикл обучения
        for epoch in range(self.n_epochs):
            for user, item, rating in ratings:
                # Предсказание текущего рейтинга
                pred = self.predict(user, item)
                error = rating - pred

                # Обновление смещений с регуляризацией
                self.user_bias[user] += self.lr * (error - self.reg * self.user_bias[user])
                self.item_bias[item] += self.lr * (error - self.reg * self.item_bias[item])

                # Обновление факторов
                u_factor = self.user_factors[user]
                i_factor = self.item_factors[item]

                # Градиентный шаг
                delta_u = error * i_factor - self.reg * u_factor
                delta_i = error * u_factor - self.reg * i_factor

                self.user_factors[user] += self.lr * delta_u
                self.item_factors[item] += self.lr * delta_i

    def predict(self, user, item):
        """
        Предсказание рейтинга для пары пользователь-товар
        """
        return (self.global_mean +
                self.user_bias.get(user, 0) +
                self.item_bias.get(item, 0) +
                np.dot(self.user_factors[user], self.item_factors[item]))

    def evaluate(self, test_ratings):
        """
        Оценка качества нашей модельки на тестиках

        Возвращает RMSE (показатель среднеквадратической ошибки (Root Mean Square Error))
        """
        squared_errors = []
        for user, item, rating in test_ratings:
            pred = self.predict(user, item)
            squared_errors.append((rating - pred) ** 2)

        return np.sqrt(np.mean(squared_errors))

    def visualize_biases(self):
        """
        Показываем на графике распределения смещений пользователей и товаров
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

def generate_synthetic_data(num_users=100, num_items=50, num_ratings=500):
    """
    Генерируем искусственные данные для тестов модельки
    """
    np.random.seed(6)
    users = np.random.randint(1, num_users+1, num_ratings)
    items = np.random.randint(1, num_items+1, num_ratings)


    ratings = np.random.randint(1, 6, num_ratings)
    return list(zip(users, items, ratings))


def train_test_split(ratings, test_size=0.2):
    """
    Разделяем данные на обучающую и тестовую выборки
    """
    np.random.seed(42)
    np.random.shuffle(ratings)
    split_idx = int(len(ratings) * (1 - test_size))
    return ratings[:split_idx], ratings[split_idx:]


if __name__ == "__main__":
    # Генерация данных
    print("Генерируем искуственые данные...")
    ratings = generate_synthetic_data()
    train_data, test_data = train_test_split(ratings)

    # Инициализация и обучение модели
    print("Обучаем модельку Funk-SVD...")
    model = FunkSVD(n_factors=15, n_epochs=25, lr=0.007, reg=0.03)
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
