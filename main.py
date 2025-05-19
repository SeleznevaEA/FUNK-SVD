from funksvd import FunkSVD
from svd import SVDRecommender
import numpy as np
import matplotlib.pyplot as plt
import time


def generate_synthetic_data(num_users=1000, num_items=500, num_ratings=10000, seed=62):
    np.random.seed(seed)
    users = np.random.randint(1, num_users + 1, num_ratings)
    items = np.random.randint(1, num_items + 1, num_ratings)

    user_effects = np.random.normal(0, 0.8, num_users)
    item_effects = np.random.normal(0, 0.5, num_items)

    ratings = []
    for i in range(num_ratings):
        user = users[i]
        item = items[i]
        base = 3 + user_effects[user - 1] + item_effects[item - 1]
        noise = np.random.normal(0, 0.3)
        rating = np.clip(base + noise, 1, 5)
        ratings.append((user, item, float(round(rating, 1))))

    return ratings


def train_test_split(ratings, test_size=0.2, seed=62):
    np.random.seed(seed)
    np.random.shuffle(ratings)
    split = int(len(ratings) * (1 - test_size))
    return ratings[:split], ratings[split:]


def plot_rmse_comparison(funk_rmse, svd_rmse):
    plt.figure(figsize=(8, 5))
    models = ['Funk-SVD', 'SVD']
    rmses = [funk_rmse, svd_rmse]
    bars = plt.bar(models, rmses, color=['royalblue', 'orange'])

    plt.title('Сравнение RMSE моделей', fontsize=14)
    plt.ylabel('RMSE', fontsize=12)
    plt.ylim(min(rmses) * 0.95, max(rmses) * 1.05)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_rating_distribution(ratings):
    ratings_values = [r[2] for r in ratings]
    plt.figure(figsize=(10, 5))

    plt.hist(ratings_values, bins=20, color='green', edgecolor='black', alpha=0.7)
    plt.title('Распределение оценок в данных', fontsize=14)
    plt.xlabel('Оценка', fontsize=12)
    plt.ylabel('Количество', fontsize=12)
    plt.xticks(np.arange(1, 5.5, 0.5))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def plot_predictions_vs_actual(test_data, funk_preds, svd_preds):
    actual = [r[2] for r in test_data]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(actual, funk_preds, alpha=0.5, color='royalblue')
    plt.plot([1, 5], [1, 5], 'r--')
    plt.title('Funk-SVD: Предсказания vs Реальные значения')
    plt.xlabel('Реальные оценки')
    plt.ylabel('Предсказанные оценки')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(actual, svd_preds, alpha=0.5, color='orange')
    plt.plot([1, 5], [1, 5], 'r--')
    plt.title('SVD: Предсказания vs Реальные значения')
    plt.xlabel('Реальные оценки')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Генерация синтетических данных...")
    data = generate_synthetic_data()
    train, test = train_test_split(data)
    plot_rating_distribution(data)

    print("\nОбучение FunkSVD...")
    start_time = time.time()
    funk = FunkSVD(n_factors=20, n_epochs=50, lr=0.01, reg=0.015)
    funk.fit(train)
    funk_time = time.time() - start_time
    funk_rmse = funk.evaluate(test)
    print(f"FunkSVD Test RMSE: {funk_rmse:.4f}, Время: {funk_time:.2f} сек")

    print("\nОбучение SVD...")
    start_time = time.time()
    svd = SVDRecommender(n_factors=20)
    svd.fit(train)
    svd_time = time.time() - start_time
    svd_rmse = svd.evaluate(test)
    print(f"SVD Test RMSE: {svd_rmse:.4f}, Время: {svd_time:.2f} сек")

    # Собираем предсказания для графиков
    funk_preds = [funk.predict(user, item) for user, item, _ in test]
    svd_preds = [svd.predict(user, item) for user, item, _ in test]

    # Визуализация
    plot_rmse_comparison(funk_rmse, svd_rmse)
    plot_predictions_vs_actual(test, funk_preds, svd_preds)

    # Графики смещений из моделей
    print("\nВизуализация смещений Funk-SVD:")
    funk.visualize_biases()

    print("\nВизуализация смещений SVD:")
    svd.visualize_biases()
