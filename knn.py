import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import euclidean_distances
from PIL import Image
import os

# Названия классов Fashion MNIST
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# 1. Загрузка Fashion MNIST
print("Загрузка Fashion MNIST...")
fashion = fetch_openml(name='Fashion-MNIST', version=1, as_frame=False)
X, y = fashion.data[:6000], fashion.target.astype(int)[:6000]  # небольшой поднабор

# 2. Обучение k-NN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# 3. Функция сравнения изображения
def compare_image(my_img_path, output_path='out/knn_match.jpg'):
    # Загрузка и обработка изображения
    img = Image.open(my_img_path).convert('L')  # в оттенки серого
    img = img.resize((28, 28))
    img_array = np.array(img)
    img_array = 255 - img_array  # инверсия цветов
    img_flat = img_array.reshape(1, -1)

    # Предсказание класса
    pred = model.predict(img_flat)[0]

    # Поиск ближайшего изображения в датасете
    dists = euclidean_distances(img_flat, X)
    best_match_idx = np.argmin(dists)
    best_match_img = X[best_match_idx].reshape(28, 28)
    best_match_label = y[best_match_idx]

    # Визуализация сравнения
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    ax1.imshow(img_array, cmap='gray')
    ax1.set_title("Ваше изображение")
    ax1.axis('off')

    ax2.imshow(best_match_img, cmap='gray')
    ax2.set_title(f"Похожее: {class_names[best_match_label]}")
    ax2.axis('off')

    # Сохранение
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Результат сохранён в {output_path}")

# 4. Запуск (замени путь на свой файл)
compare_image("in/running_shoes.jpeg")
