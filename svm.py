import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline  # Добавлен этот импорт
from PIL import Image

# 1. Загрузка датасета (цветные изображения в оригинальном качестве)
lfw = fetch_lfw_people(color=True, min_faces_per_person=70, resize=0.7)
X = lfw.data / 255.0
y = lfw.target

# 2. Обучение модели
model = make_pipeline(
    PCA(n_components=100),
    SVC(kernel='rbf', probability=True)
)
model.fit(X, y)

# 3. Функция сравнения лиц
def compare_faces(my_face_path, output_path='out/svm.jpg'):
    # Загрузка и подготовка вашего фото
    my_face = np.array(Image.open(my_face_path).convert('RGB'))
    my_face = Image.fromarray(my_face).resize((lfw.images.shape[2], lfw.images.shape[1]))
    my_face_array = np.array(my_face) / 255.0
    my_face_flat = my_face_array.reshape(1, -1)
    
    # Поиск самого похожего лица
    probs = model.predict_proba(my_face_flat)[0]
    best_match_idx = np.argmax(probs)
    best_match_face = X[y == best_match_idx][0].reshape(*lfw.images.shape[1:])
    
    # Создание изображения для сравнения
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Ваше лицо
    ax1.imshow(my_face_array)
    ax1.set_title("Ваше лицо", pad=10)
    ax1.axis('off')
    
    # Похожее лицо
    ax2.imshow(best_match_face)
    ax2.set_title(f"Самый похожий: {lfw.target_names[best_match_idx]}", pad=10)
    ax2.axis('off')
    
    # Сохранение без рамок
    plt.subplots_adjust(wspace=0.05)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()
    print(f"Результат сохранен в {output_path}")

# 4. Запуск
compare_faces("in/my_face.jpg")  # Укажите путь к вашему фото