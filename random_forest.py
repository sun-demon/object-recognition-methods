import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import seaborn as sns

# Загружаем Fashion MNIST с OpenML (не нужен TensorFlow)
print("Загрузка Fashion MNIST...")
fashion = fetch_openml(name='Fashion-MNIST', version=1, as_frame=False)
X, y = fashion.data, fashion.target.astype(int)

# Уменьшаем выборку для быстроты
X, y = X[:5000], y[:5000]

# Делим на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Обучаем модель
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, y_train)

# Предсказания
y_pred = clf.predict(X_test)

# Визуализация примеров
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {y_pred[i]}")
    plt.axis('off')
plt.suptitle("Fashion MNIST + Random Forest")
plt.tight_layout()
plt.show()

# Отчёт
print("Classification Report:\n", classification_report(y_test, y_pred))

# Матрица ошибок
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Purples')
plt.title("Матрица ошибок")
plt.xlabel("Предсказано")
plt.ylabel("Истина")
plt.show()
