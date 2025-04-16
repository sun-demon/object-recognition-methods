from skimage.feature import hog
import cv2
from skimage import exposure
import matplotlib.pyplot as plt

# Загрузка изображения
# filein = "kalyazin.jpg"
# filein = "kazanskaya_amvrosievskaya_pustyn.jpg"
filein = "vostok_night.jpg"

fileout = f"hog_{filein}"

image = cv2.imread(f"in/{filein}")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Вычисление HOG-дескрипторов
features, hog_image = hog(
    gray,
    orientations=9,           # Количество направлений градиентов
    pixels_per_cell=(8, 8),   # Размер ячейки
    cells_per_block=(2, 2),   # Блоки для нормализации
    visualize=True,           # Визуализация
    block_norm="L2-Hys"       # Метод нормализации
)

# Улучшение контраста для отображения
hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# Сохранение
plt.imshow(hog_image, cmap="gray")
plt.axis("off")
plt.savefig(f"out/{fileout}", bbox_inches="tight", pad_inches=0, dpi=300)
plt.show()