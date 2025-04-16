import cv2
import matplotlib.pyplot as plt

# Загрузка изображения
# filein = "kalyazin.jpg"
# filein = "kazanskaya_amvrosievskaya_pustyn.jpg"
filein = "vostok_night.jpg"

fileout = f"sift_{filein}"

image = cv2.imread(f"in/{filein}")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Инициализация SIFT
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Отрисовка ключевых точек
image_sift = cv2.drawKeypoints(
    image, 
    keypoints, 
    None, 
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# Сохранение и отображение
plt.imshow(cv2.cvtColor(image_sift, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.savefig(f"out/{fileout}", bbox_inches="tight", pad_inches=0, dpi=300)
plt.show()