import cv2
import matplotlib.pyplot as plt

# Загрузка изображения
# filein = "kalyazin.jpg"
# filein = "kazanskaya_amvrosievskaya_pustyn.jpg"
filein = "vostok_night.jpg"

fileout = f"orb_{filein}"

image = cv2.imread(f"in/{filein}")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Инициализация ORB
orb = cv2.ORB_create(nfeatures=500)
keypoints, descriptors = orb.detectAndCompute(gray, None)

# Визуализация
image_orb = cv2.drawKeypoints(
    image, keypoints, None, 
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.imshow(cv2.cvtColor(image_orb, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.savefig(f"out/{fileout}", bbox_inches="tight", pad_inches=0, dpi=300)
plt.show()