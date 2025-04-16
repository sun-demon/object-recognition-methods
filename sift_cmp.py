import cv2
import numpy as np
from matplotlib import pyplot as plt

def match_and_save_color(img1_path, img2_path, output_path='matched_result.png'):
    # Загрузка цветных изображений
    img1 = cv2.imread('in/' + img1_path)
    img2 = cv2.imread('in/' + img2_path)
    
    # Конвертация в grayscale для детекции признаков
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Инициализация SIFT
    sift = cv2.SIFT_create()
    
    # Поиск ключевых точек
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)
    
    # Сопоставление точек
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    # Фильтрация совпадений
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    
    # Отображение совпадений (цветное)
    matched_img = cv2.drawMatches(
        img1, kp1, 
        img2, kp2, 
        good, None,
        matchColor=(0, 255, 0),  # Зеленые линии
        singlePointColor=(255, 0, 0),  # Красные точки
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    # Конвертация из BGR (OpenCV) в RGB (Matplotlib)
    matched_img_rgb = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB)
    
    # Сохранение без фона (PNG с прозрачностью)
    plt.figure(figsize=(20, 10), frameon=False)  # frameon=False убирает фон
    plt.imshow(matched_img_rgb)
    plt.axis('off')
    
    # Сохранение с прозрачным фоном
    plt.savefig('out/' + output_path, bbox_inches='tight', pad_inches=0, transparent=True, dpi=100)
    plt.close()
    
    print(f"Результат сохранен как {output_path}")

# Использование
match_and_save_color('war1.jpeg', 'war2.jpeg', 'sift_cmp_war.png')