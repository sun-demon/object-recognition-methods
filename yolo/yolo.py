import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Проверка наличия файлов
required_files = ['yolov3.weights', 'yolov3.cfg', 'coco.names']
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    raise FileNotFoundError(f"Отсутствуют файлы: {', '.join(missing_files)}\n"
                          "Скачайте их по ссылкам:\n"
                          "1. yolov3.weights: https://pjreddie.com/media/files/yolov3.weights\n"
                          "2. yolov3.cfg: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg\n"
                          "3. coco.names: https://github.com/pjreddie/darknet/blob/master/data/coco.names")

# Загрузка модели
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Укажите путь к вашему изображению
img_path = "../in/zialkovskiy.webp"  # Замените на ваш файл

if not os.path.exists(img_path):
    raise FileNotFoundError(f"Изображение {img_path} не найдено")

# Обработка изображения
img = cv2.imread(img_path)
if img is None:
    raise ValueError("Не удалось загрузить изображение. Проверьте формат файла.")

height, width = img.shape[:2]

# Подготовка для YOLO
blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
net.setInput(blob)

# Получение детекций
output_layers = net.getUnconnectedOutLayersNames()
layer_outputs = net.forward(output_layers)

# Обработка результатов
boxes = []
confidences = []
class_ids = []

for output in layer_outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w/2)
            y = int(center_y - h/2)
            
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Применяем NMS
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Исправление для разных версий OpenCV
if len(indexes) > 0:
    if isinstance(indexes, tuple) or isinstance(indexes, np.ndarray):
        indexes = indexes.flatten()
    
    # Отрисовка результатов
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    for i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[class_ids[i]]
        
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
        cv2.putText(img, f"{label} {confidence}", (x,y-5), font, 1.5, color, 2)

    # Сохранение и отображение
    output_path = "../out/detected_objects.jpg"
    cv2.imwrite(output_path, img)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Результат обнаружения объектов")
    plt.show()
    print(f"Успешно! Результат сохранён в {output_path}")
else:
    print("На изображении не обнаружено объектов")