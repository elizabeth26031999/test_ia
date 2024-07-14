import os
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

# Rutas y configuración
MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PATH_TO_MODEL_DIR = 'model/' + MODEL_NAME + '/saved_model'
PATH_TO_LABELS = 'data/annotations/label_map.pbtxt'
MIN_CONF_THRESH = 0.5

# Cargar modelo
detect_fn = tf.saved_model.load(PATH_TO_MODEL_DIR)

# Cargar etiquetas del mapa
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Función para detectar maracuyás en una imagen
def detect_passion_fruit(image):
    image_np = np.array(image)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # Filtrar detecciones de maracuyás
    classes = detections['detection_classes']
    boxes = detections['detection_boxes']
    scores = detections['detection_scores']

    for i in range(len(classes)):
        if scores[i] > MIN_CONF_THRESH and classes[i] == 54:  # Clase 54: maracuyá en el modelo COCO
            box = boxes[i]
            h, w, _ = image.shape
            ymin, xmin, ymax, xmax = box
            left, right, top, bottom = xmin * w, xmax * w, ymin * h, ymax * h

            # Determinar tamaño y color del borde
            size = determine_size(left, right, top, bottom, w, h)
            color = determine_color(size)

            # Dibujar cuadro y texto
            cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), color, thickness=2)
            cv2.putText(image, f"Tamaño fruta maracuyá: {size}", (int(left), int(top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

# Función auxiliar para determinar el tamaño de la maracuyá
def determine_size(left, right, top, bottom, width, height):
    area = (right - left) * (bottom - top)
    image_area = width * height

    if area / image_area < 0.05:
        return "pequeño"
    elif area / image_area > 0.15:
        return "grande"
    else:
        return "mediano"

# Función auxiliar para determinar el color del borde según el tamaño
def determine_color(size):
    if size == "pequeño":
        return (0, 0, 255)  # Rojo
    elif size == "grande":
        return (0, 255, 0)  # Verde
    else:
        return (255, 0, 0)  # Azul

# Función principal para detección en vivo
def main():
    cap = cv2.VideoCapture(0)  # Captura de video desde la cámara
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = detect_passion_fruit(frame)
        
        cv2.imshow('Maracuyá Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
