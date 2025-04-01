import cv2
import numpy as np

# Descargar estos archivos pre-entrenados:
# https://github.com/chuanqi305/MobileNet-SSD/blob/master/MobileNetSSD_deploy.prototxt
# https://github.com/chuanqi305/MobileNet-SSD/blob/master/MobileNetSSD_deploy.caffemodel

# Clases que el modelo puede detectar
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Colores aleatorios para cada clase
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Cargar el modelo
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")

def detect_objects(image_path, confidence_threshold=0.5):
    # Cargar imagen
    image = cv2.imread(image_path)
    if image is None:
        print("Error: No se pudo cargar la imagen")
        return
    
    (h, w) = image.shape[:2]
    
    # Preprocesar la imagen para el modelo
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    
    # Pasar la imagen por la red neuronal
    net.setInput(blob)
    detections = net.forward()
    
    # Procesar las detecciones
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filtrar por confianza mínima
        if confidence > confidence_threshold:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Dibujar la predicción
            label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
            
            # Ajustar posición del texto si el recuadro está muy arriba
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    
    # Mostrar resultado
    cv2.imshow("Detección de Objetos", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Ejemplo de uso
detect_objects("C:\Users\pc\Pictures\Camera Roll\WIN_20250401_15_55_28_Pro")  # Reemplaza con tu imagen