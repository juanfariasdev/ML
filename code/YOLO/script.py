import cv2
import numpy as np
import os

script_directory = os.path.dirname(os.path.abspath(__file__))

# Carregar o modelo YOLO
yolo_weights_file = "yolov3-spp.weights"
yolo_weights_path = os.path.join(script_directory, "config", yolo_weights_file)


yolo_config_file = "yolov3.spp.cfg"
yolo_config_path = os.path.join(script_directory, "config", yolo_config_file)
yolo_net = cv2.dnn.readNet(yolo_weights_path, yolo_config_path)

# Carregar as classes
yolo_classes_file = "coco.names"
yolo_classes_path = os.path.join(script_directory, "config", yolo_classes_file)

with open(yolo_classes_path, "r") as classes_file:
    classes = [line.strip() for line in classes_file.readlines()]

# Obter os nomes das camadas de saída
output_layer_names = yolo_net.getUnconnectedOutLayersNames()

# Carregar o vídeo
yolo_video_file = "video1.mp4"
parent_directory = os.path.join(script_directory, os.pardir, os.pardir)
yolo_video_path = os.path.join(parent_directory, "assets", yolo_video_file)

print(yolo_video_path)

video_capture = cv2.VideoCapture(yolo_video_path)

while True:
    # Capturar o próximo frame do vídeo
    ret, frame = video_capture.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Detectar objetos usando YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    yolo_outputs = yolo_net.forward(output_layer_names)

    # Informações sobre detecções
    detected_class_ids, confidences, bounding_boxes = [], [], []

    # Processar as saídas de YOLO
    for output in yolo_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Coordenadas do objeto detectado
                center_x, center_y, box_width, box_height = (detection[0] * width, detection[1] * height,
                                                            detection[2] * width, detection[3] * height)

                # Coordenadas do retângulo
                x, y = int(center_x - box_width / 2), int(center_y - box_height / 2)

                bounding_boxes.append([x, y, int(box_width), int(box_height)])
                confidences.append(float(confidence))
                detected_class_ids.append(class_id)

    # Supressão não máxima para remover detecções duplicadas
    indexes = cv2.dnn.NMSBoxes(bounding_boxes, confidences, 0.5, 0.4)

    # Desenhar caixas delimitadoras e rótulos no frame
    for i in range(len(bounding_boxes)):
        if i in indexes:
            x, y, w, h = bounding_boxes[i]
            label = str(classes[detected_class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)

            # Desenhar retângulo
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Adicionar rótulo com confiança
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Exibir o frame resultante
    cv2.imshow("YOLO Video", frame)

    # Sair quando a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
video_capture.release()
cv2.destroyAllWindows()
