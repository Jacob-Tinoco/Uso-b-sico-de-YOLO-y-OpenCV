import cv2
import numpy as np
import os
from datetime import datetime

# Crear directorio para guardar resultados si no existe
output_dir = 'resultados'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Cargar YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Configurar para usar GPU si está disponible
try:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
except Exception as e:
    print(f"No se pudo configurar CUDA, usando CPU: {e}")

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Cargar las clases
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def detect_people(frame, confidence_threshold=0.5, nms_threshold=0.4):
    height, width = frame.shape[:2]
    detections = []

    # Preprocesar imagen
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Inferencia
    try:
        outs = net.forward(output_layers)
    except cv2.error as e:
        print(f"Error en inferencia: {e}")
        return frame, []

    # Listas para almacenar detecciones
    boxes = []
    confidences = []
    class_ids = []

    # Procesar detecciones
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filtrar solo personas con confianza suficiente
            if confidence > confidence_threshold and class_id == classes.index('person'):
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = max(0, int(center_x - w / 2))
                y = max(0, int(center_y - h / 2))

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicar Non-Maximum Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    # Verificar si indexes no es vacío antes de intentar aplanarlo
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            confidence = confidences[i]
            detections.append([x, y, w, h, confidence])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f'{int(confidence * 100)}%'
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar contador de personas
    num_people = len(indexes) if len(boxes) > 0 else 0
    cv2.putText(frame, f'# de personas: {num_people}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame, detections
def process_image(path):
    image = cv2.imread(path)
    if image is None:
        print('Image not found. Please enter a valid path.')
        return

    processed_image, detections = detect_people(image)
    output_path = os.path.join(output_dir, f'processed_image_{len(os.listdir(output_dir)) + 1}.jpg')
    cv2.imwrite(output_path, processed_image)
    print(f'Processed image saved as {output_path}')
    print(f'# personas detectadas: {len(detections)}')

    # Crear archivo de texto para registrar la información
    log_file_path = output_path.replace('.jpg', '.txt')
    with open(log_file_path, 'w') as log_file:
        log_file.write(f"Ejecución de la imagen el día {datetime.now().strftime('%d/%m/%Y')} a las {datetime.now().strftime('%H:%M:%S')} hrs\n")
        log_file.write(f"# de personas detectadas: {len(detections)}\n")
        for detection in detections:
            x, y, w, h, confidence = detection
            log_file.write(f"Confianza: {confidence:.2f}, Coordenadas: ({x}, {y}, {w}, {h})\n")

    cv2.imshow('Procesando imagen', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(path):
    video = cv2.VideoCapture(path)
    if not video.isOpened():
        print('Video not found. Please enter a valid path.')
        return

    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    output_path = os.path.join(output_dir, f'processed_video_{len(os.listdir(output_dir)) + 1}.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Crear archivo de texto para registrar la información
    log_file_path = output_path.replace('.mp4', '.txt')
    with open(log_file_path, 'w') as log_file:
        log_file.write(f"Ejecución del video el día {datetime.now().strftime('%d/%m/%Y')} a las {datetime.now().strftime('%H:%M:%S')} hrs\n")
        log_file.write("Iniciando registro de frames procesados\n")

        start_time = datetime.now()
        frame_number = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break

            processed_frame, detections = detect_people(frame)
            out.write(processed_frame)
            cv2.imshow('Detección de personas', processed_frame)

            # Registrar información de cada frame
            num_people = len(detections)
            for detection in detections:
                x, y, w, h, confidence = detection
                log_file.write(f"# de personas: {num_people}, Índice de confianza: {confidence:.2f}, #Frame: {frame_number}, Segundo del video: {frame_number / fps:.2f}\n")

            frame_number += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        end_time = datetime.now()
        processing_time = end_time - start_time
        log_file.write(f"\nTiempo total de procesamiento: {processing_time}\n")

    video.release()
    out.release()
    cv2.destroyAllWindows()
    print(f'Processed video saved as {output_path}')
    print(f'Log saved as {log_file_path}')

def process_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # Crear archivo de texto para registrar la información
    log_file_path = os.path.join(output_dir, f'log_camera_{len(os.listdir(output_dir)) + 1}.txt')
    with open(log_file_path, 'w') as log_file:
        log_file.write(f"Ejecución de la cámara el día {datetime.now().strftime('%d/%m/%Y')} a las {datetime.now().strftime('%H:%M:%S')} hrs\n")
        log_file.write("Iniciando registro de frames procesados\n")

        start_time = datetime.now()
        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            processed_frame, detections = detect_people(frame)
            cv2.imshow('Camera Feed', processed_frame)

            # Registrar información de cada frame
            num_people = len(detections)
            for detection in detections:
                x, y, w, h, confidence = detection
                log_file.write(f"# de personas: {num_people}, Índice de confianza: {confidence:.2f}, #Frame: {frame_number}\n")

            frame_number += 1

            if cv2.waitKey(1) == ord('q'):
                break

        end_time = datetime.now()
        processing_time = end_time - start_time
        log_file.write(f"\nTiempo total de procesamiento: {processing_time}\n")

    cap.release()
    cv2.destroyAllWindows()
    print(f'Log saved as {log_file_path}')

def main():
    while True:
        choice = input("Presiona 'i' for imagenes, 'v' para videos, 'c' para la camara web, o presiona 'q' para salir: ").lower()
        if choice == 'q':
            break
        elif choice == 'i':
            path = input("Coloca el path de la imagen: ")
            process_image(path)
        elif choice == 'v':
            path = input("Coloca el path del video: ")
            process_video(path)
        elif choice == 'c':
            process_camera()
        else:
            print("Eleccion invalida. Por favor presiona 'i', 'v', 'c', o para salir 'q'.")

if __name__ == "__main__":
    main()
