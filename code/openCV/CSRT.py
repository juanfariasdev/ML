import cv2
import os



script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.join(script_directory, os.pardir, os.pardir)

def initialize_tracker():
    return cv2.TrackerCSRT_create()

def initialize_video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo")
        exit()
    return cap

def read_first_frame(cap):
    ret, frame = cap.read()
    if not ret:
        print("Erro ao ler o primeiro frame do vídeo")
        exit()
    return frame

def select_roi(frame):
    bbox = cv2.selectROI(frame, False)
    return bbox

def main():
    # Inicializar o rastreador CSRT
    tracker = initialize_tracker()

    # Inicializar o vídeo
    video_filename = "video1.mp4"
    video_path = os.path.join(parent_directory, "assets", video_filename)
    print(video_path)
    cap = initialize_video_capture(video_path)

    # Ler o primeiro frame
    frame = read_first_frame(cap)

    # Selecionar a região de interesse (ROI) para rastrear
    bbox = select_roi(frame)
    tracker.init(frame, bbox)

    while True:
        # Ler o próximo frame
        ret, frame = cap.read()
        if not ret:
            break

        # Atualizar o rastreador com o novo frame
        success, bbox = tracker.update(frame)

        # Desenhar a bounding box no frame
        if success:
            (x, y, w, h) = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Exibir o frame resultante
        cv2.imshow("Tracking", frame)

        # Sair quando a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
