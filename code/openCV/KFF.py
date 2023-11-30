import cv2

def initialize_tracker():
    return cv2.TrackerKCF_create()

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
    # Inicializar o rastreador KCF
    tracker = initialize_tracker()

    # Inicializar o vídeo
    video_path = "../../assets/video2.mp4"
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
            (x, y, w, h) = tuple(map(int, bbox))
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
