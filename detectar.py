from ultralytics import YOLO
import cv2

# Cargar modelos
modelo_general = YOLO('yolov8n.pt')  # Modelo general (personas, autos, etc.)
modelo_principal = YOLO('C:/Users/paufu/runs/detect/yolov8_principal/weights/best.pt')
modelo_armas = YOLO('C:/Users/paufu/runs/detect/arma_yolov8_estable/weights/best.pt')  # Modelo armas

# Captura de video desde webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ejecutar detección con los tres modelos
    resultados_generales = modelo_general(frame)[0]
    resultados_principal = modelo_principal(frame)[0]
    resultados_armas = modelo_armas(frame)[0]

    # Dibujar resultados del modelo general (verde, etiquetas originales)
    for r in resultados_generales.boxes.data.cpu().numpy():
        x1, y1, x2, y2, score, cls = r
        if score < 0.5:
            continue
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = modelo_general.names[int(cls)]
        color = (0, 255, 0)  # Verde
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{label} {score:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Dibujar resultados del modelo principal (azul, etiqueta personalizada: Warning)
    for r in resultados_principal.boxes.data.cpu().numpy():
        x1, y1, x2, y2, score, cls = r
        if score < 0.5:
            continue
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = "Elementos reconocidos"
        color = (255, 0, 0)  # Azul
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{label} {score:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Dibujar resultados del modelo de armas (rojo, etiqueta personalizada: Objeto Sospechoso)
    for r in resultados_armas.boxes.data.cpu().numpy():
        x1, y1, x2, y2, score, cls = r
        if score < 0.5:
            continue
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = "Objeto Sospechoso"
        color = (0, 0, 255)  # Rojo
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{label} {score:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Mostrar resultados
    cv2.imshow('Detección en vivo', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
