from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="proyecto/datasetArmas/dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=4,  # MÁS PEQUEÑO = MENOS CARGA PARA LA GPU
        device=0,  # GPU (RTX 2050)
        workers=2,  # Menos procesos paralelos
        cache=False,  # No mantener todo en RAM
        name="arma_yolov8_principal"
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
