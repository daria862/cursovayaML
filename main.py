import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO  # Вместо tensorflow

app = FastAPI(title="Кожные заболевания")

os.makedirs("static/uploads", exist_ok=True)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Настройки
CLASSES = ['carcinoma', 'dermatitis', 'eczema', 'melanoma', 'psoriasis']
MODEL_PATH = "skin_medical_project/detection_results/weights/best.pt"

model = YOLO(MODEL_PATH)

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "classes": CLASSES
    })


@app.post("/detect")
async def detect_disease(request: Request, file: UploadFile = File(...)):
    temp_path = f"static/uploads/temp_{file.filename}"
    contents = await file.read()
    with open(temp_path, "wb") as f:
        f.write(contents)

    results = model.predict(temp_path, conf=0.3)
    result = results[0]  # Результат для одного фото

    image = cv2.imread(temp_path)
    disease_count = len(result.boxes)  # СЧЕТЧИК
    detected_items = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        coords = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]

        class_name = CLASSES[class_id]

        detected_items.append({
            "class_name": class_name,
            "confidence": confidence,
            "box": [int(c) for c in coords]
        })

        xmin, ymin, xmax, ymax = map(int, coords)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)

        label = f"{class_name} ({confidence:.2f})"
        cv2.putText(image, label, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(image, f"Total Detected: {disease_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(image, f"Total Detected: {disease_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1, cv2.LINE_AA)

    output_filename = f"detected_{file.filename}"
    output_path = f"static/uploads/{output_filename}"
    cv2.imwrite(output_path, image)

    if os.path.exists(temp_path):
        os.remove(temp_path)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "detected_items": detected_items,
        "disease_count": disease_count,
        "image_url": f"/static/uploads/{output_filename}",
        "all_predictions": [
            {"name": name, "prob": 0.0} for name in CLASSES
        ]
    })


if __name__ == "__main__":
    import uvicorn

    # Запуск сервера
    uvicorn.run(app, host="127.0.0.1", port=8000)

