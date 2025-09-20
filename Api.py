from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2
import base64

# model_name = "BestModelCoins.pt"
model_name = "C:/Users/lenovo/OneDrive/Desktop/Money Detection/best_money_model.pt"
model = YOLO(model_name)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/prediction")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model.predict(img)
    result = results[0]

    data = []
    for box in result.boxes:
        clss = result.names
        labels = clss[int(box.cls)]
        data.append({"label": labels})

    itms = []
    for d in data:
       label = d["label"]
       value = "".join([c for c in label if c.isdigit()]) 
       if value.isdigit():
          itms.append(int(value))

    total_n = sum(itms)

    ann_image = result.plot()
    _, buffer = cv2.imencode(".jpg", ann_image)
    ann_image_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "labels": data,
        "annotated_image": ann_image_base64,
        "total_money": total_n
    }
