from fastapi import FastAPI, File, UploadFile, status, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import numpy as np
import cv2
import os
import time
from functools import wraps
from keras.models import load_model
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import supervision as sv

app = FastAPI()

# Allow CORS
origins = [
    "http://localhost:3000",  # React
    "http://localhost:8080",  # Vue.js
    "http://localhost:8002",  # Angular
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

imageDirectory = "uploadedFile"  # store uploaded image in this folder

if not os.path.exists(imageDirectory):
    os.makedirs(imageDirectory)

model = YOLO("best.pt")


 

def rate_limited(max_calls: int, time_frame: int):
    def decorator(func):
        calls = []

        @wraps(func)
        async def wrapper(*args, **kwargs):
            now = time.time()
            calls_in_time_frame = [call for call in calls if call > now - time_frame]
            if len(calls_in_time_frame) >= max_calls:
                raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded!")
            calls.append(now)
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def objectDetector(filename):
    try:
        print("------------------------", filename)
        frame = cv2.imread("uploadedFile/" + filename)
        if frame is None:
            raise ValueError("Image not loaded correctly")

        start_time = time.time()
        results = model(frame, conf=0.7, save=True)[0]
        inference_time = time.time() - start_time

        print(f"Speed: {inference_time * 1000:.1f}ms inference per image at shape {frame.shape}")

        detections = sv.Detections.from_ultralytics(results)
        results = detections.with_nms(threshold=0.5)

        if len(results) == 0:
            return {"status": "no detections"}

        class_name = ""
        confidence = 0.0
        bbox_list = []
        bbox = results.xyxy  # get box coordinates in (top, left, bottom, right) format
        bbox_class = results.class_id  # cls, (N, )

        for r in results:
            frame = np.ascontiguousarray(frame)
            annotator = Annotator(frame)
            box = r[0]
            confidence = r[2]
            class_id = r[3]

            class_name = model.names[int(class_id)]
            annotator.box_label(box, class_name + ' ' + str(int(confidence * 100)) + '%', color=(0, 0, 255), txt_color=(255, 255, 255))
            frame = annotator.result()

            bbox_list.append({
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3]),
                "confidence": float(confidence),
                "class_name": class_name
            })

        jsonResult = {
            "status": "error"
        }

        cv2.imwrite("result.png", frame)
        if class_name is not None and class_name != "":
            jsonResult = {
                "status": "successful",
                "bbox_list": bbox_list,
                "class_name": class_name,
                "confidence": float(confidence),
                "path": "result.png",
                "inference_time_ms": inference_time * 1000
            }

        return jsonResult
    except Exception as e:
        print("Error during object detection:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
@rate_limited(max_calls=100, time_frame=60)  # decorator to limit requests
async def index():
    return {"message": "Hello World"}

@app.post("/upload")
async def uploadFile(file: UploadFile = File(...)):
    try:
        file.filename = f"{uuid.uuid4()}.jpg"
        contents = await file.read()

        # Save the file
        with open(f"{imageDirectory}/{file.filename}", "wb") as f:
            f.write(contents)

        detectionResult = objectDetector(file.filename)
        print("============================", detectionResult)
        return JSONResponse(detectionResult)
    except Exception as e:
        print("Error during file upload:", str(e))
        raise HTTPException(status_code=500, detail="File upload failed")


@app.get("/detectedImage")
async def showImage():
    try:
        if os.path.exists("result.png"):
            imagePath = "result.png"
            return FileResponse(imagePath)
        else:
            return JSONResponse({"status": "error"})
    except Exception as e:
        print("Error during image retrieval:", str(e))
        raise HTTPException(status_code=500, detail="Image retrieval failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="info")
