import cv2
import time
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.encoders import jsonable_encoder


from .inference.pipeline import Pipeline

# Initialize FastAPI app
app = FastAPI()

# Initialize Face Detection & Recognition Pipeline
pipeline = Pipeline(conf_threshold=0.45, similarity_threshold=0.35)


@app.get("/")
async def root():
    return {"message": "FastAPI is running!"}


@app.post("/detect/")
async def detect_faces(file: UploadFile = File(...)):
    """Detects faces in an image and returns bounding boxes and landmarks."""
    image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    faces = pipeline.detect_faces(image)

    return {"faces": faces}


@app.post("/recognize/")
async def recognize_faces(file: UploadFile = File(...)):
    """Recognizes faces in an image by comparing embeddings with FAISS DB."""
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    recognized_faces = pipeline.recognize_faces(image)
    return jsonable_encoder({"faces": recognized_faces})


@app.post("/add_face/")
async def add_face(
    file: UploadFile = File(...),
    name: str = Form(...),
    user_id: str = Form(...)
):
    """Adds a new face embedding to the FAISS database."""

    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    status = pipeline.add_face(image, name, user_id)

    return status


@app.post("/compare_faces/")
async def compare_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """Compares two faces to determine if they belong to the same person."""
    image1 = cv2.imdecode(np.frombuffer(await file1.read(), np.uint8), cv2.IMREAD_COLOR)
    image2 = cv2.imdecode(np.frombuffer(await file2.read(), np.uint8), cv2.IMREAD_COLOR)

    face1 = pipeline.get_single_face_info(image1)
    face2 = pipeline.get_single_face_info(image2)

    if not face1 or not face2:
        return {"error": "One or both faces not detected"}

    similarity = pipeline.compute_similarity(face1["embedding"], face2["embedding"])
    is_match = bool(similarity > pipeline.similarity_threshold)

    return {
        "similarity": float(similarity),
        "match": is_match,
        "message": "Same person" if is_match else "Different person",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8008)
