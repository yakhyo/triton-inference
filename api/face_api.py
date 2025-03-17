
import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile

from db.vector_db import face_db
from inference.detection import DetectionEngine
from inference.recognition import RecognitionEngine

router = APIRouter()

# Initialize Face Detection & Recognition Models
face_detector = TritonRetinaFace(conf_thresh=0.45)
face_recognizer = TritonFaceEngine(model_name="recognition")


@router.post("/detect/")
async def detect_faces(file: UploadFile = File(...)):
    """
    Detects faces in an image and returns bounding boxes.
    """
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    # Detect faces
    boxes, _ = face_detector.detect(img)
    faces_info = [{"bbox": list(map(int, box[:4]))} for box in boxes]

    return {"faces": faces_info}


@router.post("/recognize/")
async def recognize_faces(file: UploadFile = File(...)):
    """
    Recognizes faces in an image by comparing embeddings with FAISS DB.
    Returns name if found, otherwise 'Unknown'.
    """
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    # Detect faces & get landmarks
    boxes, landmarks = face_detector.detect(img)
    if len(boxes) == 0:
        return {"error": "No face detected"}

    faces_info = []
    for landmark in landmarks:
        # Extract face embedding
        embedding = face_recognizer.get_embedding(img, landmark)

        # Search for the face in FAISS
        name, similarity = face_db.search_face(embedding)

        faces_info.append({
            "bbox": list(map(int, boxes[0][:4])),
            "face_id": name,
            "similarity": similarity
        })

    return {"faces": faces_info}


@router.post("/add_face/")
async def add_face(file: UploadFile = File(...), name: str = "Unknown"):
    """
    Adds a new face embedding to the FAISS database.
    """
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    # Detect faces & get landmarks
    boxes, landmarks = face_detector.detect(img)
    if len(boxes) == 0:
        return {"error": "No face detected"}

    # Get face embedding
    embedding = face_recognizer.get_embedding(img, landmarks[0])

    # Store embedding in FAISS
    face_db.add_face(embedding, name)

    return {"message": f"Face {name} added to database"}
