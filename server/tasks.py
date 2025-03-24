import cv2
import numpy as np
from celery import exceptions
from server import app
from inference.pipeline import Pipeline


@app.task(ignore_result=False, bind=True, base=Pipeline, name="recognize_face_task")
def recognize(self, file: bytes):
    """
    Celery task to recognize faces in an image by comparing embeddings with the FAISS DB.
    """
    try:
        image_buffer = np.frombuffer(file, np.uint8)
        image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
        status = self.recognize_faces(image)
        return {
            "status": "success",
            "result": status
        }
    except Exception as e:
        try:
            self.retry(countdown=2)
        except exceptions.MaxRetriesExceededError:
            return {
                "status": "failed",
                "error": str(e)
            }


@app.task(ignore_result=False, bind=True, base=Pipeline, name="add_face_task")
def add_face(self, file: bytes, name: str, user_id: str):
    """
    Celery task to add a new face embedding to the FAISS DB.
    """
    try:
        image_buffer = np.frombuffer(file, np.uint8)
        image = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
        status = self.add_face(image, name, user_id)
        return {
            "status": "success",
            "result": status
        }
    except Exception as e:
        try:
            self.retry(countdown=2)
        except exceptions.MaxRetriesExceededError:
            return {
                "status": "failed",
                "error": str(e)
            }
