# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import cv2
import numpy as np
import onnxruntime as ort
import tritonclient.http as httpclient
from typing import Optional, Tuple

import uniface

from common import compute_similarity, face_alignment

TRITON_SERVER_URL = "localhost:8000"

class TritonFaceEngine:
    """
    Face recognition model using Triton Inference Server.
    """

    def __init__(self, model_name="recognition"):
        """
        Initializes the Triton client for the face recognition model.
        Args:
            model_name (str): Triton model name.
        """
        self.client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)
        self.model_name = model_name
        self.input_size = (112, 112)  # Expected input size for face recognition

        # Verify if model is ready
        if not self.client.is_model_ready(self.model_name):
            raise RuntimeError(f"Triton model '{self.model_name}' is not ready!")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image: align, resize, normalize.
        Args:
            image (np.ndarray): Input image in BGR format.
        Returns:
            np.ndarray: Preprocessed image tensor for inference.
        """
        image = cv2.resize(image, self.input_size)  # Resize to (112, 112)
        blob = cv2.dnn.blobFromImage(
            image, scalefactor=1.0 / 127.5, size=self.input_size,
            mean=(127.5, 127.5, 127.5), swapRB=True  # Convert BGR to RGB
        )
        return blob

    def get_embedding(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Extracts face embedding from an aligned image.
        Args:
            image (np.ndarray): Face image (BGR format).
            landmarks (np.ndarray): Facial landmarks (5 points for alignment).
        Returns:
            np.ndarray: 512-dimensional face embedding.
        """
        aligned_face = face_alignment(image, landmarks)  # Align face
        input_tensor = self.preprocess(aligned_face)  # Convert to tensor

        # Create Triton input tensor
        inputs = httpclient.InferInput("input", input_tensor.shape, "FP32")
        inputs.set_data_from_numpy(input_tensor)

        # Request output tensor
        outputs = [httpclient.InferRequestedOutput("output")]

        # Perform inference
        response = self.client.infer(model_name=self.model_name, inputs=[inputs], outputs=outputs)

        # Retrieve embeddings
        embedding = response.as_numpy("output")
        return embedding.flatten()  # Return as a 1D vector

def compare_faces(
        model: TritonFaceEngine,
        img1: np.ndarray,
        landmarks1: np.ndarray,
        img2: np.ndarray,
        landmarks2: np.ndarray,
        threshold: float = 0.35
) -> tuple:
    """
    Compares two face images and determines if they belong to the same person.
    Args:
        model (TritonFaceEngine): The face recognition model instance.
        img1 (np.ndarray): First face image (BGR format).
        landmarks1 (np.ndarray): Facial landmarks for img1.
        img2 (np.ndarray): Second face image (BGR format).
        landmarks2 (np.ndarray): Facial landmarks for img2.
        threshold (float): Similarity threshold for face matching.
    Returns:
        tuple[float, bool]: Similarity score and match result (True/False).
    """
    feat1 = model.get_embedding(img1, landmarks1)
    feat2 = model.get_embedding(img2, landmarks2)
    similarity = compute_similarity(feat1, feat2)
    is_match = similarity > threshold
    return similarity, is_match

# Example usage
if __name__ == "__main__":
    import uniface
    import warnings
    warnings.filterwarnings("ignore")

    # Initialize face detection and recognition models
    uniface_inference = uniface.RetinaFace(model="retinaface_mnet_v2", conf_thresh=0.45)
    face_recognizer = TritonFaceEngine(model_name="recognition")

    # Load images
    img1 = cv2.imread("assets/faces/1_01.jpg")
    img2 = cv2.imread("assets/faces/1_02.jpg")

    # Detect faces and get landmarks
    boxes, landmarks = uniface_inference.detect(img1)
    landmarks1 = landmarks[0]  # Get first face landmarks

    boxes, landmarks = uniface_inference.detect(img2)
    landmarks2 = landmarks[0]  # Get first face landmarks

    # Compare two face images
    similarity, is_same = compare_faces(face_recognizer, img1, landmarks1, img2, landmarks2, threshold=0.30)

    print(f"Similarity: {similarity:.4f} - {'Same person' if is_same else 'Different person'}")