import numpy as np
from .recognition import RecognitionEngine
from .detection import DetectionEngine
from .common import compute_similarity
from db.vector_db import face_db


class Pipeline:
    def __init__(self, conf_threshold=0.45, similarity_threshold=0.35) -> None:
        """
        Initializes the Pipeline with face detection and recognition models.

        Args:
            conf_threshold (float): Confidence threshold for face detection.
            similarity_threshold (float): Similarity threshold for face recognition.
        """
        self.conf_threshold = conf_threshold
        self.similarity_threshold = similarity_threshold

        self.face_detector = DetectionEngine(conf_thresh=conf_threshold)
        self.face_recognizer = RecognitionEngine()

    def detect_faces(self, image, max_num=0):
        """
        Detect faces in an image.

        Args:
            image (numpy.ndarray): The input image.
            max_num (int, optional): Maximum number of faces to detect. Defaults to None (detect all faces).

        Returns:
            list: A list of detected face dictionaries, each containing:
                - bbox (list): Bounding box coordinates [x_min, y_min, x_max, y_max].
                - landmark (list): Facial landmarks.
        """
        boxes, landmarks = self.face_detector.detect(image, max_num=max_num)
        if len(boxes) == 0:
            return []

        return [
            {
                "bbox": list(map(int, box[:4])),
                "landmark": [[int(point) for point in lm] for lm in landmark]
            }
            for box, landmark in zip(boxes, landmarks)
        ]

    def detect_single_face(self, image):
        """
        Detect a single face in an image.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            dict or None: Detected face information containing:
                - bbox (list): Bounding box coordinates.
                - landmark (list): Facial landmarks.
            Returns None if no face is detected.
        """
        faces = self.detect_faces(image, max_num=1)
        return faces[0] if faces else None

    def get_faces_info(self, image):
        """
        Extract face embeddings for all detected faces.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            list: A list of detected faces with embeddings, each containing:
                - bbox (list): Bounding box coordinates.
                - landmark (list): Facial landmarks.
                - embedding (numpy.ndarray): Face embedding.
            Returns {"error": "No face detected"} if no face is found.
        """
        bboxes, landmarks = self.face_detector.detect(image)
        if len(bboxes) == 0:
            return {"error": "No face detected"}

        faces_info = []
        for bbox, landmark in zip(bboxes, landmarks):
            emebdding = self.face_recognizer.get_embedding(image, landmark)
            faces_info.append({
                "bbox": bbox,
                "landmark": landmark,
                "embedding": emebdding,
            })

        return faces_info

    def get_single_face_info(self, image):
        """
        Extract an info for a single detected face.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            dict or None: A dictionary containing:
                - bbox (list): Bounding box coordinates.
                - landmark (list): Facial landmarks.
                - embedding (numpy.ndarray): Face embedding.
            Returns None if no face is detected.
        """
        bbox, landmark = self.face_detector.detect(image, max_num=1)
        if len(bbox) == 0:
            return None


        embedding = self.face_recognizer.get_embedding(image, landmark[0])

        return {
            "bbox": bbox[0],
            "landmark": landmark[0],
            "embedding": embedding.tolist(),
        }

    def recognize_faces(self, image):
        """
        Recognize faces in an image by comparing embeddings with a face database.

        Args:
            image (numpy.ndarray): The input image.


        Returns:
            dict: A dictionary containing recognized faces with:
                - bbox (list): Bounding box coordinates.
                - landmark (list): Facial landmarks.
                - face_id (str): ID of the most similar face.
                - name (str): Name of the most similar face.
                - similarity (float): Similarity score.
            Returns {"error": "No face detected"} if no face is found.
        """
        detected_faces = self.get_faces_info(image)
        if "error" in detected_faces:
            return detected_faces

        recognized_faces = []
        for face in detected_faces:
            t = {}
            best_match = {"id": None, "name": "Unknown", "similarity": 0}

            # Ensure embedding is a Python list before passing it to search_face
            face_id, name, similarity = face_db.search_face(face["embedding"], self.similarity_threshold)

            best_match["id"] = face_id
            best_match["name"] = name
            best_match["similarity"] = float(similarity)
            
            t["name"] = name
            t["face_id"] = face_id #TODO: fix and update it, think of how many logics should be inside pipeline

            # face.update(
            #     {
            #         # "face_id": best_match["id"],
            #         "name": best_match["name"],
            #         # "similarity": best_match["similarity"]
            #     }
            # )
            recognized_faces.append(t)

        return recognized_faces

    def add_face(self, image, name):
        """
        Add a new face embedding to the face database.

        Args:
            image (numpy.ndarray): The input image.
            name (str): Name of the person in the image.

        Returns:
            dict: A dictionary containing the status message.
        """
        face_info = self.get_single_face_info(image)
        if not face_info:
            return {"error": "No face detected"}

        status = face_db.add_face(face_info["embedding"], name)
        return status

    def compute_similarity(self, emb1, emb2):
        """
        Compute cosine similarity between two face embeddings.

        Args:
            emb1 (numpy.ndarray): First face embedding.
            emb2 (numpy.ndarray): Second face embedding.

        Returns:
            float: The similarity score (1 means identical, 0 means completely different).
        """
        return compute_similarity(emb1, emb2)
