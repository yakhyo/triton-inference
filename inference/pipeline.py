import numpy as np
from .recognition import DetectionEngine
from .detection import RecognitionEngine


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

    def detect_faces(self, image, max_num=None):
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
            {"bbox": list(map(int, box[:4])), "landmark": landmark.tolist()}
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

    def get_face_embeddings(self, image):
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
        faces = self.detect_faces(image)
        if not faces:
            return {"error": "No face detected"}

        return [
            {
                "bbox": face["bbox"],
                "landmark": face["landmark"],
                "embedding": self.face_recognizer.get_embedding(image, face["landmark"]),
            }
            for face in faces
        ]

    def get_single_face_embedding(self, image):
        """
        Extract an embedding for a single detected face.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            dict or None: A dictionary containing:
                - bbox (list): Bounding box coordinates.
                - landmark (list): Facial landmarks.
                - embedding (numpy.ndarray): Face embedding.
            Returns None if no face is detected.
        """
        face = self.detect_single_face(image)
        if not face:
            return None

        return {
            "bbox": face["bbox"],
            "landmark": face["landmark"],
            "embedding": self.face_recognizer.get_embedding(image, face["landmark"]),
        }

    def recognize_faces(self, image, database):
        """
        Recognize faces in an image by comparing embeddings with a face database.

        Args:
            image (numpy.ndarray): The input image.
            database (list of dict): A list of known faces in the format:
                [{"id": str, "name": str, "embedding": numpy.ndarray}, ...]

        Returns:
            dict: A dictionary containing recognized faces with:
                - bbox (list): Bounding box coordinates.
                - landmark (list): Facial landmarks.
                - face_id (str): ID of the most similar face.
                - name (str): Name of the most similar face.
                - similarity (float): Similarity score.
            Returns {"error": "No face detected"} if no face is found.
        """
        detected_faces = self.get_face_embeddings(image)
        if "error" in detected_faces:
            return detected_faces

        recognized_faces = []
        for face in detected_faces:
            best_match = {"id": None, "name": "Unknown", "similarity": 0}

            for entry in database:
                similarity = self.compute_similarity(face["embedding"], entry["embedding"])
                if similarity > self.similarity_threshold and similarity > best_match["similarity"]:
                    best_match = {"id": entry["id"], "name": entry["name"], "similarity": similarity}

            face.update(
                {
                    "face_id": best_match["id"],
                    "name": best_match["name"],
                    "similarity": best_match["similarity"]
                }
            )
            recognized_faces.append(face)

        return {"faces": recognized_faces}

    def compute_similarity(self, emb1, emb2):
        """
        Compute cosine similarity between two face embeddings.

        Args:
            emb1 (numpy.ndarray): First face embedding.
            emb2 (numpy.ndarray): Second face embedding.

        Returns:
            float: The similarity score (1 means identical, 0 means completely different).
        """
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
