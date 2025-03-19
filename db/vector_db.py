import faiss
import numpy as np
import json
import os

DB_PATH = "db/face_index.faiss"
METADATA_PATH = "db/face_metadata.json"
N_DIMENSIONS = 512  # Face embedding size


class FaceVectorDB:
    def __init__(self):
        """Initialize FAISS vector database and metadata."""
        self.index = faiss.IndexFlatIP(N_DIMENSIONS)
        self.metadata = {}

        if os.path.exists(DB_PATH):
            self.load_db()

    def add_face(self, embedding, name, user_id):
        """Add a new face embedding and return a structured response with error handling."""
        if embedding is None or not isinstance(embedding, (list, np.ndarray)) or len(embedding) != N_DIMENSIONS:
            return {
                "success": False,
                "message": "Invalid face embedding. Ensure the image contains a detectable face."
            }

        try:
            embedding = np.array([embedding], dtype=np.float32)
            # Normalize the embedding for cosine similarity
            faiss.normalize_L2(embedding)

            self.index.add(embedding)
            face_id = str(self.index.ntotal - 1)  # FAISS index starts at 0
            self.metadata[face_id] = {
                "name": name,
                "user_id": user_id
            }
            self.save_db()

            return {
                "success": True,
                "message": f"Face for `'{name}'` added successfully!",
                "face_id": face_id,
                "user_id": user_id,
                "total_faces": self.index.ntotal
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to add face due to an error: {str(e)}"
            }

    def search_face(self, embedding, threshold=0.5):
        """Search for a face embedding in the database."""
        if self.index.ntotal == 0:
            return {
                "success": False,
                "message": "No faces in the database.",
                "matched": False,
                "name": "Unknown",
                "user_id": None,
                "face_id": None,
                "similarity": 1.0
            }

        embedding = np.array([embedding], dtype=np.float32)

        # Normalize the query embedding
        faiss.normalize_L2(embedding)

        similarities, indices = self.index.search(embedding, 1)

        # If the similarity score is lower than the threshold, return "Unknown"
        if similarities[0][0] < threshold:
            return {
                "success": True,
                "message": "No matching face found above the threshold.",
                "matched": False,
                "name": "Unknown",
                "user_id": None,
                "face_id": None,
                "similarity": float(similarities[0][0])
            }

        face_id = str(indices[0][0])
        metadata = self.metadata.get(face_id, {"name": "Unknown", "user_id": None})

        return {
            "success": True,
            "message": f"Match found for '{metadata['name']}' with similarity {similarities[0][0]:.2f}",
            "matched": True,
            "name": metadata["name"],
            "user_id": metadata["user_id"],
            "face_id": face_id,
            "similarity": float(similarities[0][0])
        }

    def save_db(self):
        """Save FAISS index and metadata."""
        faiss.write_index(self.index, DB_PATH)
        with open(METADATA_PATH, "w") as f:
            json.dump(self.metadata, f)

    def load_db(self):
        """Load FAISS index and metadata."""
        if os.path.exists(DB_PATH):
            self.index = faiss.read_index(DB_PATH)
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r") as f:
                self.metadata = json.load(f)


face_db = FaceVectorDB()
