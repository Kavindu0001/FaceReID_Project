import numpy as np

class ReidMatcher:
    def __init__(self, threshold=0.6):
        self.threshold = threshold

    def cosine_distance(self, emb1, emb2):
        emb1 = np.asarray(emb1)
        emb2 = np.asarray(emb2)
        return 1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def find_best_match(self, embedding, database):
        best_id = None
        best_score = 999

        for passenger in database["passengers"]:
            db_emb = passenger["embedding"]
            score = self.cosine_distance(embedding, db_emb)

            if score < best_score and score < self.threshold:
                best_score = score
                best_id = passenger["id"]

        return best_id
