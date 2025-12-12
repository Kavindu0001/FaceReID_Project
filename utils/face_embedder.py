import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

class FaceEmbedder:
    def __init__(self, model_path="reid_model.h5"):
        self.model = load_model(model_path)

    def preprocess(self, face):
        face = Image.fromarray(face).resize((128, 128))
        face = np.asarray(face).astype("float32") / 255.0
        return np.expand_dims(face, axis=0)

    def get_embedding(self, face):
        face = self.preprocess(face)
        embedding = self.model.predict(face)[0]
        return embedding
