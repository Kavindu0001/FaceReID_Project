import tensorflow as tf
import numpy as np


class FaceEmbedder:
    def __init__(self, model_path="reid_model.h5"):
        base_model = tf.keras.models.load_model(model_path)

        # Force model build (important for Sequential)
        base_model.build((None, 128, 128, 3))

        # Use Dense(128) layer output (embedding layer)
        self.model = tf.keras.Model(
            inputs=base_model.inputs,
            outputs=base_model.layers[-2].output
        )

    def get_embedding(self, face_img):
        """
        face_img: (128, 128, 3)
        """
        face_img = face_img.astype("float32") / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        embedding = self.model.predict(face_img, verbose=0)[0]
        embedding = embedding / np.linalg.norm(embedding)

        return embedding