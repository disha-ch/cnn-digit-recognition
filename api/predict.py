from http.server import BaseHTTPRequestHandler
import base64
import json
import os
from io import BytesIO

import numpy as np
from PIL import Image
import tensorflow as tf


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "mnist_cnn_model.keras")
model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)


def preprocess_canvas(image_b64: str):
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    raw = base64.b64decode(image_b64)
    image = Image.open(BytesIO(raw)).convert("L")
    image = image.resize((28, 28))
    array = 255 - np.array(image, dtype=np.uint8)
    array = array.astype("float32") / 255.0
    array = np.expand_dims(array, axis=(0, -1))
    return array


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length)
        payload = json.loads(body.decode("utf-8"))
        image_b64 = payload.get("image")

        if not image_b64:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Missing image data"}).encode("utf-8"))
            return

        x = preprocess_canvas(image_b64)
        preds = model.predict(x, verbose=0)[0]
        pred_class = int(np.argmax(preds))

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps(
                {
                    "predictedDigit": pred_class,
                    "probabilities": [float(v) for v in preds],
                }
            ).encode("utf-8")
        )

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok"}).encode("utf-8"))
