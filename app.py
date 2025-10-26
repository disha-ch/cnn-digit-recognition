import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="🧠 CNN Digit Classifier", page_icon="🖊️", layout="wide")

# Load Model
model = tf.keras.models.load_model("mnist_cnn_model.keras", safe_mode=False)

st.title("🧠 Interactive CNN Digit Recognizer (0 ‑ 9)")
st.write("Draw a digit between **0 to 9** and see how the CNN predicts it!")

col1, col2 = st.columns(2)
with col1:
    canvas = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=15,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    if canvas.image_data is not None:
        img = Image.fromarray((255 - canvas.image_data[:, :, 0]).astype('uint8')).resize((28,28)).convert('L')
        x = np.expand_dims(np.array(img), axis=(0, -1)) / 255.0
        preds = model.predict(x)
        pred_class = np.argmax(preds)
        st.metric(label="Predicted Digit", value=str(pred_class))
        st.bar_chart(preds[0])

st.caption("Model trained with TensorFlow 2.20 using Apple Metal backend.")
