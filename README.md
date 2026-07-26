# CNN Digit Recognizer

This project was built for a college alumni interaction session where CNNs were explained through a practical, hands-on example.

The idea is simple:

1. Train a small Convolutional Neural Network on the MNIST handwritten digit dataset.
2. Save the trained model.
3. Load the model into a Streamlit app.
4. Let the user draw a digit and see the prediction instantly.

## Project Structure

- [`app.py`](./app.py) - Streamlit UI for drawing digits and running predictions.
- [`data.py`](./data.py) - Trains the CNN on MNIST and saves the model file.
- [`mnist_cnn_model.keras`](./mnist_cnn_model.keras) - Saved TensorFlow model used by the app.
- [`requirements.txt`](./requirements.txt) - Python dependencies for training and the demo app.

## What This Project Does

The app shows a drawable canvas. When you sketch a digit from `0` to `9`, the image is:

1. Captured from the canvas.
2. Converted into a grayscale `28 x 28` image.
3. Normalized to match MNIST training input.
4. Passed into the trained CNN.
5. Displayed as a predicted digit with class probabilities.

This makes the CNN easy to explain in a live session because you can show both:

- how the model was trained, and
- how it is used in a real interactive demo.

## How It Works

### 1. Training the model

[`data.py`](./data.py) loads the MNIST dataset from TensorFlow, normalizes the pixel values, reshapes the data to include a channel dimension, and trains a compact CNN:

- `Conv2D` learns edge and stroke features.
- `MaxPooling2D` reduces spatial size while keeping important patterns.
- `Flatten` converts feature maps into a vector.
- `Dense` layers learn the final classification boundaries.
- `Dropout` helps reduce overfitting.

After training, the model is saved as [`mnist_cnn_model.keras`](./mnist_cnn_model.keras).

### 2. Running the demo app

[`app.py`](./app.py) loads the saved model and starts a Streamlit interface.

When you draw on the canvas:

- the sketch is inverted so black strokes become the kind of input MNIST expects,
- resized to `28 x 28`,
- converted to a single grayscale channel,
- normalized to the `0-1` range,
- and fed into the model for prediction.

The app then shows:

- the most likely digit, and
- a bar chart of prediction probabilities for all 10 classes.

## Requirements

Install the dependencies from [`requirements.txt`](./requirements.txt):

```bash
pip install -r requirements.txt
```

## Run the App

Start the Streamlit demo:

```bash
streamlit run app.py
```

Then draw a digit in the browser and watch the prediction update.

## Re-train the Model

If you want to train the model again from scratch:

```bash
python data.py
```

This will:

- download MNIST,
- train the CNN,
- and overwrite `mnist_cnn_model.keras` with a freshly saved model.

## Notes

- The app expects the saved model file to be present in the project folder.
- The model was built for a simple educational demo, so the code is intentionally compact and easy to explain.
- The project is a good example of the full ML workflow: data loading, model training, model saving, and model inference in a UI.

## Tech Stack

- TensorFlow
- Keras
- Streamlit
- `streamlit-drawable-canvas`
- NumPy
- Pillow

