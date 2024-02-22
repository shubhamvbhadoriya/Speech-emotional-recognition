from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the saved model
model = load_model('saved_model_80-20.hdf5')

# Function to preprocess input speech data
def preprocess_input(speech_data):
    # Perform any necessary preprocessing steps
    # For example, convert to the required input shape (23, 1)
    preprocessed_data = np.expand_dims(speech_data, axis=-1)  # Assuming speech_data has shape (23,)
    return preprocessed_data

# Function to make predictions
def predict_emotion(speech_data):
    preprocessed_data = preprocess_input(speech_data)
    predictions = model.predict(np.array([preprocessed_data]))
    predicted_class = np.argmax(predictions)
    # Map predicted class to emotion labels
    emotion_labels = ["Happy", "Sad", "Angry", "Calm", "Neutral", "Disgust", "Fear"]
    predicted_emotion = emotion_labels[predicted_class]
    return predicted_emotion

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', message='No selected file')
    if file:
        # Assuming audio file is processed and converted to feature array
        # Replace the following line with your actual speech data extraction/preprocessing code
        speech_data = np.random.rand(23)  # Example data with shape (23,)
        predicted_emotion = predict_emotion(speech_data)
        return render_template('result.html', predicted_emotion=predicted_emotion)

if __name__ == '__main__':
    app.run(debug=True)
