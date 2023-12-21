# main.py

from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import requests

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('modelindonesiafood.h5')

# Define label names
label_names = {0: 'gudeg', 1: 'nasi goreng', 2: 'pempek', 3: 'rawon', 4: 'rendang', 5: 'sate'}

# Function to preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path).resize((299, 299))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to get food data from Firebase API
def get_food_data():
    firebase_api_url = 'https://capstone-project-ch2-ps342-default-rtdb.asia-southeast1.firebasedatabase.app/makanan.json'
    response = requests.get(firebase_api_url)
    data = response.json()
    return data

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "no file"})

        file = request.files['file']
        img_array = preprocess_image(file)

        # Make predictions
        predictions = model.predict(img_array)

        # Get the predicted class name
        predicted_class = label_names[np.argmax(predictions[0])]

        # Get the confidence percentage
        confidence_percentage = str(np.max(predictions))

        # Get food data from Firebase API
        food_data = get_food_data()

        # Return the results including food data
        result = {
            "prediction": predicted_class,
            "confidence": confidence_percentage,
            "food_data": food_data
        }

        return jsonify(result)

    elif request.method == 'GET':
        # Handle GET requests, if needed
        return jsonify({"message": "Send a POST request with an image file."})

    else:
        return jsonify({"error": "Method not allowed"}), 405

if __name__ == '__main__':
    app.run(debug=True)
