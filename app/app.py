from flask import Flask, request, render_template, jsonify
import sqlite3
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import json

app = Flask(__name__)
model = tf.keras.models.load_model('D:/PlantAI/models/my_model.h5')

# Load class indices for prediction decoding
with open('D:/PlantAI/models/class_indices.json', 'r') as json_file:
    class_indices = json.load(json_file)

def get_db_connection():
    conn = sqlite3.connect('Plants_For_a_Future_Updated.db')
    conn.row_factory = sqlite3.Row
    return conn

def prepare_image(img):
    img = img.resize((224, 224))  # Resize to the input size expected by the model
    img_array = np.array(img) / 255.0  # Normalize to 0-1
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

def decode_prediction(prediction):
    predicted_index = np.argmax(prediction[0])
    latin_name = class_indices[str(predicted_index)]  # Ensure string conversion for JSON keys
    return latin_name

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if not file:
            return render_template('index.html', error='Please upload an image file.')

        image = Image.open(io.BytesIO(file.read()))
        prepared_image = prepare_image(image)
        prediction = model.predict(prepared_image)
        latin_name = decode_prediction(prediction)

        plant_info = get_plant_info(latin_name)
        if plant_info:
            return render_template('results.html', plant=plant_info)
        else:
            return render_template('index.html', error='No details found for the predicted plant.')
    return render_template('index.html')

def get_plant_info(latin_name):
    conn = get_db_connection()
    plant = conn.execute('SELECT * FROM SpeciesList WHERE LatinName = ?', (latin_name,)).fetchone()
    conn.close()
    return plant

if __name__ == '__main__':
    app.run(debug=True)
