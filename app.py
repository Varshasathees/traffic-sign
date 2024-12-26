from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

app = Flask(__name__)

# Load the saved model
model = load_model("C:/Users/hp/Downloads/model.h5")

# Define the class names
classes = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing veh over 3.5 tons',
    11: 'Right-of-way at intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Veh > 3.5 tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve left',
    20: 'Dangerous curve right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End speed + passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End no passing veh > 3.5 tons'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        image = Image.open(file.stream)
        image = image.resize((30, 30))
        image = np.array(image)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        predictions = model.predict(image)
        class_index = np.argmax(predictions)
        class_name = classes[class_index]
        confidence = np.max(predictions)

        # Save the image with the prediction text
        image = cv2.cvtColor(np.array(image[0] * 255, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        cv2.putText(image, f"{class_name} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        result_image_path = os.path.join('static', 'result.jpg')
        cv2.imwrite(result_image_path, image)

        formatted_confidence = f"{confidence:.2f}"

        return render_template('result.html', class_name=class_name, confidence=formatted_confidence, image_path=result_image_path)

if __name__ == '__main__':
    app.run(debug=True)
