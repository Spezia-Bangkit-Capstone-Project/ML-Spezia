from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
from skimage import transform, io
import numpy as np
import os
from keras.models import load_model
from PIL import Image
from datetime import datetime
from keras.preprocessing import image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = load_model('spezia_model.h5')

def load_image_into_numpy_array(image):
    return np.array(image.getdata()).astype(np.uint8)

@app.route('/')
def main():
    return 'Welcome to Spezia ML Team API for predict many spices'

@app.route('/predict', methods=['POST'])
def recognize_image():
    if 'file' not in request.files:
        resp = jsonify({'message': 'No image in the request'})
        resp.status_code = 400
        return resp
    files = request.files['file']
    filename = files.filename
    errors = {}
    success = False

    if files and allowed_file(filename):
            files.save(os.path.join(app.config['UPLOAD_FOLDER'], 'save.png'))
            success = True
            image_path = Image.open(files)
            image_path = image_path.convert('RGB')

            image_path.close()
            # prepare image for prediction
            # img = keras.utils.load_img(dir, target_size=(150, 150))
            img = np.array(Image.open(files).resize((150,150)))
            # x = load_image_into_numpy_array(img)
            x = np.expand_dims(img, axis=0)
            images = np.vstack([x])
            image_path.close()


            # predict
            prediction_array = model.predict(images)

            class_names =  ['asam jawa', 'cengkeh', 'daun jeruk', 'daun salam', 'jahe', 'kayu manis', 'keluak', 'kemiri', 'ketumbar', 'kunyit', 'lada hitam', 'pekak','serai']
            result = {
                "prediction": class_names[np.argmax(prediction_array)],
                "confidence": '{:2.0f}%'.format(100 * np.max(prediction_array))
            }

            return jsonify(isError=False, message="Success", statusCode=200, data=result), 200
    else:
            errors["message"] = 'File type of {} is not allowed'.format(files.filename)

    if not success:
        resp = jsonify(errors)
        resp.status_code = 400
        return resp
    # img_url = os.path.join(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=False)