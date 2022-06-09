from flask import Flask, request, jsonify
import numpy as np
from tensorflow import keras
from keras.models import load_model
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = load_model('spezia_model.h5')

@app.route('/')
def main():
    return 'Welcome to Spezia ML Team API for predict many spices'

@app.route('/predict', methods=['POST'])
def recognize_image():
    try:        
        # open image from request
        img_sample = Image.open(request.files['image'])
        image_path = img_sample
        image_path = image_path.convert('RGB')
        image_path.close()

        # prepare image for prediction
        img = np.array(img_sample.resize((150,150)))
        
        x = np.expand_dims(img, axis=0)
        images = np.vstack([x])
        image_path.close()


        # predict
        prediction_array = model.predict(images)

        class_names =  ['asam jawa', 'cengkeh', 'daun jeruk', 'daun salam', 
                        'jahe', 'kayu manis', 'keluak', 'kemiri', 'ketumbar', 
                        'kunyit', 'lada hitam', 'pekak','serai']
        result = {
            'prediction': class_names[np.argmax(prediction_array)],
            'confidence': '{:2.0f}%'.format(100 * np.max(prediction_array))
        }

        return jsonify(isError=False, message='Success', statusCode=200, data=result), 200

    except Exception as e:
        print(str(e))
        return jsonify(message='Something went wrong'), 500

if __name__ == '__main__':
    app.run(debug=True, port=3000)