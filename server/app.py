from __future__ import division, print_function

# coding=utf-8
import os
import re
import sys
import glob
import pickle
import numpy as np
from itertools import chain

# Keras
import tensorflow as tf
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Defining the flask app
app = Flask(__name__)

# Defining= the path for image and model
UPLOAD_FOLDER = '/Users/sparshbohra/Desktop/skin-lesion/server/static'

from keras.initializers import glorot_uniform

#Reading the model from JSON file
with open('/Users/sparshbohra/Desktop/skin-lesion/server/models/model_a.json', 'r') as json_file:
    json_savedModel = json_file.read()

#load the model architecture & weights
feature_model = tf.keras.models.model_from_json(json_savedModel)
feature_model.load_weights('/Users/sparshbohra/Desktop/skin-lesion/server/models/model_a_weights.h5')

# Load XGB model
file_name = "/Users/sparshbohra/Desktop/skin-lesion/server/models/xgb_classifier.pkl"
xgb_classifier = pickle.load(open(file_name, "rb"))

def model_predict(sex, dx_type, localization, age, img_path, model):
    # Load image
    img = image.load_img(img_path, target_size=(75, 75))

    # Preprocess the image
    x = image.img_to_array(img)
    x = x/255
    x = x.reshape(1, 75, 75, 3)

    predictions = feature_model.predict(x)

    sex_dict = {
        'female' : 0,
        'male' : 1
    }
    type_dict = {
        'confocal' : 0,
        'consensus' : 1,
        'follow_up' : 2,
        'histo' : 3
    }
    localization_dict = {
        'abdomen' : 0,
        'acral' : 1,
        'back' : 2,
        'chest' : 3,
        'ear' : 4,
        'face' : 5,
        'foot' : 6,
        'genital' : 7,
        'hand' : 8,
        'lower extremity' : 9,
        'neck' : 10,
        'scalp' : 11,
        'trunk' : 12,
        'unknown': 13,
        'upper extremity' : 14
    }

    # Preparing the data
    sex_data = np.zeros(2)
    type_data = np.zeros(4)
    localization_data = np.zeros(15)

    # One hot encoding
    sex_n = sex_dict[sex]
    sex_data[sex_n] = 1
    type_n = type_dict[dx_type]
    type_data[type_n] = 1
    localization_n = localization_dict[localization]
    localization_data[localization_n] = 1
    age_data = [age]

    # Concatenating all data
    data = []
    data = np.concatenate([data, sex_data])
    data = np.concatenate([data, type_data])
    data = np.concatenate([data, localization_data])
    data = np.concatenate([data, age_data])
    data = data.flatten()
    data = data.reshape(1, 22)

    df = pd.DataFrame(data, columns = ['female',
                                         'male',
                                         'abdomen',
                                         'acral',
                                         'back',
                                         'chest',
                                         'ear',
                                         'face',
                                         'foot',
                                         'genital',
                                         'hand',
                                         'lower extremity',
                                         'neck',
                                         'scalp',
                                         'trunk',
                                         'unknown',
                                         'upper extremity',
                                         'confocal',
                                         'consensus',
                                         'follow_up',
                                         'histo',
                                         'age',])

    # Merging image and text data
    test_data = pd.concat([df, pd.DataFrame(predictions)], axis=1)

    pred = xgb_classifier.predict(test_data)[0]
    return pred

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file and form data from post request
        image_file = request.files['file']
        var_age = request.form['age']
        var_sex = request.form['sex']
        var_dx_type = request.form['dx_type']
        var_localization = request.form['localization']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        image_location = os.path.join(
            basepath, 'static', secure_filename(image_file.filename))
        image_file.save(image_location)

        # Make prediction
        preds = model_predict(sex, dx_type, localization, age, image_location, model)

        # Map output to labels
        output_labels = ["Actinic Keratoses and Intraepithelial Carcinoma / Bowen's Disease (akiec)",
                         "Basal Cell Carcinoma (bcc)",
                         "Benign Keratosis-like Lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, bkl)",
                         "Dermatofibroma (df)",
                         "Melanocytic Nevi (nv)",
                         "Vascular Lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, vasc)",
                         "Melanoma (mel)"]

        output_length = len(output_labels)

        # Process your result for human
        result = output_labels[pred_class]
        return result
    return None

if __name__ == '__main__':
    app.run(port=12000, debug = True)
