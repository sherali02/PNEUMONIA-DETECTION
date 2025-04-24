import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19


base_model = VGG19(include_top=False, input_shape=(128,128,3))
x = base_model.output
flat=Flatten()(x)
class_1 = Dense(4608, activation='relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(drop_out)
output = Dense(2, activation='softmax')(class_2)
model_03 = Model(base_model.inputs, output)
model_03.load_weights('full_model.h5')
app = Flask(__name__)

print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
	if classNo==0:
		return "Normal"
	elif classNo==1:
		return "Pneumonia"


# def getResult(img):
#     image=cv2.imread(img)
#     image = Image.fromarray(image, 'RGB')
#     image = image.resize((128, 128))
#     image=np.array(image)
#     input_img = np.expand_dims(image, axis=0)
#     result=model_03.predict(input_img)
#     result01=np.argmax(result,axis=1)
#     return result01
# def getResult(img):
#     image = cv2.imread(img)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, (128, 128))
#     image = image / 255.0  # 🔥 Important: normalize
#     input_img = np.expand_dims(image, axis=0)
#     result = model_03.predict(input_img)
#     resultx = model_03.predict(input_img)[0]  # resultx = [prob_normal, prob_pneumonia]
#     class_index = np.argmax(resultx)
#     confidence = round(resultx[class_index] * 100, 2)
    
#     class_name = get_className(class_index)
#     print(f"Class: {class_name}, Confidence: {confidence}%")
#     result01 = np.argmax(result, axis=1)[0]
#     print(f"Class: {result01}")
#     return result01

# def getResult(img):
#     image = cv2.imread(img)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, (128, 128))
#     image = image / 255.0  # 🔥 Important: normalize
#     input_img = np.expand_dims(image, axis=0)
    
#     result = model_03.predict(input_img)[0]  # result = [prob_normal, prob_pneumonia]
#     class_index = np.argmax(result)
#     confidence = round(result[class_index] * 100, 2)
#     pneumonia_prob = round(float(result[1]) * 100, 2)
#     class_name = get_className(class_index)
#     return f"{class_name} ({pneumonia_prob}%)"

def getResult(img):
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    input_img = np.expand_dims(image, axis=0)
    result = model_03.predict(input_img)

    pneumonia_prob = float(result[0][1])  # softmax probability for Pneumonia
    predicted_class = np.argmax(result, axis=1)[0]

    # Get severity level
    severity_score = round(pneumonia_prob * 100, 2)
    if severity_score < 40:
        severity_level = "Mild"
    elif severity_score < 70:
        severity_level = "Moderate"
    else:
        severity_level = "Severe"

    return predicted_class, severity_score, severity_level




@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# @app.route('/predict', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         f = request.files['file']

#         basepath = os.path.dirname(__file__)
#         file_path = os.path.join(
#             basepath, 'uploads', secure_filename(f.filename))
#         f.save(file_path)
#         result = getResult(file_path)
#         return result
#     return None
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        class_id, severity_score, severity_level = getResult(file_path)
        class_name = get_className(class_id)

        if class_name == "Normal":
            return f"{class_name} (Pneumonia Probability: {severity_score}%)"
        else:
            return f"{class_name} (Severity Score: {severity_score}%, Level: {severity_level})"


if __name__ == '__main__':
    app.run(debug=True)