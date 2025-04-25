import os
import numpy as np
from PIL import Image
import uuid

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

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tensorflow.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tensorflow.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tensorflow.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tensorflow.newaxis]
    heatmap = tensorflow.squeeze(heatmap)

    heatmap = tensorflow.maximum(heatmap, 0) / tensorflow.math.reduce_max(heatmap)
    return heatmap.numpy()
def save_and_superimpose_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    cv2.imwrite(cam_path, superimposed_img)

def get_className(classNo):
	if classNo==0:
		return "Normal"
	elif classNo==1:
		return "Pneumonia"


def getResult(img):
    print("Image path: ", img)
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    input_img = np.expand_dims(image, axis=0)

    result = model_03.predict(input_img)
    pneumonia_prob = float(result[0][1])
    predicted_class = np.argmax(result, axis=1)[0]

    # Grad-CAM
    heatmap = make_gradcam_heatmap(input_img, model_03, last_conv_layer_name="block5_conv4", pred_index=predicted_class)
    basepath = os.getcwd()  # Define basepath as the current working directory
    heatmap_folder = os.path.join(basepath, 'static', 'heatmap')  # move to static folder
    original_filename = os.path.basename(img)
    name, ext = os.path.splitext(original_filename)
    cam_filename = f"cam_{name}.jpeg"  # Save as .jpg regardless of original extension
    cam_path = os.path.join(heatmap_folder, cam_filename)

    os.makedirs(heatmap_folder, exist_ok=True)  # Ensures folder exists
    # cam_path = os.path.join(heatmap_folder, 'cam.jpg')

    save_and_superimpose_gradcam(img, heatmap, cam_path)

    # Severity
    severity_score = round(pneumonia_prob * 100, 2)
    if severity_score < 40:
        severity_level = "Mild"
    elif severity_score < 70:
        severity_level = "Moderate"
    else:
        severity_level = "Severe"

    return predicted_class, severity_score, severity_level, cam_path



@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print(file_path," saved")
        class_id, severity_score, severity_level , f= getResult(file_path)
        class_name = get_className(class_id)

        if class_name == "Normal":
            return f"{class_name}"
        else:
            return f"{class_name} (Severity Score: {severity_score}%, Level: {severity_level})"

@app.route('/show_heatmap/<filename>')
def show_heatmap(filename):
    return render_template('show_heatmap.html', filename=filename)

if __name__ == '__main__':
    app.run(debug=True)