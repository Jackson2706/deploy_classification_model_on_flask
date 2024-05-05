import cv2
from flask import Flask, render_template, request
from InferenceModel import ClassificationInferenceModel
from config import CLS_ONNX_MODEL_PATH, INPUT_SHAPE, LABEL
import os
import shutil
app = Flask(__name__)
cls_inference_model = None
STATIC_PATH = "static"

@app.before_request
def init_model():
    global cls_inference_model
    cls_inference_model = ClassificationInferenceModel(CLS_ONNX_MODEL_PATH, INPUT_SHAPE, LABEL)
    if not os.path.exists(STATIC_PATH):
        os.mkdir(STATIC_PATH)

@app.route('/')
def index():
    try:
        shutil.rmtree(STATIC_PATH)  # Xóa toàn bộ thư mục và nội dung bên trong
    except Exception as e:
        print(f"Không thể xóa thư mục {STATIC_PATH}: {e}")
    return render_template('index.html')


import base64


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    filename = file.filename
    file_path = 'static/' + filename
    file.save(file_path)

    # Chuyển đổi hình ảnh thành base64
    with open(file_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode('utf-8')

    return render_template('index.html', filename=filename, encoded_string=encoded_string)


@app.route('/predict', methods=['POST'])
def predict():
    filename = request.form['filename']
    file_path = 'static/' + filename
    with open(file_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
    image = cv2.imread(file_path)
    result = cls_inference_model.run_native_inference(image)
    result = "Result: {}".format(result)
    return render_template('index.html', filename=filename, encoded_string=encoded_string, result=result)


if __name__ == '__main__':
    app.run(debug=True)
