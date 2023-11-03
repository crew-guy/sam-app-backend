from flask import Flask, request, jsonify
import boto3
import numpy as np
import onnxruntime as ort
import cv2
import io
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# Assuming 'segment_anything.onnx' is your ONNX model file name
MODEL_PATH = './sam_vit_b_01ec64.encoder.preprocess.quant.onnx'

# Initialize ONNX runtime session for the model
ort_session = ort.InferenceSession(MODEL_PATH)


@app.route('/')
def home():
    return "Hello, Flask!"


def get_image_embedding(image_data):
    # Read image data using OpenCV
    img_array = np.asarray(bytearray(image_data), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    print(img.shape)

    # Resize image to match the model's expected input size
    dimension = 1024
    height, width = img.shape[:2]
    img = cv2.resize(img, (dimension, int(dimension * height / width)))

    # Ensure the image is in RGB (OpenCV loads images in BGR)

    # Normalize and add batch dimension
    np_image = img.astype(np.float32) / 255.0
    # np_image = np.transpose(np_image, (2, 0, 1))
    # Check if the model expects a 3D input and remove the batch dimension if necessary
    # np_image = np.expand_dims(np_image, 0)
    # if len(np_image.shape) == 4 and np_image.shape[0] == 1:
    #     np_image = np.squeeze(np_image, axis=0)

    print(np_image.shape)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create ONNX runtime input
    ort_tensor = ort.OrtValue.ortvalue_from_numpy(np_image, "cpu", 0)
    # Load the ONNX model and compute the embeddings
    session = ort.InferenceSession(MODEL_PATH)
    feeds = {'input_image': ort_tensor}
    results = session.run(None, feeds)

    # Get the embeddings
    image_embeddings = results[0]
    return image_embeddings


@app.route('/convert-image-to-embeddings', methods=['POST'])
def image_to_embedding_converter():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    if file:
        # Get the embeddings for the image
        image_embeddings = get_image_embedding(file.read())

        # Prepare and send the response
        response = {
            'statusCode': 200,
            'imageEmbeddings': image_embeddings.tolist()
        }
        return jsonify(response), 200


if __name__ == '__main__':
    app.run(debug=True)
