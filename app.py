from flask import Flask, request, jsonify
import boto3
import numpy as np
import onnxruntime as ort
import cv2
import io
import json
from flask_cors import CORS
from PIL import Image
import logging

# Setup logging
logging.basicConfig(filename='image_processing.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

app = Flask(__name__)
CORS(app)

# Assuming 'segment_anything.onnx' is your ONNX model file name
MODEL_PATH = './sam_vit_b_01ec64.encoder.preprocess.quant.onnx'

# Initialize ONNX runtime session for the model
ort_session = ort.InferenceSession(MODEL_PATH)


@app.route('/')
def home():
    return "Hello, Flask!"


def get_image_embedding_2(image):
    try:
        # Read image data using OpenCV
        # img_array = np.asarray(bytearray(image))
        # img = cv2.imdecode(img_array, -1)

        # logging.info(
        #     f'---\nProcessing started for image of size::{img.shape}\n---')
        # # Save data to JSON files

        # with open('img.json', 'w') as f:
        #     json.dump(img.tolist(), f)

        # # Resize the image to the expected input size
        # desired_height = 684
        # desired_width = 1024
        # resized_img = cv2.resize(img, (desired_width, desired_height))
        # with open('resized_img.json', 'w') as f:
        #     json.dump(resized_img.tolist(), f)
        resized_img = Image.open('image.jpg').resize((1024, 684))

        # Convert the image to RGB if it's not already in that format
        # if resized_img.shape[2] == 4:  # Assuming the fourth channel is alpha
        #     resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGRA2RGB)
        # # If already has 3 channels, assume it's BGR
        # elif resized_img.shape[2] == 3:
        #     resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        # else:
        #     raise ValueError("Image does not have 3 channels")

        # Normalize the pixel values if necessary (assuming the model expects values [0, 1])
        # img_data = resized_img.astype(np.float32) / 255.0
        # with open('img_data.json', 'w') as f:
        #     json.dump(img_data.tolist(), f)

        # # Reshape image data to remove batch dimension and move channel dimension to the last position
        # img_data = np.transpose(img_data, (1, 2, 0))  # From CHW to HWC format

        # logging.info(
        #     f'---\nImage data tensor shape after conversion::{img_data.shape}\n---')
        # transposed = np.transpose(img_data, (2, 1, 0))
        # with open('converted_img_data.json', 'w') as f:
        #     json.dump(img_data.tolist(), f)

        # # Now, we need to swap the channels axis with the width axis to get to (height, width, channels)
        # # The desired shape is (height, width, channels)
        # final_shape = np.transpose(transposed, (0, 2, 1))
        # with open('final_shape.json', 'w') as f:
        # json.dump(final_shape.tolist(), f)

        # Create ONNX runtime input, this assumes the model doesn't expect a batch dimension
        final_input = np.array(resized_img)
        ort_session = ort.InferenceSession(MODEL_PATH)
        ort_inputs = {ort_session.get_inputs()[0].name: final_input}
        print(final_input.shape)

        # Run the model
        ort_outs = ort_session.run(None, ort_inputs)

        # Get the embeddings
        image_embeddings = ort_outs[0]

        logging.info(
            f'---\nEncoding result obtained::{image_embeddings.shape}\n---')

        return image_embeddings
    except Exception as e:
        logging.error(f'Error occurred while processing image::{e}')
        return None


@app.route('/convert-image-to-embeddings', methods=['POST'])
def image_to_embedding_converter():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400
    file = request.files['file']
    Image.open(file).save('image.jpg')
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    if file:
        # Get the embeddings for the image
        image_embeddings = get_image_embedding_2(
            file.read())

        # Prepare and send the response
        response = {
            'statusCode': 200,
            'imageEmbeddings': image_embeddings.tolist()
        }
        # response.headers.add('Access-Control-Allow-Origin', '*')
        return jsonify(response), 200


if __name__ == '__main__':
    app.run(debug=True)
