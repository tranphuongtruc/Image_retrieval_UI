from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import torch
from models.transnet_model import extract_frames
from models.clip_model import load_clip_model, encode_image, encode_text
from models.faiss_index import build_faiss_index, search_index

app = Flask(__name__, static_folder='static', template_folder='templates')

# Global variables for storing model and index
clip_model, preprocess = load_clip_model()
faiss_index = None
image_vectors = []
image_paths = []


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/image-to-image', methods=['GET', 'POST'])
def image_to_image():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            # Save the file to the 'uploads' folder
            upload_folder = 'uploads'
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)

            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)

            # Encode uploaded image and search in the index
            query_vector = encode_image(
                file_path, clip_model, preprocess).detach().numpy()
            indices, _ = search_index(faiss_index, query_vector, k=5)
            results = [image_paths[i] for i in indices[0]]

            return render_template('image-to-image.html', results=results)

    return render_template('image-to-image.html')


@app.route('/text-to-image', methods=['GET', 'POST'])
def text_to_image():
    if request.method == 'POST':
        query = request.form.get('query')
        if query:
            # Encode query text and search in the index
            query_vector = encode_text(query, clip_model).detach().numpy()
            indices, _ = search_index(faiss_index, query_vector, k=5)
            results = [image_paths[i] for i in indices[0]]

            return render_template('text-to-image.html', results=results)

    return render_template('text-to-image.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the file to the 'uploads' folder
        upload_folder = 'uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        # Extract frames and update the index
        process_frames_and_update_index(file_path)

        return 'Successfully uploaded'

    return 'Failed to upload image.'


def process_frames_and_update_index(video_path):
    global faiss_index, image_vectors, image_paths

    output_folder = 'processed_images'
    extract_frames(video_path, output_folder)

    for image_file in os.listdir(output_folder):
        image_path = os.path.join(output_folder, image_file)
        image_paths.append(image_path)
        image_vector = encode_image(
            image_path, clip_model, preprocess).detach().numpy().flatten()
        image_vectors.append(image_vector)

    # Build the FAISS index
    faiss_index = build_faiss_index(image_vectors)


if __name__ == '__main__':
    app.run(debug=True)
