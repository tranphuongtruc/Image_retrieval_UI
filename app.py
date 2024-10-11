import json
import glob
from utils.faiss import Myfaiss
from utils.query_processing import Translation
import pandas as pd
import numpy as np
from flask import Flask, render_template, Response, request, send_file, jsonify
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


app = Flask(__name__, static_folder='static', template_folder='templates')

# Load the image paths and FAISS index
with open('image_path.json') as json_file:
    json_dict = json.load(json_file)

DictImagePath = {int(key): value for key, value in json_dict.items()}
LenDictPath = len(DictImagePath)

bin_file = 'faiss_normal_ViT.bin'
MyFaiss = Myfaiss(bin_file, DictImagePath, 'cpu', Translation(), "ViT-B/32")


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/feedback')
def feedback():
    print("feedback page")
    # backend để đây nè
    return render_template('feedback.html')


@app.route('/text-to-image', methods=['GET'])
def text_to_image():
    print("text search")

    pagefile = []
    text_query = request.args.get('textquery')

    # Check if the text_query is provided
    if text_query:
        _, list_ids, _, list_image_paths = MyFaiss.text_search(
            text_query, k=100)

        imgperindex = 100

        for imgpath, id in zip(list_image_paths, list_ids):
            pagefile.append({'imgpath': imgpath, 'id': int(id)})

        data = {'num_page': int(LenDictPath/imgperindex) +
                1, 'pagefile': pagefile}
    else:
        # No query provided, return empty state or error message
        data = {'num_page': 0, 'pagefile': []}

    return render_template('text-to-image.html', data=data, query=text_query)


POSITION = (30, 80)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 2.5
THICKNESS = 7
OUTLINE_COLOR = (0, 0, 0)
FILL_COLOR = (255, 255, 255)


@app.route('/get_img')
def get_img():
    print("get_img")
    fpath = request.args.get('fpath')

    list_image_name = fpath.split("/")
    image_name = "/".join(list_image_name[-1:])

    if os.path.exists(fpath):
        img = cv2.imread(fpath)
    else:
        print("load 404.jpg")
        img = cv2.imread("./static/images/404.jpg")

    img = cv2.resize(img, (1280, 720))

    img = cv2.putText(img, image_name, POSITION, FONT,
                      FONT_SCALE, OUTLINE_COLOR, THICKNESS + 8, cv2.LINE_AA)
    img = cv2.putText(img, image_name, POSITION, FONT,
                      FONT_SCALE, FILL_COLOR, THICKNESS, cv2.LINE_AA)
    # img = cv2.putText(img, image_name, (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
    #                   2.5, (255, 0, 0), 4, cv2.LINE_AA)

    ret, jpeg = cv2.imencode('.jpg', img)
    return Response((b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
