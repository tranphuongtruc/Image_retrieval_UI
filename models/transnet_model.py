import os
from transnetv2 import TransNetV2
from PIL import Image


def extract_frames(video_path, output_folder):
    model = TransNetV2()
    frames, _ = model.predict_video(video_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i, frame in enumerate(frames):
        image_path = os.path.join(output_folder, f"frame_{i}.jpg")
        Image.fromarray(frame).save(image_path)
