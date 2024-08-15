import os
import cv2
from transnetv2 import TransNetV2
import torch

# Initialize TransNetV2 model
model = TransNetV2()


def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load video and create output directory
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    # Predict scenes
    _, single_frame_predictions, _ = model.predict_video(video_path)
    scenes = model.predictions_to_scenes(single_frame_predictions)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save every frame to the output directory
        frame_path = os.path.join(output_folder, f"frame_{
                                  frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()

    return scenes
