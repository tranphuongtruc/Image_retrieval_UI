import clip
import torch
from PIL import Image


def load_clip_model():
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    return model, preprocess


def encode_image(image_path, model, preprocess):
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features


def encode_text(text, model):
    text_features = model.encode_text(clip.tokenize([text]))
    return text_features
