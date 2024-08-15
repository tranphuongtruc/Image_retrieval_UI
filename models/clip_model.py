import torch
import clip
from PIL import Image


def load_clip_model():
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    return model, preprocess


def encode_image(image_path, model, preprocess):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to("cpu")
    return model.encode_image(image)


def encode_text(text, model):
    text_tokens = clip.tokenize([text]).to("cpu")
    return model.encode_text(text_tokens)
