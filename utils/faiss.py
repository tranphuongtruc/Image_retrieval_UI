from PIL import Image
import faiss
import matplotlib.pyplot as plt
import math
import numpy as np
import clip
from langdetect import detect


class Myfaiss:
    def __init__(self, bin_file: str, id2img_fps, device, translater, clip_backbone="ViT-B/32"):
        self.index = self.load_bin_file(bin_file)
        self.id2img_fps = id2img_fps
        self.device = device
        self.model, _ = clip.load(clip_backbone, device=device)
        self.translater = translater

    def load_bin_file(self, bin_file: str):
        return faiss.read_index(bin_file)

    def show_images(self, image_paths):
        fig = plt.figure(figsize=(15, 10))
        columns = int(math.sqrt(len(image_paths)))
        rows = int(np.ceil(len(image_paths)/columns))

        for i in range(1, columns*rows + 1):
            img = plt.imread(image_paths[i - 1])
            ax = fig.add_subplot(rows, columns, i)
            ax.set_title('/'.join(image_paths[i - 1].split('/')[-3:]))

            plt.imshow(img)
            plt.axis("off")

        plt.show()

    def image_search(self, id_query, k):
        query_feats = self.index.reconstruct(id_query).reshape(1, -1)

        scores, idx_image = self.index.search(query_feats, k=k)
        idx_image = idx_image.flatten()

        infos_query = list(map(self.id2img_fps.get, list(idx_image)))
        image_paths = [info for info in infos_query]

        return scores, idx_image, infos_query, image_paths

    def text_search(self, text, k):
        if detect(text) == 'vi':
            text = self.translater(text)

        ###### TEXT FEATURES EXTRACTION ######
        text = clip.tokenize([text]).to(self.device)
        text_features = self.model.encode_text(
            text).cpu().detach().numpy().astype(np.float32)

        ###### SEARCHING ######
        scores, idx_image = self.index.search(text_features, k=k)
        print(f"Requested k={k}, got {len(idx_image.flatten())} results")

        ###### DEBUG ######
        # print(f"Length of scores: {len(scores)}")
        # print(f"Shape of scores: {scores.shape}")

        ###### TURN TO SCALAR ######
        scores = scores.flatten()
        idx_image = idx_image.flatten()

        ###### GET INFOS KEYFRAMES_ID ######
        infos_query = list(map(self.id2img_fps.get, list(idx_image)))
        image_paths = [info for info in infos_query]

        ###### GET IMAGE FEATURES FOR RE-RANKING ######
        image_features = np.array(
            [self.index.reconstruct(int(idx)) for idx in idx_image])

        ###### NORMALIZATION ######
        text_features_normalized = text_features / \
            np.linalg.norm(text_features)
        image_features_normalized = image_features / \
            np.linalg.norm(image_features, axis=1, keepdims=True)
        text_features_normalized = np.tile(
            text_features_normalized, (image_features_normalized.shape[0], 1))

        ###### COMPUTE RSS ######
        # squared_residuals = np.square(text_features - image_features)
        squared_residuals = np.square(
            text_features_normalized - image_features_normalized)
        rss = np.sum(squared_residuals, axis=1)

        ###### DEBUGGING ######
        print(f"Shape of RSS: {rss.shape}")

        ###### RE-RANKING ######
        re_ranked_results = []

        ###### DEBUGGING ######
        # print(f"Length of image_paths (b4 loop): {len(image_paths)}")
        # print(f"Length of idx_image (b4 loop): {len(idx_image)}")
        # print(f"Length of scores (b4 loop): {len(scores)}")
        # print(f"Length of RSS (b4 loop): {len(rss)}")

        ###### SCORE READJUSTING ######
        for img_path, img_id, score, residual in zip(image_paths, idx_image, scores, rss):
            penalty_factor = 1 / (1 + residual)**2
            # penalty_factor = 1 / (1 + np.sqrt(residual))
            adjusted_score = score * penalty_factor

            ###### DEBUGGING ######
            print(f"Image Path: {img_path}, Raw Score: {score}, RSS: {
                  residual}, Adjusted Score: {adjusted_score}")

            re_ranked_results.append((img_path, img_id, adjusted_score))

        ###### FINAL RANKING ######
        re_ranked_results = sorted(
            re_ranked_results, key=lambda x: x[2], reverse=True)

        image_paths = [result[0] for result in re_ranked_results]
        idx_image = [result[1] for result in re_ranked_results]
        scores = [result[2] for result in re_ranked_results]

        ###### DEBUGGING ######
        # print(f"Total indexed items: {self.index.ntotal}")
        # print(f"Final count of re-ranked results: {len(re_ranked_results)}")
        # print(f"Image Paths: {image_paths}")
        # print(f"Image IDs: {idx_image}")
        # print(f"Shape of reranked scores: {len(scores)}")

        return scores, idx_image, infos_query, image_paths
