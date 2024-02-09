# -- coding: UTF-8 --
"""
Convert text into audio.
====================================
"""
import pandas as pd
import requests
import re
from io import BytesIO
from PIL import Image
import numpy as np
import seaborn as sns
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from matplotlib import gridspec
import matplotlib.pyplot as plt
import os

__author__ = "Md Shahbaz Alam"
__email__ = "msalamiitd@hmail.com"
__credits__ = "Md Shahbaz Alam"


class Recommendation_BOW:
    def __init__(self, asin, num_results):
        self.asin_index = None
        self.title_features = None
        self.data = pd.read_pickle('database/16k_apperal_data_preprocessed')
        self.asin = asin
        self.num_results = num_results

    def display_img(self, url, ax, fig):
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        plt.imshow(img)

    def plot_heatmap(self, keys, values, labels, url, text):
        gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1])
        fig = plt.figure(figsize=(25, 3))
        ax = plt.subplot(gs[0])
        ax = sns.heatmap(np.array([values]), annot=np.array([labels]))
        ax.set_xticklabels(keys)
        ax.set_title(text)
        ax = plt.subplot(gs[1])
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        self.display_img(url, ax, fig)
        plt.show()

    def plot_heatmap_image(self, doc_id, vec1, vec2, url, text, model):
        intersection = set(vec1.keys()) & set(vec2.keys())
        for i in vec2:
            if i not in intersection:
                vec2[i] = 0
        keys = list(vec2.keys())

        values = [vec2[x] for x in vec2.keys()]

        labels = values

        self.plot_heatmap(keys, values, labels, url, text)

    def text_to_vector(self, text):
        word = re.compile(r'\w+')
        words = word.findall(text)
        return Counter(words)

    def get_result(self, doc_id, content_a, content_b, url, model):
        text1 = content_a
        text2 = content_b
        vector1 = self.text_to_vector(text1)
        vector2 = self.text_to_vector(text2)
        self.plot_heatmap_image(doc_id, vector1, vector2, url, text2, model)

    def vectorized(self):
        title_vectorizer = CountVectorizer()
        self.title_features = title_vectorizer.fit_transform(self.data['title'])

    def get_similar_product(self):
        self.data = self.data.reset_index(drop=True)
        self.asin_index = self.data[self.data['asin'] == self.asin].index
        if self.asin in self.data['asin'].values:
            self.vectorized()
            pairwise_dist = pairwise_distances(self.title_features, self.title_features[self.asin_index])
            indices = np.argsort(pairwise_dist.flatten())[0:self.num_results]
            return indices
        else:
            return None

    def bag_of_words_model(self):
        self.data = self.data.reset_index(drop=True)
        self.asin_index = self.data[self.data['asin'] == self.asin].index
        if self.asin in self.data['asin'].values:
            self.vectorized()
            pairwise_dist = pairwise_distances(self.title_features, self.title_features[self.asin_index])
            indices = np.argsort(pairwise_dist.flatten())[0:self.num_results]
            pdists = np.sort(pairwise_dist.flatten())[0:self.num_results]
            df_indices = list(self.data.index[indices])
            for i in range(0, len(indices)):
                self.get_result(indices[i], self.data['title'].loc[df_indices[0]],
                                self.data['title'].loc[df_indices[i]], self.data['medium_image_url'].loc[df_indices[i]],
                                'bag_of_words')
                print('ASIN :', self.data['asin'].loc[df_indices[i]])
                print('Brand:', self.data['brand'].loc[df_indices[i]])
                print('Title:', self.data['title'].loc[df_indices[i]])
                print('Euclidean similarity with the query image :', pdists[i])
                print('=' * 100)
        else:
            print('ASIN number is not in data base')


if __name__ == '__main__':
    recommendation = Recommendation_BOW('B015YKMU80', 3)
    recommendation.bag_of_words_model()
