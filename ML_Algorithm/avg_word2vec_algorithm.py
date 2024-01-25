# -- coding: UTF-8 --
"""
Convert text into audio.
====================================
"""
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle
import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import seaborn as sns
import pandas as pd
from sklearn.metrics import pairwise_distances
from matplotlib import gridspec
import numpy as np


class Recommendation_word2vec:
    def __init__(self, asin, num_results):
        self.vocab = None
        self.model = None
        self.w2v_title = []
        self.data = pd.read_pickle(
            '/home/shobot/Desktop/Project pro/Amazon Product Reviews/pickels/16k_apperal_data_preprocessed')
        self.asin = asin
        self.num_results = num_results

    def display_img(self, url, ax, fig):
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        plt.imshow(img)

    def get_distance(self, vec1, vec2):
        final_dist = []
        for i in vec1:
            dist = []
            for j in vec2:
                dist.append(np.linalg.norm(i - j))
            final_dist.append(np.array(dist))
        return np.array(final_dist)

    def get_word_vec(self, sentence, doc_id, m_name):
        vec = []
        for i in sentence.split():
            if i in self.vocab:
                vec.append(self.model[i])
            else:
                vec.append(np.zeros(shape=(300,)))
        return np.array(vec)

    def build_avg_vec(self, sentence, num_features, doc_id, m_name):

        featureVec = np.zeros((num_features,), dtype="float32")
        nwords = 0
        for word in sentence.split():
            nwords += 1
            if word in self.vocab:
                featureVec = np.add(featureVec, self.model[word])
        if nwords > 0:
            featureVec = np.divide(featureVec, nwords)
        return featureVec

    def heat_map_w2v(self, sentence1, sentence2, url, doc_id1, doc_id2, model):
        s1_vec = self.get_word_vec(sentence1, doc_id1, model)
        s2_vec = self.get_word_vec(sentence2, doc_id2, model)
        s1_s2_dist = self.get_distance(s1_vec, s2_vec)
        gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[2, 1])
        fig = plt.figure(figsize=(15, 15))
        ax = plt.subplot(gs[0])
        ax = sns.heatmap(np.round(s1_s2_dist, 4), annot=True)
        ax.set_xticklabels(sentence2.split())
        ax.set_yticklabels(sentence1.split())
        ax.set_title(sentence2)

        ax = plt.subplot(gs[1])
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        self.display_img(url, ax, fig)

        plt.show()

    def get_similar_product(self):
        self.data = self.data.reset_index(drop=True)
        self.asin_index = self.data[self.data['asin'] == self.asin].index
        if self.asin in self.data['asin'].values:
            doc_id = 0
            with open('/home/shobot/Desktop/Project pro/Amazon Product Reviews/word2vec_model', 'rb') as f:
                self.model = pickle.load(f)
            self.vocab = self.model.keys()
            for i in self.data['title']:
                self.w2v_title.append(self.build_avg_vec(i, 300, doc_id, 'avg'))
                doc_id += 1
            self.w2v_title = np.array(self.w2v_title)
            pairwise_dist = pairwise_distances(self.w2v_title, self.w2v_title[self.asin_index].reshape(1, -1))
            indices = np.argsort(pairwise_dist.flatten())[0:self.num_results]
            return indices
        else:
            return None

    def avg_w2v_model(self):
        self.data = self.data.reset_index(drop=True)
        self.asin_index = self.data[self.data['asin'] == self.asin].index
        if self.asin in self.data['asin'].values:
            doc_id = 0
            with open('/home/shobot/Desktop/Project pro/Amazon Product Reviews/word2vec_model', 'rb') as f:
                self.model = pickle.load(f)
            self.vocab = self.model.keys()
            for i in self.data['title']:
                self.w2v_title.append(self.build_avg_vec(i, 300, doc_id, 'avg'))
                doc_id += 1
            self.w2v_title = np.array(self.w2v_title)
            pairwise_dist = pairwise_distances(self.w2v_title, self.w2v_title[self.asin_index].reshape(1, -1))
            indices = np.argsort(pairwise_dist.flatten())[0:self.num_results]
            pdists = np.sort(pairwise_dist.flatten())[0:self.num_results]
            df_indices = list(self.data.index[indices])

            for i in range(0, len(indices)):
                self.heat_map_w2v(self.data['title'].loc[df_indices[0]], self.data['title'].loc[df_indices[i]],
                                  self.data['medium_image_url'].loc[df_indices[i]], indices[0], indices[i], 'avg')
                print('ASIN :', self.data['asin'].loc[df_indices[i]])
                print('BRAND :', self.data['brand'].loc[df_indices[i]])
                print('euclidean distance from given input image :', pdists[i])
                print('=' * 125)
            else:
                print('ASIN number is not in data base')


if __name__ == '__main__':
    recommendation = Recommendation_word2vec('B015YKMU80', 3)
    recommendation.avg_w2v_model()
