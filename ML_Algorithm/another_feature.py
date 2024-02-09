# -- coding: UTF-8 --
"""
Method to recommend similar product based on color, brand, product type, and title.
========================================================================================
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import requests
import plotly
import pickle
import logging
import plotly.figure_factory as ff
from matplotlib import gridspec
from scipy.sparse import hstack
from io import BytesIO
import numpy as np
from PIL import Image
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

__author__ = "msalamiitd@gmail.com"
LOGGER = logging.getLogger(__name__)


class Recommend_with_features:
    """

    """
    def __init__(self, asin, num_results, w1, w2):
        """

        :param asin:
        :param num_results:
        :param w1:
        :param w2:
        """
        self.extra_features = None
        self.model = None
        self.title_features = None
        self.title_vectorizer = None
        self.w1 = w1
        self.w2 = w2
        self.colors = None
        self.types = None
        self.brands = None
        self.data = pd.read_pickle('database/16k_apperal_data_preprocessed')
        self.asin = asin
        self.num_results = num_results
        self.w2v_title = []

    def add_features(self):
        """

        :return:
        """
        self.data['brand'].fillna(value="Not given", inplace=True)
        self.brands = [x.replace(" ", "-") for x in self.data['brand'].values]
        self.types = [x.replace(" ", "-") for x in self.data['product_type_name'].values]
        self.colors = [x.replace(" ", "-") for x in self.data['color'].values]

        brand_vectorizer = CountVectorizer()
        brand_features = brand_vectorizer.fit_transform(self.brands)

        type_vectorizer = CountVectorizer()
        type_features = type_vectorizer.fit_transform(self.types)

        color_vectorizer = CountVectorizer()
        color_features = color_vectorizer.fit_transform(self.colors)

        extra_features = hstack((brand_features, type_features, color_features)).tocsr()
        return extra_features

    def display_img(self, url: str) -> list:
        """

        :param url: string
        :rtype: None
        """
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        plt.imshow(img)

    def get_distance(self, vec1: list, vec2: list) -> list:
        final_dist = []
        for i in vec1:
            dist = []
            for j in vec2:
                dist.append(np.linalg.norm(i - j))
            final_dist.append(np.array(dist))
        return np.array(final_dist)

    def get_word_vec(self, sentence: str, doc_id: str) -> list:
        vec = []
        for i in sentence.split():
            if i in self.vocab and i in self.title_vectorizer.vocabulary_:
                vec.append(self.title_features[doc_id, self.title_vectorizer.vocabulary_[i]] * self.model[i])
            else:
                vec.append(np.zeros(shape=(300,)))
        return np.array(vec)

    def vectorized(self):
        """

        :return:
        """
        title_vectorizer = TfidfVectorizer(min_df=0.0)
        title_features = title_vectorizer.fit_transform(self.data['title'])
        return title_vectorizer, title_features

    def build_avg_vec(self, sentence, num_features, doc_id, m_name):
        """

        :param sentence:
        :param num_features:
        :param doc_id:
        :param m_name:
        :return:
        """
        featureVec = np.zeros((num_features,), dtype="float32")
        nwords = 0
        for word in sentence.split():
            nwords += 1
            if word in self.vocab:
                featureVec = np.add(featureVec, self.model[word])
        if nwords > 0:
            featureVec = np.divide(featureVec, nwords)
        return featureVec

    def heat_map_w2v_brand(self, sentence1, sentence2, url, doc_id1, doc_id2, df_id1, df_id2, model):
        """

        :param sentence1:
        :param sentence2:
        :param url:
        :param doc_id1:
        :param doc_id2:
        :param df_id1:
        :param df_id2:
        :param model:
        :return:
        """
        s1_vec = self.get_word_vec(sentence1, doc_id1)
        s2_vec = self.get_word_vec(sentence2, doc_id2)
        s1_s2_dist = self.get_distance(s1_vec, s2_vec)
        data_matrix = [['Asin', 'Brand', 'Color', 'Product type'],
                       [self.data['asin'].loc[df_id1], self.brands[doc_id1], self.colors[doc_id1], self.types[doc_id1]],
                       [self.data['asin'].loc[df_id2], self.brands[doc_id2], self.colors[doc_id2], self.types[doc_id2]]]
        colorscale = [[0, '#1d004d'], [.5, '#f2e5ff'], [1, '#f2e5d1']]
        table = ff.create_table(data_matrix, index=True, colorscale=colorscale)
        plotly.offline.iplot(table, filename='simple_table')
        gs = gridspec.GridSpec(25, 15)
        fig = plt.figure(figsize=(25, 5))
        ax1 = plt.subplot(gs[:, :-5])
        ax1 = sns.heatmap(np.round(s1_s2_dist, 6), annot=True)
        ax1.set_xticklabels(sentence2.split())
        ax1.set_yticklabels(sentence1.split())
        ax1.set_title(sentence2)
        ax2 = plt.subplot(gs[:, 10:16])
        ax2.grid(False)
        ax2.set_xticks([])
        ax2.set_yticks([])
        self.display_img(url)
        plt.show()

    def get_similar_product(self):
        """

        :return:
        """
        self.data = self.data.reset_index(drop=True)
        self.asin_index = self.data[self.data['asin'] == self.asin].index
        if self.asin in self.data['asin'].values:
            doc_id = 0
            self.title_vectorizer, self.title_features = self.vectorized()  # Fix here
            with open('database/word2vec_model', 'rb') as f:
                self.model = pickle.load(f)
            self.vocab = self.model.keys()
            self.extra_features = self.add_features()
            for i in self.data['title']:
                self.w2v_title.append(self.build_avg_vec(i, 300, doc_id, 'avg'))
                doc_id += 1
            self.w2v_title = np.array(self.w2v_title)
            idf_w2v_dist = pairwise_distances(self.w2v_title, self.w2v_title[self.asin_index].reshape(1, -1))
            ex_feat_dist = pairwise_distances(self.extra_features, self.extra_features[self.asin_index])
            pairwise_dist = (self.w1 * idf_w2v_dist + self.w2 * ex_feat_dist) / float(self.w1 + self.w2)
            indices = np.argsort(pairwise_dist.flatten())[0:self.num_results]
            return indices
        else:
            return None

    def idf_w2v_brand(self):
        """

        :return:
        """
        self.data = self.data.reset_index(drop=True)
        self.asin_index = self.data[self.data['asin'] == self.asin].index
        if self.asin in self.data['asin'].values:
            doc_id = 0
            self.title_vectorizer, self.title_features = self.vectorized()  # Fix here
            with open('database/word2vec_model', 'rb') as f:
                self.model = pickle.load(f)
            self.vocab = self.model.keys()
            self.extra_features = self.add_features()
            for i in self.data['title']:
                self.w2v_title.append(self.build_avg_vec(i, 300, doc_id, 'avg'))
                doc_id += 1
            self.w2v_title = np.array(self.w2v_title)
            idf_w2v_dist = pairwise_distances(self.w2v_title, self.w2v_title[self.asin_index].reshape(1, -1))
            ex_feat_dist = pairwise_distances(self.extra_features, self.extra_features[self.asin_index])
            pairwise_dist = (self.w1 * idf_w2v_dist + self.w2 * ex_feat_dist) / float(self.w1 + self.w2)
            indices = np.argsort(pairwise_dist.flatten())[0:self.num_results]
            pdists = np.sort(pairwise_dist.flatten())[0:self.num_results]
            df_indices = list(self.data.index[indices])

            for i in range(0, len(indices)):
                self.heat_map_w2v_brand(self.data['title'].loc[df_indices[0]], self.data['title'].loc[df_indices[i]],
                                        self.data['medium_image_url'].loc[df_indices[i]], indices[0], indices[i],
                                        df_indices[0],
                                        df_indices[i], 'weighted')
                print('ASIN :', self.data['asin'].loc[df_indices[i]])
                print('Brand :', self.data['brand'].loc[df_indices[i]])
                print('euclidean distance from input :', pdists[i])
                print('=' * 125)
        else:
            print('ASIN number is not in data base')


if __name__ == '__main__':
    recommendation = Recommend_with_features('B0758TBKRJ', 3, 5, 5)
    recommendation.idf_w2v_brand()
