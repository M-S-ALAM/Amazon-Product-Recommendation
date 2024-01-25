import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import requests
from io import BytesIO


class Deep_Learning_VGG16:
    def __init__(self, asin, num_results):
        self.asin = asin
        self.num_results = num_results
        self.data = pd.read_pickle(
            '/home/shobot/Desktop/Project pro/Amazon Product Reviews/16k_apperal_data_preprocessed')
        self.df_asins = list(self.data['asin'])
        self.bottleneck_features_train = np.load(
            '/home/shobot/Desktop/Project pro/Amazon Product Reviews/16k_data_cnn_features.npy')
        self.asins = np.load('/home/shobot/Desktop/Project pro/Amazon Product Reviews/16k_data_cnn_feature_asins.npy')

    # get similar products using CNN features (VGG-16)

    def get_similar_product(self):
        self.data = self.data.reset_index(drop=True)
        doc_id = self.data[self.data['asin'] == self.asin].index
        asins = list(self.asins)
        # doc_id = asins.index(self.df_asins[doc_id])
        pairwise_dist = pairwise_distances(self.bottleneck_features_train,
                                           self.bottleneck_features_train[doc_id].reshape(1, -1))

        indices = np.argsort(pairwise_dist.flatten())[0:self.num_results]
        return indices


    def get_similar_products_cnn(self):
        self.data = self.data.reset_index(drop=True)
        doc_id = self.data[self.data['asin'] == self.asin].index
        asins = list(self.asins)
        #doc_id = asins.index(self.df_asins[doc_id])
        pairwise_dist = pairwise_distances(self.bottleneck_features_train,
                                           self.bottleneck_features_train[doc_id].reshape(1, -1))

        indices = np.argsort(pairwise_dist.flatten())[0:self.num_results]
        pdists = np.sort(pairwise_dist.flatten())[0:self.num_results]

        for i in range(len(indices)):
            rows = self.data[['asin', 'brand', 'medium_image_url', 'title']].loc[self.data['asin'] == asins[indices[i]]]
            for indx, row in rows.iterrows():
                img = Image.open(BytesIO(requests.get(row['medium_image_url']).content))
                plt.imshow(img)
                plt.show()
                print('Product Title: ', row['title'])
                print('ASIN :', row['asin'])
                print('Brand:', row['brand'])
                print('Euclidean Distance from input image:', pdists[i])
                print('Amazon Url: www.amzon.com/dp/' + asins[indices[i]])


if __name__ == '__main__':
    recommendation = Deep_Learning_VGG16('B015YKMU80', 10)
    recommendation.get_similar_products_cnn()