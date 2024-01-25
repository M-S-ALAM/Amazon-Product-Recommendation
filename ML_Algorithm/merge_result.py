import pandas as pd
from ML_Algorithm.BOW_algorithm import Recommendation_BOW
from ML_Algorithm.TF_IDF_algorithm import Recommendation_TFIDF
from ML_Algorithm.idf_algorithm import Recommendation_IDF
from ML_Algorithm.avg_word2vec_algorithm import Recommendation_word2vec
from ML_Algorithm.weighted_word2vec import Recommendation_weighted_word2vec
from ML_Algorithm.Deep_Learning_vgg16 import Deep_Learning_VGG16
from ML_Algorithm.another_feature import Recommend_with_features


class Combine_results:
    def __init__(self, asin, num_resuls, model):
        self.asin = asin
        self.num_results = num_resuls
        self.model = model
        self.data =  pd.read_pickle(
            '/home/shobot/Desktop/Project pro/Amazon Product Reviews/pickels/16k_apperal_data_preprocessed')

    def Recommended_results(self):
        Indices = set()
        if self.model == 'Term frequency-inverse document frequency(TF-IDF)':
            tfidf_indices = Recommendation_TFIDF(self.asin, self.num_results).get_similar_product()
            return tfidf_indices
        elif self.model == 'Bag of Word(BOW)':
            bow_indices = Recommendation_BOW(self.asin, self.num_results).get_similar_product()
            return bow_indices
        elif self.model == 'Inverse document frequency(IDF)':
            Recommendation_IDF(self.asin, self.num_results).idf_model()
        elif self.model == 'IDF weighted Word 2 vec':
            weighted_indices = Recommendation_weighted_word2vec(self.asin, self.num_results).get_similar_product()
            return weighted_indices
        elif self.model == 'Average Word 2 Vec':
            avg_w2v_indices = Recommendation_word2vec(self.asin, self.num_results).get_similar_product()
            return avg_w2v_indices
        elif self.model == 'Algorithm with brand and color':
            extra_features_indices = Recommend_with_features(self.asin, self.num_results, w1=3, w2=7).get_similar_product()
            return extra_features_indices
        elif self.model == 'Deep Learning(VGG16)':
            vgg_16_indices = Deep_Learning_VGG16(self.asin, self.num_results).get_similar_product()
            return vgg_16_indices
        else:
            tfidf_indices = Recommendation_TFIDF(self.asin, self.num_results).get_similar_product()
            bow_indices = Recommendation_BOW(self.asin, self.num_results).get_similar_product()
            weighted_indices = Recommendation_weighted_word2vec(self.asin, self.num_results).get_similar_product()
            avg_w2v_indices = Recommendation_word2vec(self.asin, self.num_results).get_similar_product()
            extra_features_indices = Recommend_with_features(self.asin, self.num_results, w1=3,
                                                             w2=7).get_similar_product()
            vgg_16_indices = Deep_Learning_VGG16(self.asin, self.num_results).get_similar_product()
            for value in [tfidf_indices, bow_indices, weighted_indices, avg_w2v_indices, extra_features_indices, vgg_16_indices]:
                for element in value:
                    Indices.add(element)
            Indices = list(Indices)
            return Indices






def main():
    recommendation = Combine_results('B015YKMU80', 3)
    recommendation.Recommended_results()


if __name__ == '__main__':
    main()
