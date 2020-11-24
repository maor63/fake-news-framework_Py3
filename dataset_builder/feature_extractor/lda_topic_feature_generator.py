#  Created by YY at 1/2/2019
#  Modified by YY at 1/12/19


from dataset_builder.feature_extractor.base_feature_generator import BaseFeatureGenerator
from preprocessing_tools.abstract_controller import AbstractController
from commons.commons import *
import numpy as np
from scipy.stats import skew, kurtosis
# import datetime
# from dateutil import parser
import logging
import time

# from copy import deepcopy

try:
    from gensim import *

    lda_model = models.ldamodel
except:
    print("WARNING! gensim is not available! This module is not usable.")

from operator import itemgetter

from DB.schema_definition import *

'''
This class is responsible for generating features based on claim's posts properties
Each post-feature pair will be written in the AuthorFeature table
'''


class LDATopicFeatureGenerator(BaseFeatureGenerator):
    def __init__(self, db, **kwargs):
        BaseFeatureGenerator.__init__(self, db, **kwargs)
        self._features_list = self._config_parser.eval(self.__class__.__name__, "features_list")
        self._prefix = self.__class__.__name__
        self._targeted_fields = self._config_parser.eval(self.__class__.__name__, "targeted_fields")
        self._num_of_terms_in_topic = self._config_parser.eval(self.__class__.__name__, "num_of_terms_in_topic")
        self.num_topics = self._config_parser.eval(self.__class__.__name__, "number_of_topics")
        self.stopword_file = self._config_parser.get(self.__class__.__name__, "stopword_file")
        self.stemlanguage = self._config_parser.get(self.__class__.__name__, "stem_language")
        self.topics = []
        self.topic = None
        self.post_id_to_topic_id = None
        self.topic_id_to_topics = {}
        self.model = None
        self.dictionary = None
        self.corpus = None

    def execute(self, window_start=None):
        start_time = time.time()
        info_msg = "execute started for LDA topic feature generator started at " + str(start_time)
        logging.info(info_msg)
        for targeted_fields_dict in self._targeted_fields:
            author_features = []
            print("Get sourse id target dict")
            i = 1
            for source_id_destination_target_fields in self.get_source_id_destination_target_field(
                    targeted_fields_dict):
                for source_id, target_fields in source_id_destination_target_fields.items():
                    print("\rextract LDA topic features {}/{}"
                          .format(i, len(source_id_destination_target_fields)), end="")
                    i += 1
                    words_list = list(map(self._text_to_words, target_fields))
                    source_topics = self._calculate_topics(words_list)
                    topics = set(source_topics)
                    topic_words_dict = {}
                    for topic_id in topics:
                        topic = self.model.show_topic(topic_id, self._num_of_terms_in_topic)
                        topic_words_dict[topic_id] = list(map(itemgetter(0), topic))
                    kwargs = {'topics': source_topics, 'topic_words': topic_words_dict}
                    for feature in self._features_list:
                        author_feature = self.run_and_create_author_feature(kwargs, source_id, feature)
                        if author_feature:
                            author_features.append(author_feature)
            print()
            self._db.add_author_features(author_features)
            self._db.session.commit()
        stop_time = time.time()
        info_msg = "execute ended at " + str(stop_time)
        logging.info(info_msg)

    def cleanUp(self):
        pass

    def _text_to_words(self, element_text):
        return clean_tweet(element_text).split()

    def _calculate_topics(self, words_list):
        words = words_list
        self.dictionary = corpora.Dictionary(words)
        self.corpus = [self.dictionary.doc2bow(content_words) for content_words in words]
        self.model = lda_model.LdaModel(self.corpus, num_topics=self.num_topics)
        self.topic_id_to_topics = {}
        self.topics = []
        post_topic_max = []
        for words in words_list:
            content_words = words
            bow = self.dictionary.doc2bow(content_words)
            topic_id_to_probability = self.model.get_document_topics(bow)
            max_topic_probability = max(topic_id_to_probability, key=lambda item: item[1])
            post_topic_max.append(max_topic_probability[0])
        return post_topic_max

    """ Aggregated features from list """

    def topic_std_dev(self, **kwargs):
        return np.std(kwargs['topics'])

    def topic_skewness(self, **kwargs):
        return skew(kwargs['topics'])

    def topic_kurtosis(self, **kwargs):
        return kurtosis(kwargs['topics'])

    def topic_uniqueness(self, **kwargs):
        pass
