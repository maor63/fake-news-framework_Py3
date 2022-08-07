#  Created by Asaf at 24/1/2019
#  Lst Modified by Asaf at 24/1/2019


import timeit

from dataset_builder.feature_extractor.base_feature_generator import BaseFeatureGenerator
import numpy as np
from scipy.stats import kurtosis, skew
import pandas as pd
from sklearn.metrics import jaccard_similarity_score
from sklearn.preprocessing import LabelEncoder

try:
    from gensim import *
    # Notice: bag_of_words_model, using 'dictionary.doc2bow' function.

except:
    print("WARNING! gensim is not available! This module is not usable.")

from DB.schema_definition import *


def contains_posts_decorator(func):
    def func_wrapper(*args, **kwargs):
        if 'posts' not in kwargs:
            raise ValueError('{} feature require posts as destination'.format(func))
        return func(*args, **kwargs)

    return func_wrapper


class CooperationTopicFeatureGenerator(BaseFeatureGenerator):

    def __init__(self, db, **kwargs):
        BaseFeatureGenerator.__init__(self, db, **kwargs)
        self._feature_list = self._config_parser.eval(self.__class__.__name__, "feature_list")
        self._prefix = self.__class__.__name__
        self.stemlanguage = self._config_parser.get(self.__class__.__name__, "stem_language")
        self._aggregated_functions = self._config_parser.eval(self.__class__.__name__, "aggregated_functions")
        self._jaccard_threshs = self._config_parser.eval(self.__class__.__name__, "jaccard_threshs")
        self._targeted_fields = self._config_parser.eval(self.__class__.__name__, "targeted_fields")
        self._jaccard_thresh = self._jaccard_threshs[0]
        self.topics = []
        self.topic = None
        self.post_id_to_topic_id = None
        self.topic_id_to_topics = {}
        self.dictionary = None
        self.corpus = None
        self._post_dict = {}
        self._features = self._config_parser.eval(self.__class__.__name__, "feature_list")
        self._features_names_count = len(self._features) * len(self._aggregated_functions) * len(self._jaccard_threshs)

    def execute(self, window_start=None):
        # Logger setup.
        start_time = time.time()
        info_msg = "execute started for Cooperation topic feature generator started at " + str(start_time)
        logging.info(info_msg)
        logging.info("Cooperation execute window_start %s" % self._window_start)
        self._post_dict = self._db.get_post_dictionary()
        self._jaccard_threshs = set(self._jaccard_threshs) | {1.0}
        aggregated_functions = list(map(eval, self._aggregated_functions))
        total_authors_features = []

        for targeted_fields_dict in self._targeted_fields:
            for source_id_element_dict, source_targets_dict in self._load_data_using_arg(targeted_fields_dict):
                for self._jaccard_thresh in self._jaccard_threshs:
                    start = timeit.default_timer()
                    print('Jaccared threshold {}'.format(self._jaccard_thresh))
                    suffix = 'jaccard_{}'.format(self._jaccard_thresh)
                    authors_features = self._get_features(source_id_element_dict, source_targets_dict,
                                                          suffix, targeted_fields_dict, aggregated_functions)
                    total_authors_features.extend(self._add_suffix_to_author_features(authors_features, suffix))
                    if len(total_authors_features) > self._max_objects_save:
                        self._db.add_author_features_fast(total_authors_features)
                        total_authors_features = []
                    end = timeit.default_timer()
                    print('Run time for jaccared thresh {} is: {} sec'.format(self._jaccard_thresh, end - start))

        self._db.add_author_features_fast(total_authors_features)
        self._db.session.commit()

    def cleanUp(self, **kwargs):
        pass

    def setUp(self):
        self._post_dict = self._db.get_post_dictionary()

    def _create_post_id_to_content_words(self, curr_posts):
        post_id_to_ngrams = {}
        for doc_id, post in enumerate(curr_posts):
            if post.content is not None:
                words = clean_tweet(post.content, self.stemlanguage)
                post_id_to_ngrams[post.post_id] = words.split()
        return post_id_to_ngrams

    def calculate_author_cooperation(self, post_id_to_words, jaccard_threshold=0.9):
        author_cooperation_counter_dict = defaultdict(set)
        # Iterates over all post_id and words pairs combinations only once.
        for post_id1, post_id2 in itertools.combinations(post_id_to_words, 2):
            first_author_id = self._post_dict[post_id1].author_guid
            second_author_id = self._post_dict[post_id2].author_guid
            j_score = jaccard_index(post_id_to_words[post_id1], post_id_to_words[post_id2])
            # Checks similarity.
            if first_author_id not in author_cooperation_counter_dict:
                author_cooperation_counter_dict[first_author_id] = set()
            if second_author_id not in author_cooperation_counter_dict:
                author_cooperation_counter_dict[second_author_id] = set()
            if first_author_id != second_author_id and j_score >= jaccard_threshold:
                author_cooperation_counter_dict[first_author_id].add(second_author_id)
                author_cooperation_counter_dict[second_author_id].add(first_author_id)
        return author_cooperation_counter_dict

    """ Aggregated features from list """

    @contains_posts_decorator
    def authors_cooperation(self, **kwargs):
        post_id_to_words = self._create_post_id_to_content_words(kwargs['posts'])
        authors_cooperation_dic = self.calculate_author_cooperation(post_id_to_words, self._jaccard_thresh)
        if authors_cooperation_dic == {}:
            return [0]
        else:
            return list(map(len, list(authors_cooperation_dic.values())))
