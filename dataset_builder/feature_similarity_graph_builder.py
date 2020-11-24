'''
@author: Jorge Bendahan jorgeaug@post.bgu.ac.il
'''

import time
import logging
from dataset_builder.graph_builder import GraphBuilder
from itertools import combinations
from commons.data_frame_creator import DataFrameCreator
from commons.consts import Connection_Type
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy import spatial


class GraphBuilder_Feature_Similarity(GraphBuilder):
    """Generate graphs where nodes represent authors.
    There exists an edge between two authors if their feature vectors are 'close enough' (measured by cosine similarity,
    'close enough' threshold is defined in config.ini"""

    def __init__(self, db):
        GraphBuilder.__init__(self, db)
        self._connection_types = self._config_parser.eval(self.__class__.__name__, "connection_types")
        self._topic_distr_features_min_distance_threshold = \
            self._config_parser.eval(self.__class__.__name__, "topic_distr_features_min_distance_threshold")
        self._topic_distr_features = \
            self._config_parser.eval(self.__class__.__name__, "topic_distr_features")

        self._profile_prop_features_min_distance_threshold = \
            self._config_parser.eval(self.__class__.__name__, "profile_prop_features_min_distance_threshold")
        self._profile_prop_features = \
            self._config_parser.eval(self.__class__.__name__, "profile_prop_features")


    def setUp(self):
        #load all features from author_features table
        dataframe_creator = DataFrameCreator(self._db)
        dataframe_creator.execute(window_start=self._window_start)
        self._authors_features = dataframe_creator.get_author_features_data_frame()

        if self._num_of_random_authors_for_graph is not None and not self._are_already_randomize_authors_for_graphs():
            self._db.create_author_guid_num_of_posts_view()
            self._db.randomize_authors_for_graph(self._min_number_of_posts_per_author, self._domain,
                                                 self._num_of_random_authors_for_graph)


    def execute(self, window_start=None):
        start_time = time.time()
        logging.info("execute started for " + self.__class__.__name__ + " started at " + str(start_time))

        author_guids = list(self._authors_features.index.values)
        if self._num_of_random_authors_for_graph is not None:
            random_author_guid_post_id_dict = self._db.get_random_author_guid_post_id_dictionary()
            random_author_features = []
            for author_guid in author_guids:
                if author_guid in random_author_guid_post_id_dict:
                    random_author_features.append(author_guid)
            author_guids = random_author_features

        for connection_type in self._connection_types:

            pairs = combinations(author_guids, 2)
            author_guid_length = len(author_guids)
            num_of_pairs = author_guid_length * (author_guid_length - 1) / 2

            self._connection_type = connection_type

            logging.info("getting "+connection_type+" features from DB ")

            if self._connection_type == Connection_Type.TOPIC_DISTR_SIMILARITY:
                author_features = self._authors_features[self._topic_distr_features].apply(pd.to_numeric)
                min_distance = float(self._topic_distr_features_min_distance_threshold)
            else:
                author_features = self._authors_features[self._profile_prop_features].apply(pd.to_numeric)
                min_distance = float(self._profile_prop_features_min_distance_threshold)

            # replace None with NaN
            author_features.fillna(value=np.nan, inplace=True)

            # fill Nan with column average
            author_features = author_features.apply(lambda x: x.fillna(x.mean()), axis=0)

            # drop columns without values
            author_features.dropna(axis=1, how='all', inplace=True)

            # drop columns with constant value for all observations
            author_features = author_features.loc[:, (author_features != author_features.ix[0]).any()]

            # Normalize features
            normalized_author_features = pd.DataFrame(preprocessing.normalize(author_features, axis=0),
                                                      index=author_features.index)
            i = 0
            for author_guid_a, author_guid_b in pairs:
                i += 1
                msg = '\r {0}/{1}'.format(i, num_of_pairs)
                print(msg, end="")
                vector_author_guid_a = normalized_author_features.loc[author_guid_a]
                vector_author_guid_b = normalized_author_features.loc[author_guid_b]
                distance = spatial.distance.cosine(list(map(float, vector_author_guid_a)),
                                                   list(map(float, vector_author_guid_b)))

                if distance <= min_distance:
                    weight = 1 - distance
                    self._create_and_optional_save_connection(author_guid_a, author_guid_b, weight)
