# Aviad Elyashar

import logging
from dataset_builder.graph_builder import GraphBuilder
import time
from itertools import *
import pandas as pd
import numpy
from math import*
from decimal import Decimal

from commons import commons
class GraphBuilder_Word_Embedding(GraphBuilder):
    def __init__(self, db):
        GraphBuilder.__init__(self, db)
        self._connection_types = self._config_parser.eval(self.__class__.__name__, "connection_types")
        self._similarity_functions = self._config_parser.eval(self.__class__.__name__, "similarity_functions")
        self._word_embedding_table_name = self._config_parser.eval(self.__class__.__name__,
                                                                   "word_embedding_table_name")

    def execute(self, window_start=None):
        start_time = time.time()
        logging.info("execute started for " + self.__class__.__name__ + " started at " + str(start_time))
        logging.info("getting posts from DB ")

        for connection_type_dict in self._connection_types:
            table_name = connection_type_dict["table_name"]
            targeted_field_name = connection_type_dict["targeted_field_name"]
            word_embedding_type = connection_type_dict["word_embedding_type"]

            if self._num_of_random_authors_for_graph is None:
                author_guid_word_embedding_vector_dict = self._db.get_author_guid_word_embedding_vector_dict(
                    self._word_embedding_table_name, table_name, targeted_field_name, word_embedding_type)
            else:
                author_guid_word_embedding_vector_dict = self._db.get_random_author_guid_word_embedding_vector_dict(
                    self._word_embedding_table_name, table_name, targeted_field_name, word_embedding_type,
                    self._num_of_random_authors_for_graph)

            logging.info("computing similarity between word_embedding ")

            df = pd.DataFrame.from_dict(author_guid_word_embedding_vector_dict, orient='index')
            normalized_df = (df - df.min()) / (df.max() - df.min())

            for similarity_function in self._similarity_functions:
                self._connection_type = table_name + "_" + targeted_field_name + "_" + word_embedding_type + "_" + similarity_function
                print("calculating {}".format(self._connection_type))

                all_pairs, total_combinations = self._create_combinations(author_guid_word_embedding_vector_dict)

                getattr(self, similarity_function)(normalized_df, total_combinations, all_pairs)

                end_time = time.time()
                duration = end_time - start_time
                logging.info(" total time taken " + str(duration))

    def compute_jaccard_index(self, set_1, set_2):
        n = len(set_1.intersection(set_2))
        return n / float(len(set_1) + len(set_2) - n)

    def jaccard_similarity(self, normlized_df, total_combinations, all_pairs):
        author_connections = []
        i = 0
        for author_a, author_b in all_pairs:
            author_a_vector = normlized_df.loc[author_a]
            author_b_vector = normlized_df.loc[author_b]

            #author_a_vector = author_guid_word_embedding_vector_dict[author_a]
            #author_b_vector = author_guid_word_embedding_vector_dict[author_b]

            # weight = self.compute_jaccard_index(author_a_vector, author_b_vector)
            weight = commons.jaccard_index(author_a_vector, author_b_vector)
            author_connections.append((author_a, author_b, weight))
            i += 1
            if i % self._max_objects_without_saving == 0:
                print('\r done pair ' + str(i) + ' out of ' + str(total_combinations), end='')
                self._fill_author_connections(author_connections)
                author_connections = []
        self._fill_author_connections(author_connections)


    def cosine_similarity(self, normlized_df, total_combinations, all_pairs):
        author_connections = []
        i = 0
        for author_a, author_b in all_pairs:
            #author_a_vector = author_guid_word_embedding_vector_dict[author_a]
            #author_b_vector = author_guid_word_embedding_vector_dict[author_b]

            author_a_vector = normlized_df.loc[author_a]
            author_b_vector = normlized_df.loc[author_b]


            #author_a_vector = set(author_a_vector)
            #author_b_vector = set(author_b_vector)

            # weight = self._compute_cosine_similarity(author_a_vector, author_b_vector)
            weight = commons.cosine_similarity(author_a_vector, author_b_vector)
            author_connections.append((author_a, author_b, weight))
            i += 1
            if i % self._max_objects_without_saving == 0:
                print('\r done pair ' + str(i) + ' out of ' + str(total_combinations), end='')
                self._fill_author_connections(author_connections)
                author_connections = []
        self._fill_author_connections(author_connections)


    def euclidean_distance(self, normlized_df, total_combinations, all_pairs):
        author_a_author_b_distance_dict = self._create_author_a_author_b_euclidian_distance_dict(all_pairs, normlized_df)
        self._calculate_measure_normlize_and_save_connections(author_a_author_b_distance_dict, total_combinations)

    def manhattan_distance(self, author_guid_word_embedding_vector_dict, total_combinations, all_pairs):
        author_a_author_b_distance_dict = self._create_author_a_author_b_manhattan_distance_dict(all_pairs, author_guid_word_embedding_vector_dict)
        self._calculate_measure_normlize_and_save_connections(author_a_author_b_distance_dict, total_combinations)

    def minkowski_distance(self, author_guid_word_embedding_vector_dict, total_combinations, all_pairs):
        author_a_author_b_distance_dict = self._create_author_a_author_b_minkowski_distance_dict(all_pairs, author_guid_word_embedding_vector_dict)
        self._calculate_measure_normlize_and_save_connections(author_a_author_b_distance_dict, total_combinations)

    def _create_author_a_author_b_euclidian_distance_dict(self, all_pairs, normlized_df):
        author_a_author_b_distance_dict = {}
        i = 0
        for author_a, author_b in all_pairs:
            i += 1
            author_a_vector = normlized_df.loc[author_a]
            author_b_vector = normlized_df.loc[author_b]

            #author_a_vector = author_guid_word_embedding_vector_dict[author_a]
            #author_b_vector = author_guid_word_embedding_vector_dict[author_b]

            #author_a_vector = numpy.array(author_a_vector)
            #author_b_vector = numpy.array(author_b_vector)


            # calculating euclidian distance
            # eucli_distance = numpy.linalg.norm(author_a_vector - author_b_vector)
            eucli_distance = commons.euclidean_distance(author_a_vector, author_b_vector)
            author_a_author_b_distance_dict[author_a + "_" + author_b] = eucli_distance
        return author_a_author_b_distance_dict

    def _create_author_a_author_b_manhattan_distance_dict(self, all_pairs, normlized_df):
        author_a_author_b_distance_dict = {}
        i = 0
        for author_a, author_b in all_pairs:
            i += 1
            author_a_vector = normlized_df.loc[author_a]
            author_b_vector = normlized_df.loc[author_b]

            #author_a_vector = author_guid_word_embedding_vector_dict[author_a]
            #author_b_vector = author_guid_word_embedding_vector_dict[author_b]

            #author_a_vector = numpy.array(author_a_vector)
            #author_b_vector = numpy.array(author_b_vector)
            # calculating manhattan distance
            # man_distance = sum(abs(a - b) for a,b in zip(author_a_vector, author_b_vector))
            man_distance = commons.minkowski_distance(author_a_vector, author_b_vector)
            author_a_author_b_distance_dict[author_a + "_" + author_b] = man_distance
        return author_a_author_b_distance_dict

    def _calculate_measure_normlize_and_save_connections(self, author_a_author_b_distance_dict, total_combinations):
        df = pd.DataFrame.from_dict(author_a_author_b_distance_dict, orient='index')
        normalized_df = (df - df.min()) / (df.max() - df.min())
        j = 0
        author_connections = []
        for connection, normilized_distance_series in normalized_df.iterrows():
            authors = connection.split("_")
            author_a = authors[0]
            author_b = authors[1]
            normilized_distance = normilized_distance_series.ix[0]
            author_connections.append((author_a, author_b, normilized_distance))
            j += 1
            if j % self._max_objects_without_saving == 0:
                print('\r done pair ' + str(j) + ' out of ' + str(total_combinations), end='')
                self._fill_author_connections(author_connections)
                author_connections = []

        self._fill_author_connections(author_connections)

    def _square_rooted(self, x):
        return round(sqrt(sum([a * a for a in x])), 3)

    def _compute_cosine_similarity(self, vector_a, vector_b):
        numerator = sum(a * b for a, b in zip(vector_a, vector_b))
        denominator = self._square_rooted(vector_a) * self._square_rooted(vector_b)
        return round(numerator / float(denominator), 3)

    def _create_author_a_author_b_minkowski_distance_dict(self, all_pairs, normlized_df):
        author_a_author_b_distance_dict = {}
        i = 0
        for author_a, author_b in all_pairs:
            i += 1
            author_a_vector = normlized_df.loc[author_a]
            author_b_vector = normlized_df.loc[author_b]
            #author_a_vector = author_guid_word_embedding_vector_dict[author_a]
            #author_b_vector = author_guid_word_embedding_vector_dict[author_b]

            #author_a_vector = numpy.array(author_a_vector)
            #author_b_vector = numpy.array(author_b_vector)
            # calculating minkowski distance
            # mink_distance = self._compute_minkowski_distance(author_a_vector, author_b_vector, 3)
            mink_distance = commons.minkowski_distance(author_a_vector, author_b_vector)
            author_a_author_b_distance_dict[author_a + "_" + author_b] = mink_distance
        return author_a_author_b_distance_dict

    def _nth_root(self, value, n_root):
        root_value = 1 / float(n_root)
        return round(Decimal(value) ** Decimal(root_value), 3)

    def _compute_minkowski_distance(self, vector_a, vector_b, p_value):
        return self._nth_root(sum(pow(abs(a - b), p_value) for a, b in zip(vector_a, vector_b)), p_value)

    def _create_combinations(self, author_guid_word_embedding_vector_dict):
        author_guids = list(author_guid_word_embedding_vector_dict.keys())
        total_number_of_authors = len(author_guids)
        total_combinations = (total_number_of_authors * (total_number_of_authors - 1)) / 2

        all_pairs = combinations(author_guids, 2)

        return all_pairs, total_combinations