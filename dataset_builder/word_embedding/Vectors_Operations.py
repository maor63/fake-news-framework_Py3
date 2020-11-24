

from collections import deque

import numpy as np
from functools import partial
import commons
from dataset_builder.feature_extractor.base_feature_generator import BaseFeatureGenerator
from operator import sub
import logging.config


class Vector_Operations():
    @staticmethod
    def create_authors_feature_from_two_vectors(func, first_author_vector_dict, second_author_vector_dict,
                                                first_table_name,
                                                first_targeted_field_name, first_word_embedding_type, second_table_name,
                                                second_targeted_field_name, second_word_embedding_type, window_start,
                                                window_end, prefix=''):
        authors_features = []
        for author_id in list(first_author_vector_dict.keys()):
            feature_name = prefix + 'subtraction_' + first_table_name + "_" + first_targeted_field_name + "_" + first_word_embedding_type + "_TO_" \
                           + second_table_name + "_" + second_targeted_field_name + "_" + second_word_embedding_type + "_DISTANCE-FUNCTION_" + func
            first_vector = first_author_vector_dict[author_id]
            second_vector = second_author_vector_dict[author_id]
            # attribute_value = getattr(commons.commons, func)(first_vector, second_vector
            attribute_value = Vector_Operations.oparate_on_two_vectors(commons.commons, func,
                                                                       first_vector,
                                                                       second_vector)
            feature = BaseFeatureGenerator.create_author_feature(feature_name, author_id, attribute_value,
                                                                 window_start,
                                                                 window_end)
            authors_features.append(feature)
        return authors_features

    @staticmethod
    def oparate_on_two_vectors(func_location, func, vector_1, vector_2):
        value = getattr(func_location, func)(vector_1, vector_2)
        return value

    @staticmethod
    def create_features_from_word_embedding_dict(author_guid_word_embedding_dict, targeted_table, targeted_field_name,
                                                 targeted_word_embedding_type, word_embedding_table_name,
                                                 window_start, window_end, db, commit_treshold, prefix=''):
        authors_features = deque()
        author_guids = list(author_guid_word_embedding_dict.keys())
        kwargs = {'window_start': window_start, 'window_end': window_end, 'prefix_name': prefix}
        create_author_feature_fn = partial(Vector_Operations.create_author_feature_for_each_dimention, **kwargs)
        for i, author_guid in enumerate(author_guids):
            msg = "\rCalculating word embeddings features: {0}/{1}:{2}".format(i, len(author_guids), author_guid)
            print(msg, end="")
            if len(authors_features) > commit_treshold:
                db.add_author_features(authors_features)
                authors_features = deque()
            author_vector = author_guid_word_embedding_dict[author_guid]
            feature_name = "{0}_{1}_{2}_{3}".format(targeted_word_embedding_type, targeted_table, targeted_field_name,
                                                    word_embedding_table_name)
            dimentions_feature_for_author = create_author_feature_fn(author_vector, feature_name, author_guid)
            authors_features.extend(dimentions_feature_for_author)
        db.add_author_features(authors_features)

    @staticmethod
    def create_author_feature_for_each_dimention(vector, feature_name, author_guid, window_start, window_end,
                                                 prefix_name=''):
        authors_features = []
        for dimension_counter, dimension in enumerate(vector):
            dimension = round(dimension, 4)

            final_feature_name = prefix_name + feature_name + "_d" + str(dimension_counter)
            feature = BaseFeatureGenerator.create_author_feature(final_feature_name, author_guid, dimension,
                                                                 window_start,
                                                                 window_end)
            authors_features.append(feature)
        return authors_features

    @staticmethod
    def create_subtruction_dimension_features_from_authors_dict(first_author_guid_word_embedding_vector_dict,
                                                                second_author_guid_word_embedding_vector_dict,
                                                                first_table_name, first_targeted_field_name,
                                                                first_word_embedding_type, second_table_name,
                                                                second_targeted_field_name, second_word_embedding_type,
                                                                window_start, window_end, prefix=''):
        author_features = []
        for author_guid in list(first_author_guid_word_embedding_vector_dict.keys()):
            first_vector = first_author_guid_word_embedding_vector_dict[author_guid]
            second_vector = second_author_guid_word_embedding_vector_dict[author_guid]
            current_authors_feature = Vector_Operations.create_subtruction_dimention_features(first_vector,
                                                                                              second_vector,
                                                                                              first_table_name,
                                                                                              first_targeted_field_name,
                                                                                              first_word_embedding_type,
                                                                                              second_table_name,
                                                                                              second_targeted_field_name,
                                                                                              second_word_embedding_type,
                                                                                              window_start, window_end,
                                                                                              author_guid, prefix)
            author_features = author_features + current_authors_feature
        return author_features

    @staticmethod
    def create_subtruction_dimention_features(vector_1, vector_2, first_table_name, first_targeted_field_name,
                                              first_word_embedding_type, second_table_name,
                                              second_targeted_field_name, second_word_embedding_type,
                                              window_start, window_end, author_guid, prefix=''):
        result_vector = tuple(map(sub, vector_1, vector_2))
        feature_name = prefix + "subtraction_" + first_table_name + "_" + first_targeted_field_name + "_" + first_word_embedding_type + "_TO_" + \
                       second_table_name + "_" + second_targeted_field_name + "_" + second_word_embedding_type
        feature_id = author_guid
        author_features = Vector_Operations.create_author_feature_for_each_dimention(result_vector, feature_name,
                                                                                     feature_id,
                                                                                     window_start, window_end)
        return author_features
