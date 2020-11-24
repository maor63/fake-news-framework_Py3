# Written by Lior Bass 3/22/2018


import logging
import numpy as np
from commons import commons
from dataset_builder.feature_extractor.base_feature_generator import BaseFeatureGenerator
from dataset_builder.word_embedding.Vectors_Operations import Vector_Operations
from preprocessing_tools.abstract_controller import AbstractController


class Word_Embedding_Differential_Feature_Generator(AbstractController):
    def __init__(self, db, **kwargs):
        AbstractController.__init__(self, db)
        self._db = db
        self._word_vector_dict = {}
        self._table_name = self._config_parser.eval(self.__class__.__name__, "table_name")
        self._aggregation_functions = self._config_parser.eval(self.__class__.__name__, "aggregation_functions")
        self._pairs_targets = self._config_parser.eval(self.__class__.__name__, "pairs_targets")
        self._distance_functions = self._config_parser.eval(self.__class__.__name__, "distance_functions")

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def is_well_defined(self):
        return True

    def execute(self, window_start=None):
        self._word_vector_dict = self._db.get_word_vector_dictionary(self._table_name)

        for target1, target2 in self._pairs_targets:
            features = []
            target1_tuples = self._get_records_by_target_dict(target1)
            target2_tuples = self._get_records_by_target_dict(target2)
            for id in target1_tuples:
                dif_set1, dif_set2 = self.get_word_differences(id, target1_tuples, target2_tuples)

                for aggregation_function in self._aggregation_functions:
                    try:
                        dif1_word_embedding, dif2_word_embedding, subtraction_vec = self._get_differential_vectors(
                            aggregation_function, dif_set1, dif_set2)

                        feature_name = self._get_feature_names(target1, target2, aggregation_function)

                        features = features + Vector_Operations.create_author_feature_for_each_dimention(
                            subtraction_vec, feature_name, id, self._window_start, self._window_end,
                            self.__class__.__name__ + '_')
                        features = features + self.create_distance_features(id, aggregation_function,
                                                                            dif1_word_embedding,
                                                                            dif2_word_embedding, target1, target2, self.__class__.__name__ + '_')
                    except Exception as e1:
                        logging.info(e1)
            for feature in features:
                feature.attribute_value = float(feature.attribute_value)
            self._db.add_author_features_fast(features)

    def create_distance_features(self, author_id, aggregation_function, word_embedding_vector1, dif2_word_embedding,
                                 target1, target2, prefix=''):
        distance_features = []
        for distance_function in self._distance_functions:
            feature_name = prefix + 'differential_' + "distance_function_" + distance_function + '_' + target1[
                'table_name'] + "_" + target1['targeted_field_name'] + "_" + str(aggregation_function) + "_TO_" \
                           + target2['table_name'] + "_" + target2['targeted_field_name'] + "_" + str(
                aggregation_function)

            attribute_value = Vector_Operations.oparate_on_two_vectors(commons, distance_function,
                                                                       word_embedding_vector1,
                                                                       dif2_word_embedding)
            feature = BaseFeatureGenerator.create_author_feature(feature_name, author_id,
                                                                 attribute_value,
                                                                 self._window_start,
                                                                 self._window_end)
            distance_features.append(feature)
        return distance_features

    def _get_differential_vectors(self, aggregation_function, word_set_1, word_set_2):
        vector1 = self._collect_word_vector(word_set_1, aggregation_function)
        vector2 = self._collect_word_vector(word_set_2, aggregation_function)
        if vector1 == []:
            vector1 = [0] * 300
        if vector2 == []:
            vector2 = [0] * 300
        subtraction_vector = list(vector1[i] - vector2[i] for i in range(len(vector1)))
        return vector1, vector2, subtraction_vector

    def get_word_differences(self, id, target1_tuples, target2_tuples):
        text1 = target1_tuples[id]
        text2 = target2_tuples[id]
        set1 = self._get_set_from_string(text1)
        set2 = self._get_set_from_string(text2)
        dif1, dif2 = self._get_differentials(set1, set2)
        return dif1, dif2

    def _get_feature_names(self, target1, target2, aggregation_function):
        name_prefix = 'differential'
        name1 = self._get_key_string_from_target_dict(target1)
        name2 = self._get_key_string_from_target_dict(target2)
        result1 = name_prefix + '_' + str(aggregation_function) + '_' + name1 + '_to_' + name2
        return result1

    def _get_key_string_from_target_dict(self, target):
        s = target['table_name'] + '_' + target['targeted_field_name']
        return s

    def _get_differentials(self, set1, set2):
        return set1.difference(set2), set2.difference(set1)

    def _get_set_from_string(self, text):
        try:
            return set(text.split(" "))
        except:
            return set()

    def _get_records_by_target_dict(self, targeted_fields_dict):
        table_name = targeted_fields_dict["table_name"]
        id_field = targeted_fields_dict["id_field"]
        targeted_field_name = targeted_fields_dict["targeted_field_name"]
        where_clauses = targeted_fields_dict["where_clauses"]
        tupples = self._db.get_records_by_id_targeted_field_and_table_name(id_field, targeted_field_name, table_name,
                                                                           where_clauses)
        result = {tuple[0]: tuple[1] for tuple in tupples}
        return result

    def _collect_word_vector(self, words, aggregation_function):
        word_vectors = []
        for word in words:
            if word in self._word_vector_dict:
                word_vector = self._word_vector_dict[word]
                word_vectors.append(word_vector)
            else:
                clean_word = commons.clean_word(word)
                if clean_word in self._word_vector_dict:
                    word_vector = self._word_vector_dict[clean_word]
                    word_vectors.append(word_vector)
        word_vectors = list(zip(*word_vectors))
        function = eval(aggregation_function)
        result = list(map(function, word_vectors))
        return result
