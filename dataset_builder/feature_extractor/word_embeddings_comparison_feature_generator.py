
from dataset_builder.feature_extractor.base_feature_generator import BaseFeatureGenerator
from dataset_builder.word_embedding.Vectors_Operations import Vector_Operations


class Word_Embeddings_Comparison_Feature_Generator(BaseFeatureGenerator):
    def __init__(self, db, **kwargs):
        BaseFeatureGenerator.__init__(self, db, **kwargs)
        self._connection_types = self._config_parser.eval(self.__class__.__name__, "connection_types")
        self._similarity_functions = self._config_parser.eval(self.__class__.__name__, "similarity_functions")
        self._word_embedding_table_name = self._config_parser.eval(self.__class__.__name__, "word_embedding_table_name")

    def execute(self):
        i = 0
        for connection in self._connection_types:
            i += 1

            first_field = connection[0]
            first_table_name = first_field["table_name"]
            first_targeted_field_name = first_field["targeted_field_name"]
            first_word_embedding_type = first_field["word_embedding_type"]

            second_field = connection[1]
            second_table_name = second_field["table_name"]
            second_targeted_field_name = second_field["targeted_field_name"]
            second_word_embedding_type = second_field["word_embedding_type"]

            print('\r {0}/{1} Current connection:{2}_{3}_{4}-{5}_{6}_{7}'.format(i, len(self._connection_types),
                                                                                 first_table_name,
                                                                                 first_targeted_field_name,
                                                                                 first_word_embedding_type,
                                                                                 second_table_name,
                                                                                 second_targeted_field_name,
                                                                                 second_word_embedding_type
                                                                                 ),
                  end='')

            first_author_guid_word_embedding_vector_dict = self._db.get_author_guid_word_embedding_vector_dict(
                self._word_embedding_table_name, first_table_name, first_targeted_field_name, first_word_embedding_type)
            second_author_guid_word_embedding_vector_dict = self._db.get_author_guid_word_embedding_vector_dict(
                self._word_embedding_table_name, second_table_name, second_targeted_field_name,
                second_word_embedding_type)

            for function in self._similarity_functions:
                if function == "subtruct_and_split":
                    authors_features = Vector_Operations.create_subtruction_dimension_features_from_authors_dict(
                        first_author_guid_word_embedding_vector_dict, second_author_guid_word_embedding_vector_dict,
                        first_table_name, first_targeted_field_name, first_word_embedding_type,
                        second_table_name, second_targeted_field_name, second_word_embedding_type,
                        self._window_start, self._window_end, self.__class__.__name__ + '_')
                else:
                    authors_features = Vector_Operations.create_authors_feature_from_two_vectors(
                        function, first_author_guid_word_embedding_vector_dict,
                        second_author_guid_word_embedding_vector_dict, first_table_name, first_targeted_field_name,
                        first_word_embedding_type, second_table_name, second_targeted_field_name,
                        second_word_embedding_type, self._window_start, self._window_end, self.__class__.__name__ + '_')

                self.insert_author_features_to_db(authors_features)  # create in batches
