
import numpy as np
import pandas as pd
from functools import partial
from commons import commons
from commons.commons import get_words_by_content
from dataset_builder.feature_extractor.base_feature_generator import BaseFeatureGenerator
from DB.schema_definition import *


class AbstractWordEmbaddingTrainer(BaseFeatureGenerator):
    def __init__(self, db, **kwargs):
        super(AbstractWordEmbaddingTrainer, self).__init__(db, **kwargs)
        self._word_vector_dict = pd.DataFrame()
        self._aggregation_functions_names = self._config_parser.eval(self.__class__.__name__,
                                                                     "aggregation_functions_names")
        self._table_name = self._config_parser.eval(self.__class__.__name__, "table_name")
        self._targeted_fields_for_embedding = self._config_parser.eval(self.__class__.__name__,
                                                                       "targeted_fields_for_embedding")
        self._num_of_dimensions = self._config_parser.eval(self.__class__.__name__, "num_of_dimensions")
        self._word_to_clean_word = {}

    def setUp(self):
        self._db.drop_table(self._get_table_name())


    def _calculate_word_embedding_to_authors(self, source_id_elements_dict, targeted_fields_dict, word_vector_dict):
        source_id_words_dict = self._fill_source_id_words_dictionary(source_id_elements_dict, targeted_fields_dict)
        # self._word_vector_dict = word_vector_dict
        word_vector_dict = pd.DataFrame(
            {word: np.array(vector, dtype=np.float) for word, vector in word_vector_dict.items()}, dtype=np.float)
        word_embeddings = []

        source_count = len(source_id_words_dict)
        i = 0
        id_field, table_name, targeted_field_name = self.get_fields(targeted_fields_dict)
        for source_id, words in source_id_words_dict.items():
            msg = "\rCalculating word embeddings: {1}/{2}".format(source_id, str(i + 1), source_count)
            print(msg, end='')

            word_vectors = self._collect_word_vector_per_source(words, word_vector_dict)
            word_embeddings.append((source_id, id_field, table_name, targeted_field_name, word_vectors))
            i += 1
        print()
        return word_embeddings

    def _add_word_embeddings_to_db(self, word_embeddings):
        print("Add word embedding to DB")
        self._results_dataframe = pd.DataFrame()
        if len(word_embeddings) > 0:
            self._add_word_embeddings_to_df(word_embeddings)
            # if result is None: #added by Lior, need to check for if no author_id
            #     self._fill_zeros(results_dataframe, author_id, table_name, id_field, targeted_field_name)
            column_names = ["author_id", "table_name", "id_field", "targeted_field_name", "word_embedding_type"]
            dimensions = np.arange(self._num_of_dimensions)
            column_names.extend(dimensions)
            self._results_dataframe.columns = column_names
            engine = self._db.engine
            self._results_dataframe.to_sql(name=self._get_table_name(),
                                           con=engine, index=False, if_exists='append')

    def _get_table_name(self):
        return "author_word_embeddings_{0}_{1}_dim".format(self._table_name,
                                                           self._num_of_dimensions)

    def _add_word_embeddings_to_df(self, word_embeddings):
        rows = []
        for i, word_embedding in enumerate(word_embeddings):
            if i % 10 == 0 or i == len(word_embeddings) - 1:
                print("\rAdd word embedding to DF {0}/{1}".format(str(i + 1), len(word_embeddings)), end='')
            rows += self._fill_results_dataframe(*word_embedding)
        self._results_dataframe = pd.DataFrame(rows)
        print()

    def _merge_results_with_existing_table(self, result):
        pass

    def _fill_zeros(self, results_dataframe, author_id, table_name, id_field, targeted_field_name):
        for aggregation_function_name in self._aggregation_functions_names:
            author_vector = [author_id, table_name, id_field, targeted_field_name,
                             aggregation_function_name]
            zero_vector = np.zeros((300,), dtype=np.int)
            author_vector.extend(zero_vector)
            series = pd.Series(data=author_vector)
            results_dataframe = results_dataframe.append(series, ignore_index=True)

    def _fill_source_id_words_dictionary(self, source_id_target_fields_dict, targeted_fields_dict):
        source_id_words_dict = {}
        print("Starting fill_author_id_words_dictionary")

        i = 1
        source_count = len(source_id_target_fields_dict)
        for source_id, target_elements in source_id_target_fields_dict.items():
            msg = "\rFilling author_words_dict: {1}/{2}".format(source_id, i, source_count)
            print(msg, end='')
            i += 1
            total_words = []
            for target_element in target_elements:
                if "destination" in targeted_fields_dict and "target_field" in targeted_fields_dict["destination"]:
                    text = getattr(target_element, targeted_fields_dict["destination"]["target_field"])
                else:
                    text = getattr(target_element, targeted_fields_dict["source"]["target_field"])
                if text is not None:
                    # print("author_guid = " + author_id, "text =" + text)
                    words = get_words_by_content(text)
                    total_words += words
            source_id_words_dict[source_id] = total_words
        print()
        print("Finishing fill_author_id_words_dictionary")
        return source_id_words_dict

    def _collect_word_vector_per_source(self, words, word_vector_dict):
        clean_words = []
        for word in words:
            if word not in self._word_to_clean_word:
                self._word_to_clean_word[word] = commons.remove_punctuation_chars(word)
            clean_words.append(self._word_to_clean_word[word])
        word_vectors = word_vector_dict[word_vector_dict.columns & clean_words]
        word_vectors = word_vectors.T
        return word_vectors

    def _fill_results_dataframe(self, author_id, id_field, table_name, targeted_field_name, transposed):
        # id_field, table_name, targeted_field_name = self.get_fields(targeted_fields_dict)

        word_embedding_df_rows = []
        for aggregation_function_name in self._aggregation_functions_names:
            author_vector = [author_id, table_name, id_field, targeted_field_name,
                             aggregation_function_name]
            result = getattr(transposed, aggregation_function_name)(axis=0)
            result.fillna(0.0, inplace=True)
            if len(result) > 0:
                author_vector.extend(result)
            else:
                if len(transposed) <= 0:
                    dimensions = self._num_of_dimensions
                else:
                    dimensions = len(transposed)
                zero_vector = np.zeros((dimensions,), dtype=np.int)
                author_vector.extend(zero_vector)
            word_embedding_df_rows.append(author_vector)
        return word_embedding_df_rows

    def get_fields(self, targeted_fields_dict):
        table_name = targeted_fields_dict['source']["table_name"]
        id_field = targeted_fields_dict['source']["id"]
        if 'destination' in targeted_fields_dict and targeted_fields_dict['destination'] != {}:
            targeted_field_name = targeted_fields_dict['destination']["target_field"]
            where_clauses = targeted_fields_dict['destination']["where_clauses"]
        else:
            targeted_field_name = targeted_fields_dict['source']["target_field"]
            where_clauses = targeted_fields_dict['source']["where_clauses"]
        if len(where_clauses) > 0:
            where_clause_dict = where_clauses[0]
            values = list(where_clause_dict.values())
            additional_str = ""
            if values[0] != 1 and values[1] != 1:
                for value in values:
                    additional_str += "_" + value
            if len(additional_str) > 0:
                targeted_field_name = additional_str + "_" + targeted_field_name
        return id_field, table_name, targeted_field_name
