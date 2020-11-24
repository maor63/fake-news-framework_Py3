

import numpy as np
import pandas as pd

from commons.commons import *
from dataset_builder.word_embedding.abstract_word_embadding_trainer import AbstractWordEmbaddingTrainer


class Word_Embedding_Model_Creator(AbstractWordEmbaddingTrainer):
    def __init__(self, db):
        super(Word_Embedding_Model_Creator, self).__init__(db, **{'authors': [], 'posts': {}})
        self._is_load_wikipedia_300d_glove_model = self._config_parser.eval(self.__class__.__name__,
                                                                            "is_load_wikipedia_300d_glove_model")
        self._wikipedia_model_file_path = self._config_parser.eval(self.__class__.__name__, "wikipedia_model_file_path")

        # key = author_id (could be author_guid, or other field) and array of the contents (maybe the author has many posts' content)
        self._word_vector_dict_full_path = "data/output/word_embedding/"

    def setUp(self):
        if not os.path.exists(self._word_vector_dict_full_path):
            os.makedirs(self._word_vector_dict_full_path)

    def execute(self, window_start=None):
        if self._is_load_wikipedia_300d_glove_model:
            if not self._db.is_table_exist(self._table_name):
                self._load_wikipedia_300d_glove_model()
            word_embeddings = []
            for targeted_fields_dict in self._targeted_fields_for_embedding:
                source_id_target_elements_dict = self._get_source_id_target_elements(targeted_fields_dict)
                word_vector_dict = self._db.get_word_vector_dictionary(self._table_name)
                word_embeddings = self._calculate_word_embedding_to_authors(source_id_target_elements_dict, targeted_fields_dict, word_vector_dict)
                self._add_word_embeddings_to_db(word_embeddings)

    def _load_wikipedia_300d_glove_model(self):
        logging.info("_load_wikipedia_300d_glove_model")
        dataframe = pd.DataFrame()

        with open(self._wikipedia_model_file_path, 'r') as file:
            i = 1
            for line in file:
                msg = '\r Reading line #' + str(i)
                print(msg, end="")
                i += 1
                word_vector_array = line.split(' ')
                word = str(word_vector_array[0], 'utf-8')
                vector_str = word_vector_array[1:]
                vector_str = np.array(vector_str)
                vector = vector_str.astype(np.float)
                dataframe[word] = vector

        dataframe = dataframe.T
        info_msg = "\r transposed dataframe"
        print(info_msg, end="")
        # add the index as column before removing the index
        dataframe['word'] = dataframe.index
        # Change the order - the index column now first.
        cols = dataframe.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        dataframe = dataframe[cols]

        info_msg = "\r  remove the index"
        print(info_msg, end="")

        engine = self._db.engine
        # index = false is important for removing index.
        dataframe.to_sql(name=self._table_name, con=engine, index=False)

        # save model
        if not os.path.exists(self._word_vector_dict_full_path):
            os.makedirs(self._word_vector_dict_full_path)

    # def _fill_author_id_text_dictionary(self, targeted_fields_dict):
    #     author_id_texts_dict =
    #
    #     return author_id_texts_dict

    # added by Lior

