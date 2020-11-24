

import timeit

import numpy as np
import pandas as pd

from commons.commons import *
from dataset_builder.word_embedding.abstract_word_embadding_trainer import AbstractWordEmbaddingTrainer


class GloveWordEmbeddingModelCreator(AbstractWordEmbaddingTrainer):
    def __init__(self, db):
        super(GloveWordEmbeddingModelCreator, self).__init__(db, **{'authors': [], 'posts': {}})
        self._load_limit = self._config_parser.eval(self.__class__.__name__, "load_limit")
        self._is_load_wikipedia_300d_glove_model = self._config_parser.eval(self.__class__.__name__,
                                                                            "is_load_wikipedia_300d_glove_model")
        self._wikipedia_model_file_path = self._config_parser.eval(self.__class__.__name__, "wikipedia_model_file_path")
        # key = author_id (could be author_guid, or other field) and array of the contents (maybe the author has many posts' content)
        self._word_vector_dict_full_path = "data/output/word_embedding/"

    def setUp(self):
        # super(GloveWordEmbeddingModelCreator, self).setUp()
        if not os.path.exists(self._word_vector_dict_full_path):
            os.makedirs(self._word_vector_dict_full_path)

    def execute(self, window_start=None):
        if self._is_load_wikipedia_300d_glove_model:
            start = timeit.default_timer()
            if not self._db.is_table_exist(self._table_name):
                self._load_wikipedia_300d_glove_model()
                print()
                print('load glove table time: {} sec'.format(timeit.default_timer() - start))
            for targeted_fields_dict in self._targeted_fields_for_embedding:
                start = timeit.default_timer()
                # return generator
                # source_id_target_elements_dict = self._get_source_id_target_elements(targeted_fields_dict)
                total_sources = 0
                for i, source_id_target_elements_dict in enumerate(self._get_source_id_target_elements_generator(
                        targeted_fields_dict, self._load_limit)):
                    word_embeddings = []
                    print('create model for {} sources'.format(total_sources))
                    total_sources += len(source_id_target_elements_dict)
                    print('load dict params from db: {} sec'.format(timeit.default_timer() - start))
                    start = timeit.default_timer()
                    word_vector_dict = self._db.get_word_vector_dictionary(self._table_name)
                    print('load word vector dict from db: {} sec'.format(timeit.default_timer() - start))
                    start = timeit.default_timer()
                    word_embeddings += self._calculate_word_embedding_to_authors(source_id_target_elements_dict,
                                                                                targeted_fields_dict, word_vector_dict)
                    print('calculate word embeddings: {} sec'.format(timeit.default_timer() - start))
                    start = timeit.default_timer()
                    self._add_word_embeddings_to_db(word_embeddings)
                print('Add word embeddings to DB: {} sec'.format(timeit.default_timer() - start))

    def _load_wikipedia_300d_glove_model(self):
        logging.info("_load_wikipedia_300d_glove_model")
        vectors = []
        word_vector = []

        with open(self._wikipedia_model_file_path, 'r', encoding='utf-8') as file:
            i = 1
            for line in file:
                msg = '\r Reading line #' + str(i)
                print(msg, end="")
                i += 1
                word_vector_array = line.split(' ')
                word = str(word_vector_array[0])
                vector_str = word_vector_array[1:]
                vector_str = np.array(vector_str)
                # vector = [word] + list(map(float, vector_str))
                # vectors.append(vector)
                vector = vector_str.astype(np.float)
                vectors.append(vector)
                word_vector.append(word)
                # dataframe[word] = vector

        print("\r load file to dataframe", end="")
        dataframe = pd.DataFrame(vectors)
        dataframe['word'] = word_vector

        cols = dataframe.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        dataframe = dataframe[cols]

        print("\r  save table to DB", end="")

        # index = false is important for removing index.
        dataframe.to_sql(name=self._table_name, con=self._db.engine, index=False)

        # save model
        if not os.path.exists(self._word_vector_dict_full_path):
            os.makedirs(self._word_vector_dict_full_path)

    # def _fill_author_id_text_dictionary(self, targeted_fields_dict):
    #     author_id_texts_dict =
    #
    #     return author_id_texts_dict

    # added by Lior
