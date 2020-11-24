import logging
from abc import ABCMeta, abstractmethod

from dataset_builder.feature_extractor.base_feature_generator import BaseFeatureGenerator
from dataset_builder.word_embedding.Vectors_Operations import Vector_Operations


class WordEmbeddingsFeatureGenerator(BaseFeatureGenerator, metaclass=ABCMeta):
    def __init__(self, db, **kwargs):
        BaseFeatureGenerator.__init__(self, db, **kwargs)
        self._targeted_author_word_embeddings = self._config_parser.eval(self.__class__.__name__,
                                                                         "targeted_author_word_embeddings")
        self._max_objects_without_saving = self._config_parser.eval(self.__class__.__name__,
                                                                    "max_objects_without_saving")
        self._word_embedding_table_name = self._config_parser.eval(self.__class__.__name__,
                                                                   "word_embedding_table_name")


    @abstractmethod
    def load_author_guid_word_embedding_dict(self, targeted_field_name, targeted_table,
                                             targeted_word_embedding_type):
        pass

    def execute(self):
        logging.info("started extracting word_embeddings feature generator:")
        for counter, target_author_word_embeddings_dict in enumerate(self._targeted_author_word_embeddings):
            targeted_table = target_author_word_embeddings_dict["table_name"]
            targeted_field_name = target_author_word_embeddings_dict["targeted_field_name"]
            targeted_word_embedding_type = target_author_word_embeddings_dict["word_embedding_type"]

            targeted_word_embeddings_combination = targeted_table + "_" + targeted_field_name + "_" + targeted_word_embedding_type
            logging.info("currently extracting features of " + targeted_word_embeddings_combination + ": " + str(
                counter + 1) + " out of " + str(
                len(self._targeted_author_word_embeddings)))

            author_guid_word_embeding_dict = self.load_author_guid_word_embedding_dict(targeted_field_name,
                                                                                       targeted_table,
                                                                                       targeted_word_embedding_type)
            Vector_Operations.create_features_from_word_embedding_dict(author_guid_word_embeding_dict,
                                                                       targeted_table, targeted_field_name,
                                                                       targeted_word_embedding_type,
                                                                       self._word_embedding_table_name,
                                                                       self._window_start, self._window_end,
                                                                       self._db, self._max_objects_without_saving,
                                                                       self.__class__.__name__ + '_')