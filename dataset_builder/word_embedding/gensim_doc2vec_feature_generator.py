
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import os

from dataset_builder.word_embedding.Vectors_Operations import Vector_Operations
from dataset_builder.word_embedding.abstract_word_embadding_trainer import AbstractWordEmbaddingTrainer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

__author__ = "Aviad Elyashar"


class GensimDoc2VecFeatureGenerator(AbstractWordEmbaddingTrainer):
    def __init__(self, db):
        super(GensimDoc2VecFeatureGenerator, self).__init__(db, **{'authors': [], 'posts': {}})
        self._saved_models_path = self._config_parser.eval(self.__class__.__name__, "saved_models_path")
        self._max_vocabulary_size = self._config_parser.eval(self.__class__.__name__, "max_vocabulary_size")
        #self._number_of_dimensions_in_hidden_layer = self._config_parser.eval(self.__class__.__name__,
        #                                                                      "number_of_dimensions_in_hidden_layer")
        self._window_size_doc2vec = self._config_parser.eval(self.__class__.__name__, "window_size")
        self._max_objects_without_saving = self._config_parser.eval(self.__class__.__name__, "max_objects_without_saving")

        self.file_output_path = self._saved_models_path + self._table_name + ".csv"

        self._prefix = self.__class__.__name__ + "_"

        self.name = self.__class__.__name__

    def setUp(self):
        if not os.path.exists(self._saved_models_path):
            os.makedirs(self._saved_models_path)
        if os.path.exists(self.file_output_path):
            os.remove(self.file_output_path)

    def execute(self, window_start=None):
        for targeted_fields_dict in self._targeted_fields_for_embedding:
            source_id_target_elements_dict = self._get_source_id_target_elements(targeted_fields_dict)
            source_id_document_tuples = self._create_documents(source_id_target_elements_dict)

            msg = "\rStarting training doc2vec"
            print(msg, end='')
            model = self._train_doc2vec_model(source_id_document_tuples)

            msg = "\rFinishing training doc2vec"
            print(msg, end='')


            targeted_table = targeted_fields_dict['source']['table_name']
            targeted_field_name = targeted_fields_dict['destination']['table_name'] + "-" \
                                  + targeted_fields_dict['destination']['target_field']


            model_type = "{0}_dimensions_{1}_window_size".format(
                self._num_of_dimensions, self._window_size_doc2vec)

            source_ids = list(source_id_target_elements_dict.keys())
            counter = 0
            authors_features = []
            for i, source_id in enumerate(source_ids):
                msg = "\rExtracting doc2vec features: {0}/{1}:{2}".format(i, len(source_ids), source_id)
                print(msg, end="")
                counter += 1
                if counter % self._max_objects_without_saving == 0:
                    self._db.add_author_features(authors_features)
                    self._db.session.commit()

                source_id_vector = model[source_id]
                feature_name = model_type + '_' + targeted_table + "_" + targeted_field_name
                dimentions_feature_for_author = Vector_Operations.create_author_feature_for_each_dimention(
                    source_id_vector,
                    feature_name,
                    source_id,
                    self._window_start,
                    self._window_end,
                    self._prefix)
                authors_features = authors_features + dimentions_feature_for_author
            self._db.add_author_features(authors_features)
            self._db.session.commit()



            # Vector_Operations.create_features_from_word_embedding_dict(model,
            #                                                            targeted_table, targeted_field_name,
            #                                                            model_type,
            #                                                            self._window_start, self._window_end,
            #                                                            self._db, self._max_objects_without_saving,
            #                                                            self.__class__.__name__ + '_')


    def _train_doc2vec_model(self, source_id_document_tuples):
        documents = [TaggedDocument(document_content, [source_id]) for source_id, document_content in source_id_document_tuples]
        doc2vec_model = Doc2Vec(documents,
                        vector_size=self._num_of_dimensions,
                        window=self._window_size_doc2vec,
                        min_count=1,
                        max_vocab_size=self._max_vocabulary_size,
                        workers=4)

        doc2vec_model.save(self._saved_models_path + "doc2vec_model")
        return doc2vec_model



    def _create_documents(self, source_id_target_elements_dict):
        source_id_document_tuples =[]
        for source_id, target_elements in source_id_target_elements_dict.items():
            document_content = ""
            for target_element in target_elements:
                target_element_content = target_element.content
                document_content += target_element_content
                last_char = target_element_content[-1:]
                if last_char != ".":
                    document_content += ". "

            source_id_document_tuples.append((source_id, document_content))

        return source_id_document_tuples


