import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from dataset_builder.word_embedding.abstract_word_embadding_trainer import AbstractWordEmbaddingTrainer

__author__ = "Maor Reuben"


class GensimWordEmbeddingsModelTrainer(AbstractWordEmbaddingTrainer):
    def __init__(self, db):
        super(GensimWordEmbeddingsModelTrainer, self).__init__(db, **{'authors': [], 'posts': {}})
        self._saved_models_path = self._config_parser.eval(self.__class__.__name__, "saved_models_path")
        self._max_vocabulary_size = self._config_parser.eval(self.__class__.__name__, "max_vocabulary_size")
        self._number_of_dimensions_in_hidden_layer = self._config_parser.eval(self.__class__.__name__,
                                                                              "number_of_dimensions_in_hidden_layer")
        self._use_cbow = self._config_parser.eval(self.__class__.__name__, "use_cbow")
        self._window_size = self._config_parser.eval(self.__class__.__name__, "window_size")
        self._epochs = self._config_parser.eval(self.__class__.__name__, "epochs")
        self._selected_optimizer = self._config_parser.eval(self.__class__.__name__, "selected_optimizer")
        self._seed = self._config_parser.eval(self.__class__.__name__, "seed")
        self._loss_function = 0

        self.name = self.__class__.__name__

    def execute(self, window_start=None):
        word_embeddings = []
        dimensions = 0
        for targeted_fields_dict in self._targeted_fields_for_embedding:
            source_id_target_elements_dict = self._get_source_id_target_elements(targeted_fields_dict)
            sentences = self._get_target_field_from_elements(source_id_target_elements_dict, targeted_fields_dict)
            model = self._train_model_by_sentences(sentences)

            word_vector_dict = self._get_word_embedding_dict(model)
            dimensions = max(dimensions, len(list(word_vector_dict.values())[0]))
            self._write_word_vector_dict_to_csv(word_vector_dict)
            word_embeddings += self._calculate_word_embedding_to_authors(source_id_target_elements_dict,
                                                                         targeted_fields_dict, word_vector_dict)
        # self._add_word_embeddings_to_db(word_embeddings)

        self._write_word_embedding_to_csv(word_embeddings)

    def _train_model_by_sentences(self, sentences):
        model_seed = self._seed
        model_name_path = self._prepare_model_name_path()
        # Check if there is a pre-trained model
        # if self._config_parser.use_pretrained_model and os.path.isfile(model_name_path):
        #     self._result = Word2Vec.load(model_name_path)
        #     return
        # Train the model
        model = Word2Vec(size=self._number_of_dimensions_in_hidden_layer,
                         window=self._window_size,
                         min_count=3,
                         max_vocab_size=self._max_vocabulary_size,
                         sg=self._use_cbow,
                         seed=model_seed,
                         compute_loss=True,
                         iter=self._epochs)
        model.build_vocab(sentences)
        model.save(model_name_path)
        return model

    def get_loss_function(self):
        return self._loss_function

    def _prepare_model_name_path(self):
        # model_name = dataset-(number_of_graphs_to_compare)-(window_size)-(use_cbow)-epochs-(number_of_dimensions_in_hidden_layer)-
        # (threshold_for_creating_edge)-(distance_measure)-(percent_of_words_to_use)-(top_k_distances)-(edges_creation_method)-(selected_optimizer)-
        # (graph_number)

        model_name = self._saved_models_path + \
                     str(self._use_cbow) + "-" + \
                     str(self._epochs) + "-" + \
                     str(self._number_of_dimensions_in_hidden_layer) + "-" + \
                     self._selected_optimizer

        return model_name

    def _get_target_field_from_elements(self, source_id_target_elements_dict, targeted_fields_dict):
        if "destination" in targeted_fields_dict and "target_field" in targeted_fields_dict["destination"]:
            target_field = targeted_fields_dict["destination"]["target_field"]
            pass
        else:
            target_field = targeted_fields_dict["source"]["target_field"]
        target_fields = []
        for source_id, elements in source_id_target_elements_dict.items():
            # target_fields.extend([getattr(element, target_field) for element in elements])
            for element in elements:
                target_content = getattr(element, target_field)
                sentences = target_content.split('.')
                for sentence in sentences:
                    target_fields.append(sentence.strip().split(' '))
        return target_fields

    def _get_word_embedding_dict(self, model):
        word_embedding_dict = {}
        words = list(model.wv.vocab)
        # For server running
        # words = list(model.vocab)
        for word in words:
            word_embedding_dict[word] = model[word]

        return word_embedding_dict

    def _write_word_vector_dict_to_csv(self, word_vector_dict):
        dataframe = pd.DataFrame()
        for word in word_vector_dict:
            dataframe[word] = word_vector_dict[word]
        dataframe = dataframe.T
        dataframe.to_csv(self._saved_models_path + '/word_to_vector.csv')

    def _write_word_embedding_to_csv(self, word_embeddings):
        self._results_dataframe = pd.DataFrame()
        dimensions_count = 0
        for word_embedding in word_embeddings:
            self._fill_results_dataframe(*word_embedding)
            dimensions_count = len(word_embedding[2])

        column_names = ["author_id", "table_name", "id_field", "targeted_field_name", "word_embedding_type"]
        dimensions = np.arange(dimensions_count)
        column_names.extend(dimensions)
        self._results_dataframe.columns = column_names
        self._results_dataframe.to_csv(self._saved_models_path + self._table_name + ".csv")
        pass
