from DB.schema_definition import *
from commons import commons
from dataset_builder.feature_extractor.base_feature_generator import BaseFeatureGenerator


class Known_Words_Number_Feature_generator(BaseFeatureGenerator):
    def __init__(self, db, **kwargs):
        BaseFeatureGenerator.__init__(self, db, **kwargs)
        self._db = db

        self._word_lists_paths = self._config_parser.eval(self.__class__.__name__, "word_lists_path")
        self.word_lists_names = self._config_parser.eval(self.__class__.__name__, "word_lists_names")
        self._targeted_fields = self._config_parser.eval(self.__class__.__name__, "targeted_fields")
        self._author_id_texts_dict = {}
        self._words_dict = {}

    def execute(self):
        for targeted_fields_dict in self._targeted_fields:
            authors_features = []
            source_id_target_fields_dict = self._get_source_id_target_elements(targeted_fields_dict)
            source_ids = list(source_id_target_fields_dict.keys())
            source_id_source_element_dict = self._get_source_id_source_element_dict(source_ids, targeted_fields_dict)
            for source_targets_dict_item in source_id_target_fields_dict.items():
                author = source_id_source_element_dict[source_targets_dict_item[0]]
                kwargs = self._get_feature_kwargs(source_targets_dict_item, author, targeted_fields_dict)
                features = self.calc_avg_known_words(source_targets_dict_item[0], **kwargs)
                authors_features.extend(features)

            self.insert_author_features_to_db(authors_features)

    def calc_avg_known_words(self, source_id, **kwargs):
        destination_target_fields = kwargs['target']
        features = []
        logging.info("processing author " + source_id)
        for word_list_name in self.word_lists_names:
            self._load_known_words_to_dict(word_list_name)
            try:
                result = self._count_avg_known_words(destination_target_fields)
                window_start = self._window_start
                window_end = self._window_end
                attribute_name = str(self.__class__.__name__ + '_count_avg_known_word_from_' + word_list_name)
                author_feature = BaseFeatureGenerator.create_author_feature(attribute_name, source_id, result,
                                                                            window_start, window_end)
                features.append(author_feature)

            except Exception as e1:
                info_msg = e1.message
                logging.error(info_msg + word_list_name)
        return features

    def _load_known_words_to_dict(self, word_list_name):
        self._words_dict = {}
        full_path = self._word_lists_paths + "/" + word_list_name
        file_object = None
        if os.path.isfile(full_path):
            try:
                file_object = open(full_path, "r")
                for line in file_object:
                    clean_word = commons.clean_word(line).lower()
                    self._words_dict[clean_word] = 1
            except Exception as exc:
                logging.error("error in word list - " + word_list_name)
            finally:
                file_object.close()

    def _num_of_known_words_in_sentence(self, sentence):
        if sentence is None or sentence == '':
            return 0
        else:
            return sum(map(self._is_word_known, sentence.split(' ')))

    def _is_word_known(self, word):
        if word is None or word == '':
            return 0
        clean_word = commons.clean_word(word)
        if clean_word in self._words_dict:
            return 1
        return 0

    def _count_avg_known_words(self, sentences):
        if sentences is None or len(sentences) == 0:
            return -1
        counter = 0
        sum = 0
        for sentence in sentences:
            sum = sum + self._num_of_known_words_in_sentence(sentence)
            counter = counter + 1
        return float(sum) / counter
