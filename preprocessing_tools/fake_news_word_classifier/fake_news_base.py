

import os
import sys
from collections import defaultdict, Counter

from commons.method_executor import Method_Executor
from dataset_builder.feature_extractor.base_feature_generator import BaseFeatureGenerator


class FakeNewsBase(BaseFeatureGenerator):
    def __init__(self, db, **kwargs):
        super(FakeNewsBase, self).__init__(db, **{'authors': {}, 'posts': []})
        self._targeted_fields = self._config_parser.eval(self.__class__.__name__, "targeted_fields")
        self._feature_list = self._config_parser.eval(self.__class__.__name__, "feature_list")
        self._fake_news_words_path = self._config_parser.eval(self.__class__.__name__, "fake_news_words_path")
        self._bad = self._config_parser.eval(self.__class__.__name__, "bad")
        self._good = self._config_parser.eval(self.__class__.__name__, "good")
        self._prefix = self.__class__.__name__
        self._index = 1
        self._fake_news_dictionary = self._load_dictionary()



    def _load_dictionary(self):
        spaces = "\r\n\t"
        if os.path.exists(self._fake_news_words_path):
            with open(self._fake_news_words_path) as file:
                return set((line.strip(spaces) for line in file.readlines()))
        return set()

    def execute(self, window_start=None):
        for action_name in self._feature_list:
            try:
                getattr(self, action_name)()
            except AttributeError as e:
                print('\nError: {0}\n'.format(e.message), file=sys.stderr)

    def _calc_source_id_word_count_dict(self, kwargs):
        all_text = ' '.join(kwargs['target']).lower()
        _claim_id_word_count_dict = Counter(
            {x: all_text.count(x) for x in self._fake_news_dictionary})
        return _claim_id_word_count_dict

    def _calc_source_id_word_fraction_dict(self, kwargs):
        all_text = ' '.join(kwargs['target']).lower()
        targets_count = len(kwargs['target'])
        _claim_id_word_frec_dict = Counter(
            {x: all_text.count(x) / float(targets_count) for x in self._fake_news_dictionary})
        return _claim_id_word_frec_dict
