import timeit

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from .base_feature_generator import BaseFeatureGenerator
import numpy as np
from scipy.stats import skew, kurtosis
import nltk


class Sentiment_Feature_Generator(BaseFeatureGenerator):
    def __init__(self, db, **kwargs):
        BaseFeatureGenerator.__init__(self, db, **kwargs)
        self._features = self._config_parser.eval(self.__class__.__name__, "feature_list")
        self._targeted_fields = self._config_parser.eval(self.__class__.__name__, "targeted_fields")
        self._aggregated_functions = self._config_parser.eval(self.__class__.__name__, "aggregated_functions")
        nltk.download('vader_lexicon')
        self._sentence_analyser = SentimentIntensityAnalyzer()
        self._clear_memory()
        # self._target_fields = []
        self._features = self._config_parser.eval(self.__class__.__name__, "feature_list")
        self._features_names_count = len(self._features) * len(self._aggregated_functions)

    def _clear_memory(self):
        self._post_sentiment_dict = {}

    def execute(self, window_start=None):
        aggregated_functions = list(map(eval, self._aggregated_functions))
        total_authors_features = []
        start = timeit.default_timer()
        for target_fields in self._targeted_fields:
            for source_id_element_dict, source_targets_dict in self._load_data_using_arg(target_fields):
                suffix = ''
                authors_features = self._get_features(source_id_element_dict, source_targets_dict,
                                                      suffix, target_fields, aggregated_functions)
                total_authors_features.extend(self._add_suffix_to_author_features(authors_features, suffix))
                if len(total_authors_features) > self._max_objects_save:
                    self._db.add_author_features_fast(total_authors_features)
                    total_authors_features = []
                self._clear_memory()
            self._db.add_author_features_fast(total_authors_features)
            total_authors_features = []
        print("sentiment feature generator execution time {} sec".format(str(timeit.default_timer() - start)))


    def authors_posts_semantic_compound(self, **kwargs):
        if 'posts' in list(kwargs.keys()):
            return self._get_posts_semantic_scores_base(kwargs['posts'], 'compound')

    def authors_posts_semantic_positive(self, **kwargs):
        if 'posts' in list(kwargs.keys()):
            return self._get_posts_semantic_scores_base(kwargs['posts'], 'pos')

    def authors_posts_semantic_negative(self, **kwargs):
        if 'posts' in list(kwargs.keys()):
            return self._get_posts_semantic_scores_base(kwargs['posts'], 'neg')

    def authors_posts_semantic_neutral(self, **kwargs):
        if 'posts' in list(kwargs.keys()):
            return self._get_posts_semantic_scores_base(kwargs['posts'], 'neu')

    def _get_posts_semantic_scores_base(self, posts, scores_param):
        return [self._get_sentence_scores(post.content, scores_param, post.post_id) for post in posts]

    def cleanUp(self, **kwargs):
        pass

    def _get_sentence_scores(self, sentence, param, post_id):  # param = 'neg','neu','pos','compound'
        if post_id not in self._post_sentiment_dict:
            if sentence is None:
                return 0.0
            else:
                self._post_sentiment_dict[post_id] = self._sentence_analyser.polarity_scores(sentence)

        score = self._post_sentiment_dict[post_id]
        return score[param]
