import pandas as pd
from .base_feature_generator import BaseFeatureGenerator
from commons.commons import *

__author__ = "Joshua Grogin"


class RedditByClaimFeatureGenerator(BaseFeatureGenerator):
    def __init__(self, db, **kwargs):
        BaseFeatureGenerator.__init__(self, db, **kwargs)
        self._measure_names = self._config_parser.eval(self.__class__.__name__, "measure_names")
        self._aggregation_functions = self._config_parser.eval(self.__class__.__name__, "aggregation_functions")

    def execute(self, window_start=None):
        claims_reddit_post_connection = self._get_reddit_posts_claims_connections()
        for measure_name in self._measure_names:
            for source_id, target_elements in claims_reddit_post_connection.items():
                features = getattr(self, measure_name)(source_id, target_elements)
                self._db.add_author_features(features)

    def apply_aggregation_functions(self, target_elements, source_id, field, name):
        field_series = [getattr(target_element, field, None) for target_element in target_elements
                        if getattr(target_element, field, None) is not None]
        authors_features = []
        pd_series = pd.Series(field_series)
        for aggregation_functions_name in self._aggregation_functions:
            aggregated__score = getattr(pd_series, aggregation_functions_name)()
            author_feature = self._create_author_feature(aggregation_functions_name, name, source_id, aggregated__score)
            authors_features.append(author_feature)
        return authors_features

    def _create_author_feature(self, aggregation_functions_name, name, source_id, aggregated__score):
        return self.create_author_feature(
            "{}_{}_{}".format(self.__class__.__name__, aggregation_functions_name, name), source_id,
            '{}'.format(aggregated__score), self._window_start, self._window_end)

    def _get_reddit_posts_claims_connections(self):
        pass

    def _func_name(self):
        return sys._getframe(1).f_code.co_name
