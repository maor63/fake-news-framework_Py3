

from dataset_builder.feature_extractor.base_feature_generator import BaseFeatureGenerator
from preprocessing_tools.abstract_controller import AbstractController
import json
import pandas as pd
import os

__author__ = "Aviad Elyashar"

class ClaimFeatureGenerator(AbstractController):
    def __init__(self, db, **kwargs):
        AbstractController.__init__(self, db)
        self._features_list = self._config_parser.eval(self.__class__.__name__, "features_list")
        self._good_claim_type = self._config_parser.eval(self.__class__.__name__, "good_claim_type")
        self._bad_claim_type = self._config_parser.eval(self.__class__.__name__, "bad_claim_type")
        self._prefix = self.__class__.__name__

    def execute(self, window_start=None):

        claims = self._db.get_claims()
        author_features = []
        for feature_name in self._features_list:
            for claim in claims:
                claim_id = claim.claim_id
                attribute_value = getattr(self, feature_name)(claim)
                if attribute_value:
                    attribute_name = "{0}_{1}".format(self._prefix, feature_name)
                    author_feature = BaseFeatureGenerator.create_author_feature(attribute_name, claim_id, attribute_value,
                                                                            self._window_start, self._window_end)
                    author_features.append(author_feature)

        self._db.add_author_features(author_features)

    def claim_type(self, claim):
        if claim.verdict in self._good_claim_type:
            return 'True'
        elif claim.verdict in self._bad_claim_type:
            return 'False'
        return None

    def claim_id(self, claim):
        claim_id = claim.claim_id
        return claim_id

    def verdict_date(self, claim):
        verdict_date = claim.verdict_date
        return verdict_date