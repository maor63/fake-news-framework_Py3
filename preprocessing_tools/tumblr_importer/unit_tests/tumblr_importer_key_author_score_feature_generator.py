#
# Created by Aviad on 04-Nov-16 10:22 PM.
#

import unittest
from DB.unit_tests.test_base import TestBase
from configuration.config_class import getConfig
from DB.schema_definition import DB
from preprocessing_tools.tumblr_importer.tumblr_importer import TumblrImporter
from dataset_builder.feature_extractor.behavior_feature_generator import BehaviorFeatureGenerator
from commons.consts import Domains
from dataset_builder.key_authors_model import KeyAuthorsModel
from dataset_builder.feature_extractor.key_author_score_feature_generator import KeyAuthorScoreFeatureGenerator
from dataset_builder.autotopic_executor import AutotopicExecutor

class TestTumblrImpoterKeyAuthorScoreFeatureGenerator(TestBase):

    def setUp(self):
        TestBase.setUp(self)
        self.config = getConfig()
        self._tsv_files_path = self.config .get("TumblrImporter", "tsv_test_files_key_author_score_feature_generator")
        self._db = DB()
        self._db.setUp()
        self._tumblr_parser = TumblrImporter(self._db)

        self._author_guid = "150ff707-a6eb-3051-8f3c-f623293c714b"

        self._tumblr_parser.setUp(self._tsv_files_path)
        self._tumblr_parser.execute()

        autotopic_executor = AutotopicExecutor(self._db)
        autotopic_executor.setUp()
        autotopic_executor.execute()


        key_author_model = KeyAuthorsModel(self._db)
        key_author_model.setUp()
        key_author_model.execute()

        authors = self._db.get_authors_by_domain(Domains.MICROBLOG)
        posts = self._db.get_posts_by_domain(Domains.MICROBLOG)
        parameters = {"authors": authors, "posts": posts}

        self._key_author_score_feature_generator = KeyAuthorScoreFeatureGenerator(self._db, **parameters)
        self._key_author_score_feature_generator.execute()

        self._author_features = self._db.get_author_features_by_author_guid(author_guid=self._author_guid)
        self._author_features_dict = self._create_author_features_dictionary(self._author_features)

    def test_sum_tfidf(self):

        attribute_value = self._author_features_dict["sum_tfidf"]
        attribute_value = float(attribute_value)
        self.assertEqual(120.0, attribute_value)


    def test_max_tfidf(self):

        attribute_value = self._author_features_dict["max_tfidf"]
        attribute_value = float(attribute_value)
        self.assertEqual(2.0, attribute_value)

    def tearDown(self):
        self._db.deleteDB()
        pass

    def _create_author_features_dictionary(self, author_features):
        author_features_dict = {}
        for author_feature in author_features:
            attribute_name = author_feature.attribute_name
            attribute_value = author_feature.attribute_value
            author_features_dict[attribute_name] = attribute_value

        return author_features_dict