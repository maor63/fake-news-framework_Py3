#
# Created by Aviad on 04-Nov-16 10:22 PM.
#

import unittest
from DB.unit_tests.test_base import TestBase
from configuration.config_class import getConfig
from DB.schema_definition import DB
from preprocessing_tools.tumblr_importer.tumblr_importer import TumblrImporter
from commons.consts import Domains
from dataset_builder.feature_extractor.graph_feature_generator import GraphFeatureGenerator

class TestTumblrImporterGraphFeatureGenerator(TestBase):

    def setUp(self):
        TestBase.setUp(self)
        self.config = getConfig()
        self._start_date = self.config.eval("DEFAULT", "start_date")
        #self._end_date = self.config.get("DEFAULT", "end_date")

        self._tsv_files_path = self.config.get("TumblrImporter", "tsv_test_files_graph_feature_generator")

        self._db = DB()
        self._db.setUp()

        self._tumblr_parser = TumblrImporter(self._db)
        self._tumblr_parser.setUp(self._tsv_files_path)
        self._tumblr_parser.execute()

        self._author_guid = "f0f4bb42-3fed-322a-b71a-681179d47ea1"

        authors = self._db.get_authors_by_domain(Domains.MICROBLOG)
        posts = self._db.get_posts_by_domain(Domains.MICROBLOG)
        parameters = {"authors": authors, "posts": posts}

        graph_types = self.config.eval("GraphFeatureGenerator_1", "graph_types")
        algorithms = self.config.eval("GraphFeatureGenerator_1", "algorithms")
        aggregations = self.config.eval("GraphFeatureGenerator_1", "aggregation_functions")
        neighborhood_sizes = self.config.eval("GraphFeatureGenerator_1", "neighborhood_sizes")
        distances_from_labeled_authors = self.config.eval("GraphFeatureGenerator_1", "distances_from_labeled_authors")
        graph_directed = self.config.eval("GraphFeatureGenerator_1", "graph_directed")
        graph_weights = self.config.eval("GraphFeatureGenerator_1", "graph_weights")

        parameters.update({"graph_types": graph_types, "algorithms": algorithms,
                           "aggregation_functions": aggregations, "neighborhood_sizes": neighborhood_sizes,
                           "graph_directed": graph_directed, "graph_weights": graph_weights,
                           "distances_from_labeled_authors": distances_from_labeled_authors})



        graph_feature_generator = GraphFeatureGenerator(self._db, **parameters)
        graph_feature_generator.execute()

        self._author_features = self._db.get_author_features_by_author_guid(author_guid=self._author_guid)
        self._author_features_dict = self._create_author_features_dictionary(self._author_features)


    def test_account_age(self):

        attribute_value = self._author_features_dict["account_age"]
        attribute_value = int(attribute_value)
        self.assertIsNotNone(attribute_value)


    # def test_number_followers(self):
    #
    #     attribute_value = self._author_features_dict["number_followers"]
    #     attribute_value = float(attribute_value)
    #     self.assertEquals(0.00015625, attribute_value)

    def test_number_of_crawled_posts(self):

        attribute_value = self._author_features_dict["number_of_crawled_posts"]
        attribute_value = int(attribute_value)
        self.assertEqual(2, attribute_value)


    def test_screen_name_length(self):

        attribute_value = self._author_features_dict["screen_name_length"]
        attribute_value = int(attribute_value)
        self.assertEqual(13, attribute_value)


    def test_number_followers(self):

        attribute_value = self._author_features_dict["number_followers"]
        attribute_value = int(attribute_value)
        self.assertEqual(13, attribute_value)

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
