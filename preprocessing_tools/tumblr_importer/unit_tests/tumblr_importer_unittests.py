#
# Created by Aviad on 04-Nov-16 10:22 PM.
#

import unittest
from DB.unit_tests.test_base import TestBase
from configuration.config_class import getConfig
from DB.schema_definition import DB
from preprocessing_tools.tumblr_importer.tumblr_importer import TumblrImporter
from commons.consts import Connection_Type
from dataset_builder.feature_extractor.feature_extractor import FeatureExtractor

class TestTumblrImporter(unittest.TestCase):
    def setUp(self):

        # TestBase.setUp(self)
        self.config = getConfig()

        self._db = DB()
        self._db.setUp()
        self._tumblr_importer = TumblrImporter(self._db)
        self._tumblr_importer.execute()

    def test_parse_authors_tsv_file(self):
        authors = self._db.get_authors()
        num_of_authors = len(authors)
        self.assertEqual(3, num_of_authors)

    def test_parse_posts_tsv_file(self):
        posts = self._db.get_posts()
        num_of_posts = len(posts)
        self.assertEqual(6, num_of_posts)

    def test_parse_author_connections_tsv_file(self):
        cursor = self._db.get_author_connections_by_type(Connection_Type.FOLLOWER)
        author_connections_iterator = self._db.result_iter(cursor)
        num_of_author_connections = sum(1 for i in author_connections_iterator)
        self.assertEqual(5, num_of_author_connections)

    def tearDown(self):
        self._db.session.close()
        self._db.deleteDB()