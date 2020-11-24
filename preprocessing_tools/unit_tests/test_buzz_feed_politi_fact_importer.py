# coding=utf-8
from unittest import TestCase

from DB.schema_definition import *
from preprocessing_tools.buzz_feed_politi_fact_importer.buzz_feed_politi_fact_importer import BuzzFeedPolitiFactImporter


class TestBuzzFeedPolitiFactImporter(TestCase):
    def setUp(self):
        self._db = DB()
        self._db.setUp()
        self.buzz_feed_importer = BuzzFeedPolitiFactImporter(self._db)

    def test_parse_json(self):
        json = self.buzz_feed_importer._parse_json("data/input/BuzzFeedForTests/FakeNewsContent/BuzzFeed_Fake_1-Webpage.json")
        self.assertTrue('url' in list(json.keys()) and 'text' in json)

    def test_get_authors_from_jsons(self):
        json = self.buzz_feed_importer._parse_json("data/input/BuzzFeedForTests/FakeNewsContent/BuzzFeed_Fake_10-Webpage.json")
        author = self.buzz_feed_importer.extract_author(json)
        assert (isinstance(author, Author))
        self.assertEqual(author.name, 'http://eaglerising.com')
        self.assertEqual(author.domain, 'BuzzFeed')
        self.assertTrue(author.author_guid is not None)

    def test_get_post_from_json(self):
        json = self.buzz_feed_importer._parse_json("data/input/BuzzFeedForTests/FakeNewsContent/BuzzFeed_Fake_10-Webpage.json")
        post = self.buzz_feed_importer.extract_post(json, 'Fake')
        assert (isinstance(post, Post))
        self.assertEqual(post.title, "Charity: Clinton Foundation Distributed “Watered-Down” AIDS Drugs to Sub-Saharan Africa – Eagle Rising")
        self.assertEqual(post.tags, '')
        self.assertEqual(post.url, "http://eaglerising.com/36899/charity-clinton-foundation-distributed-watered-down-aids-drugs-to-sub-saharan-africa/")
        self.assertEqual(post.description, "The possibility that CHAI distributed adulterated and diluted AIDS drugs to Third World victims could shake the foundations of the Clinton charity.")

    def test_excute(self):
        self.buzz_feed_importer.execute()
