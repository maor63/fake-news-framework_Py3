# coding=utf-8

import unittest
from DB.unit_tests.test_base import TestBase
from configuration.config_class import getConfig
from commons.commons import *
from twitter_rest_api.twitter_rest_api import Twitter_Rest_Api
from preprocessing_tools.create_authors_table import CreateAuthorTables
from preprocessing_tools.xml_importer import XMLImporter
from commons.consts import Author_Type, DB_Insertion_Type
from bad_actors_collector.bad_actors_collector import BadActorsCollector


class TestXmlImporter(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        self.config = getConfig()
        from DB.schema_definition import DB
        self.db = DB()
        self.db.setUp()
        self.social_network_crawler = Twitter_Rest_Api(self.db)
        self.xml_importer = XMLImporter(self.db)
        self.create_author_table = CreateAuthorTables(self.db)
        self._targeted_twitter_author_ids = self.config.eval('BadActorsCollector', "targeted_twitter_author_ids")

        self._targeted_twitter_post_ids = self.config.eval('BadActorsCollector', "targeted_twitter_post_ids")
        self._bad_actor_collector = BadActorsCollector(self.db)

    def test_umlaut_on_xml_files(self):
        '''
        We created xmls with umlaut. We should parse it to the DB.
        After that we should extract it from DB and assert that the umlaut encoded properly.
        '''

        self.xml_importer.setUp()
        self.xml_importer.execute()

        post_guid = "1fc90ec7e0e430839fb302f682f92cd8"
        post = self.xml_importer._db.get_post_by_id(post_guid)
        post_content = "kømr ljúga róa úll (foul) / fýla sǫkk (sank) / søkkva☺☻♥♦♣♠•◘○"
        post_content = post_content.decode('utf_8')
        post_title = "kømr ljúga róa úll (foul) / fýla sǫkk (sank) / søkkva øýúœóæá"
        post_title = post_title.decode('utf_8')

        self.assertEqual(post.content, post_content)
        self.assertEqual(post.title, post_title)
        self.db.session.close()

    def test_XML_importer_not_overwriting_bad_actor_collector(self):
        self._bad_actor_collector.crawl_bad_actors_followers()
        self._bad_actor_collector.crawl_bad_actors_retweeters()
        self.xml_importer.setUp()
        self.xml_importer.execute(getConfig().eval("DEFAULT", "start_date"))
        self.create_author_table.setUp()
        self.create_author_table.execute(getConfig().eval("DEFAULT", "start_date"))
        res = self.db.get_author_by_author_guid(compute_author_guid_by_author_name('adoula01'))
        author = res
        self.assertTrue(
            author.xml_importer_insertion_date != None and author.bad_actors_collector_insertion_date != None)
        self.db.session.close()

    def tearDown(self):
        self.db.session.close()
        self.db.deleteDB()
        pass
