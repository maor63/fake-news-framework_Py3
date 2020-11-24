from unittest import TestCase

from DB.schema_definition import DB
from preprocessing_tools.politi_fact_posts_crawler.politi_fact_posts_crawler import PolitiFactPostsCrawler


class TestPolitiFactPostsCrawler(TestCase):
    def setUp(self):
        self.db = DB()
        self.db.setUp()
        self.posts_crawler = PolitiFactPostsCrawler(self.db)

    def test_execute(self):
        self.posts_crawler.execute(None)
        posts = self.posts_crawler._listdic
        expected_url = "http://www.politifact.com/truth-o-meter/statements/2018/apr/06/blog-posting/fake-news-says-cnn-pulled-plug-during-youtube-shoo/"
        found = False
        for post in posts:
            if post["url"] == expected_url:
                self.assertEqual(post["date"], "2018-04-04 00:00:00")
                self.assertEqual(post["title"], "Fake news says CNN \"pulled the plug\" during YouTube shooting coverage")
                self.assertEqual(post["created_at"], "2018-04-04 00:00:00")
                self.assertEqual(post["content"], "YouTube witness makes shocking admission on live TV CNN pulls plug immediately")
                self.assertEqual(post["domain"], "PolitiFact")
                self.assertTrue(post["guid"] is not None)
                self.assertTrue(post["post_guid"] is not None)
                self.assertTrue(post["post_id"] is not None)
                self.assertTrue(post["author_guid"] is not None)
                self.assertEqual(post["author"], expected_url)
                self.assertTrue(post["author_osn_id"] is not None)
                self.assertEqual(post["references"], "")
                self.assertEqual(post["post_type"], "false")
                found = True
        if not found:
            self.fail()


    def test__build_qury_for_subject(self):
        query = self.posts_crawler._build_qury_for_subject("apple")
        self.assertEqual(query, "http://www.politifact.com/api/statements/truth-o-meter/subjects/apple/json/?n=10")

