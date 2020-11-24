import unittest

from DB.schema_definition import DB
from twitter_crawler.generic_twitter_crawler import Generic_Twitter_Crawler
from twitter_crawler.twitter_crawler import Twitter_Crawler


class TwitterCrawlerTests(unittest.TestCase):
    def setUp(self):
        self._db = DB()
        self._db.setUp()
        self._twitter_crawler = Generic_Twitter_Crawler(self._db)

    def test_get_posts_by_terms(self):
        keyword = 'security'
        terms = [keyword]
        posts = self._twitter_crawler.get_posts_by_terms(terms)
        self.assertEqual(100, len(posts[keyword]))

    def tearDown(self):
        self._db.session.close()
        self._db.deleteDB()
        self._db.session.close()





if __name__ == '__main__':
    unittest.main()
