from unittest import TestCase

from DB.schema_definition import DB
from dataset_builder.news_api_crawler.news_api_crawler import NewsApiCrawler


class TestNewsApiCrawler(TestCase):
    def setUp(self):
        self._db = DB()
        self._db.setUp()
        self._news_api = NewsApiCrawler(self._db)
        self._news_api.setUp()

    def test_get_claims_and_articles_less_then_100(self):
        source = 'cnn'
        self._news_api._source = source
        limit_results = 78
        claims, articles = self._news_api.get_claims_and_articles(limit_results)
        self.assertEqual(limit_results, len(claims))
        self.assertEqual(limit_results, len(articles))
        self.assertClaimsDomain(claims, source)

        source = 'bbc-news'
        self._news_api._source = source
        claims, articles = self._news_api.get_claims_and_articles(limit_results)
        self.assertEqual(limit_results, len(claims))
        self.assertEqual(limit_results, len(articles))
        self.assertClaimsDomain(claims, source)

    def test_get_claims_and_articles_equal_100(self):
        source = 'cnn'
        self._news_api._source = source
        limit_results = 78
        claims, articles = self._news_api.get_claims_and_articles(limit_results)
        self.assertEqual(limit_results, len(claims))
        self.assertEqual(limit_results, len(articles))
        self.assertClaimsDomain(claims, source)

        source = 'bbc-news'
        self._news_api._source = source
        claims, articles = self._news_api.get_claims_and_articles(limit_results)
        self.assertEqual(limit_results, len(claims))
        self.assertEqual(limit_results, len(articles))
        self.assertClaimsDomain(claims, source)

    def test_get_claims_and_articles_greater_100(self):
        source = 'cnn'
        self._news_api._source = source
        limit_results = 148
        claims, articles = self._news_api.get_claims_and_articles(limit_results)
        self.assertEqual(limit_results, len(claims))
        self.assertEqual(limit_results, len(articles))
        self.assertClaimsDomain(claims, source)

        source = 'bbc-news'
        self._news_api._source = source
        claims, articles = self._news_api.get_claims_and_articles(limit_results)
        self.assertEqual(limit_results, len(claims))
        self.assertEqual(limit_results, len(articles))
        self.assertClaimsDomain(claims, source)

    def assertClaimsDomain(self, claims, source):
        for claim in claims:
            self.assertEqual(claim.domain, source)
