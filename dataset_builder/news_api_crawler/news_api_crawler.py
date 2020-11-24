
from newsapi import NewsApiClient
from newsapi.newsapi_exception import NewsAPIException

from commons.commons import *
from DB.schema_definition import Claim, NewsArticle
from preprocessing_tools.abstract_controller import AbstractController


class NewsApiCrawler(AbstractController):
    def __init__(self, db):
        super(NewsApiCrawler, self).__init__(db)
        self._api_keys = self._config_parser.eval(self.__class__.__name__, "api_keys")
        self._current_key = 0
        self._news_api = NewsApiClient(api_key=self._api_keys[0])
        self._sources = self._config_parser.eval(self.__class__.__name__, "sources")
        self._limit = self._config_parser.eval(self.__class__.__name__, "limit")
        self._language = self._config_parser.eval(self.__class__.__name__, "language")
        self._num_of_requests = 0

    def execute(self, window_start=None):
        claims, news_articles = self.get_claims_and_articles(self._limit)
        self._save_to_db(claims, news_articles)

        pass

    def _save_to_db(self, claims, news_articles):
        self._db.add_claims_fast(claims)
        self._db.add_news_articles_fast(news_articles)

    def get_claims_and_articles(self, limit):
        total_claims, total_news_articles = [], []
        for i, source in enumerate(self._sources):
            # total_results = self._news_api.get_top_headlines(sources=source, language=self._language)[u'totalResults']
            total_results = 1000
            try:
                if self._num_of_requests == 1000:
                    self.swap_api_key()
                    self._num_of_requests = 0

                # if limit is None:
                #     limit = total_results
                max_articles = limit
                max_page = int(max_articles / 100) + 1
                if max_page == 1:
                    max_page += 1
                for page in range(1, max_page):
                    articles = self._news_api.get_everything(sources=source, page=page, page_size=100,
                                                                language=self._language)
                    article_count = len(articles['articles'])
                    if max_articles < article_count:
                        article_count = max_articles
                    print(
                        '\r{} {}/{} retrieved {}/{} articles'.format(source, str(i + 1), len(self._sources),
                                                                     article_count * page, min(article_count, limit)),
                        end='')
                    claims, news_articles = self.convert_articles_to_claims_and_news_articles(articles, source)
                    total_claims.extend(claims)
                    total_news_articles.extend(news_articles)
                    max_articles -= article_count
                    self._num_of_requests += 1
                print()
            except NewsAPIException as e:
                print()
                print(e.exception['message'])
                self.swap_api_key()
                self._num_of_requests = 0
            except Exception as e:
                print(e)
                self._save_to_db(total_claims, total_news_articles)
                total_claims, total_news_articles = [], []
                self.swap_api_key()
                self._num_of_requests = 0
        return total_claims, total_news_articles

    def swap_api_key(self):
        self._current_key += 1
        if self._current_key == len(self._api_keys):
            self._current_key = 0
        self._news_api = NewsApiClient(api_key=self._api_keys[self._current_key])

    def convert_articles_to_claims_and_news_articles(self, articles, source):
        claims = []
        news_articles = []
        for article in articles['articles']:
            claim = self.get_claim_from_article(article, source)
            news_article = self._get_news_article_from_article(article, source)

            claims.append(claim)
            news_articles.append(news_article)
        return claims, news_articles

    def _get_news_article_from_article(self, article, source):
        news_article = NewsArticle()
        news_article.url = article['url']
        news_article.title = article['title']
        news_article.description = article['description']
        news_article.author = article.get('author', '')
        try:
            publish_date = str_to_date(article['publishedAt'].split('+')[0], '%Y-%m-%dT%H:%M:%SZ')
        except:
            publish_date = str_to_date(article['publishedAt'].split('+')[0], '%Y-%m-%dT%H:%M:%S')
        news_article.published_date = publish_date
        news_article.content = article.get('content', '')
        news_article.url_to_image = article.get('urlToImage', '')
        news_article.domain = article['source'].get('id', str(source))
        news_article.article_id = compute_post_guid(self._social_network_url, news_article.url,
                                                    date_to_str(news_article.published_date))
        return news_article

    def get_claim_from_article(self, article, source):
        claim = Claim()
        claim.title = article['title']
        claim.description = article['description']
        claim.url = article['url']
        try:
            publish_date = str_to_date(article['publishedAt'].split('+')[0], '%Y-%m-%dT%H:%M:%SZ')
        except:
            publish_date = str_to_date(article['publishedAt'].split('+')[0], '%Y-%m-%dT%H:%M:%S')
        claim.verdict_date = publish_date
        claim.domain = article['source'].get('id', str(source))
        claim.verdict = True
        claim.claim_id = compute_post_guid(self._social_network_url, claim.url, date_to_str(claim.verdict_date))
        claim.category = str(source)
        return claim
