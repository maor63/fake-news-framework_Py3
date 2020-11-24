

import googleapiclient

from DB.schema_definition import Post, Claim_Tweet_Connection, GooglePostKeywords
from commons.commons import compute_author_guid_by_author_name
from commons.method_executor import Method_Executor
from googleapiclient.discovery import build
import time
from datetime import datetime

__author__ = "Joshua Grogin"


def _get_claim_tweet_connection(claim_id, post_id):
    claim_tweet_connection = Claim_Tweet_Connection()
    claim_tweet_connection.claim_id = claim_id
    claim_tweet_connection.post_id = post_id
    return claim_tweet_connection


def _get_google_post_keywords_connection(post_id, keywords):
    connection = GooglePostKeywords()
    connection.post_id = post_id
    connection.keywords = keywords
    connection.insertion_date = '{}'.format(datetime.utcnow().strftime("%Y-%m-%d"))
    return connection


def _extract_result(result, keywords, domain="google_search"):
    post = Post()
    post.title = result['title']
    post.url = result['link']
    post.domain = domain
    post.content = result['snippet']
    post.post_type = result['kind']
    if 'cacheId' not in result:
        post.guid = compute_author_guid_by_author_name('{}_{}'.format(post.url, keywords))
    else:
        post.guid = compute_author_guid_by_author_name('{}_{}_{}'.format(post.url, keywords, result['cacheId']))
    post.post_id = post.guid
    return post


class GoogleLinksByKeywords(Method_Executor):
    def __init__(self, db):
        super(GoogleLinksByKeywords, self).__init__(db)
        self._submission_limit = self._config_parser.eval(self.__class__.__name__, "submission_limit")
        self._my_api_keys = self._submission_limit = self._config_parser.eval(self.__class__.__name__, "api_keys")
        self._my_cse_id = "002858961093097572205:mf4p_dacyzo"
        self._max_results = self._submission_limit = self._config_parser.eval(self.__class__.__name__,
                                                                              "max_results")
        self._websites = self._config_parser.eval(self.__class__.__name__, "websites")
        self._results_posts = []
        self._claim_tweet_connections = []
        self._google_post_keywords_connections = []

    def google_search(self, search_term, api_key, **kwargs):
        service = build("customsearch", "v1", developerKey=api_key, cache_discovery=False)
        res = service.cse().list(q=search_term, cx=self._my_cse_id, **kwargs).execute()
        if 'items' in res:
            return res['items']
        else:
            return []

    def import_links_from_keywords(self):
        claims = self._db.get_claims()
        for i, claim in enumerate(claims):
            print("\rGoogleLinksByKeywords finished {0}/{1}".format(i, len(claims)), end='')
            for keyword in claim.keywords.split("|| "):
                for site in self._websites:
                    self._get_links_from_keywords('{} site:{}'.format(keyword, site), claim)
        self._save_db()

    def _save_db(self):
        self._db.addPosts(self._results_posts)
        self._db.addPosts(self._google_post_keywords_connections)
        self._db.add_claim_connections(self._claim_tweet_connections)
        self._results_posts = []
        self._claim_tweet_connections = []
        self._google_post_keywords_connections = []

    def _get_links_from_keywords(self, keywords, claim):
        while True:
            for api_key in self._my_api_keys:
                try:
                    results = self.google_search(keywords, api_key, num=self._max_results)
                    for result in results:
                        post = _extract_result(result, keywords)
                        self._results_posts.append(post)
                        self._claim_tweet_connections.append(_get_claim_tweet_connection(claim.claim_id, post.post_id))
                        self._google_post_keywords_connections.append(_get_google_post_keywords_connection(post.post_id,
                                                                                                           keywords))
                        if len(self._results_posts) >= self._submission_limit:
                            self._save_db()
                    return
                except googleapiclient.errors.HttpError as e:
                    print(e.content)
                    time.sleep(10800)
