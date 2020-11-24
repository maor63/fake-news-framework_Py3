from collections import defaultdict
from unittest import TestCase

from DB.schema_definition import DB, Post, Author, Claim
from commons.commons import str_to_date, convert_str_to_unicode_datetime, date_to_str
from .old_tweets_crawler import OldTweetsCrawler


class TestOldTweetsCrawler(TestCase):
    # I checked the test at 21/08/2018 there is a chance that the return tweet count will change (I hope not)
    def setUp(self):
        self._db = DB()
        self._db.setUp()
        self.tweets_crawler = OldTweetsCrawler(self._db)
        self.tweets_crawler._domain = 'Claim'
        self._add_author("author_guid")
        self._claims = {}

    def tearDown(self):
        self._db.session.close()

    def test_retrieve_tweets_by_content_between_dates_after(self):
        self._add_claim("post0", "The Rock Running for President", "2017-02-03 00:00:00")
        self._db.commit()
        date_interval_dict = defaultdict(set)
        claim_date = self._claims["post0"].verdict_date
        until_date = str_to_date("2017-08-03 00:00:00")
        self.tweets_crawler._limit_start_date = True
        self.tweets_crawler._limit_end_date = True
        tweets = self.tweets_crawler._retrieve_tweets_between_dates(self._claims["post0"],
                                                                    "The Rock Running for President",
                                                                    date_to_str(claim_date, "%Y-%m-%d"),
                                                                    date_to_str(until_date, "%Y-%m-%d"))
        tweets_date = [tweet.date for tweet in tweets]
        self.assertTrue(all([claim_date <= date < until_date for date in tweets_date]))
        self.assertEqual(100, len(tweets))

    def test_retrieve_tweets_by_content_between_dates_before(self):
        self._add_claim("post0", "The Rock Running for President", "2017-02-03 00:00:00")
        self._db.commit()
        date_interval_dict = defaultdict(set)
        claim_date = self._claims["post0"].verdict_date
        since_date = str_to_date("2016-08-03 00:00:00")
        self.tweets_crawler._limit_start_date = True
        self.tweets_crawler._limit_end_date = True
        tweets = self.tweets_crawler._retrieve_tweets_between_dates(self._claims["post0"],
                                                                    "The Rock Running for President",
                                                                    date_to_str(since_date, "%Y-%m-%d"),
                                                                    date_to_str(claim_date, "%Y-%m-%d"))
        tweets_date = [tweet.date for tweet in tweets]
        self.assertTrue(all([since_date <= date < claim_date for date in tweets_date]))
        self.assertEqual(100, len(tweets))

    def test_retrieve_tweets_by_content_between_dates_1_month_interval(self):
        self._add_claim("post0", "The Rock Running for President", "2017-02-03 00:00:00")
        self._db.commit()
        since_date = str_to_date("2017-01-03 00:00:00")
        until_date = str_to_date("2017-03-03 00:00:00")
        self.tweets_crawler._limit_start_date = True
        self.tweets_crawler._limit_end_date = True
        self.tweets_crawler._max_num_tweets = 133
        self.tweets_crawler._month_interval = 1
        tweets = self.tweets_crawler._retrieve_old_tweets(self._claims["post0"], "The Rock Running for President")

        tweets_date = [tweet.date for tweet in tweets]
        self.assertTrue(all([since_date <= date < until_date for date in tweets_date]))
        self.assertEqual(133, len(tweets))

    def test_retrieve_tweets_by_content_between_dates_no_limit_after(self):
        self._add_claim("post0", "The Rock Running for President", "2017-02-03 00:00:00")
        self._db.commit()
        since_date = str_to_date("2017-01-03 00:00:00")
        until_date = str_to_date("2017-03-03 00:00:00")
        self.tweets_crawler._limit_start_date = True
        self.tweets_crawler._limit_end_date = False
        self.tweets_crawler._max_num_tweets = 250
        self.tweets_crawler._month_interval = 1
        tweets = self.tweets_crawler._retrieve_old_tweets(self._claims["post0"], "The Rock Running for President")

        tweets_date = [tweet.date for tweet in tweets]
        self.assertTrue(all([since_date <= date for date in tweets_date]))
        self.assertEqual(250, len(tweets))

    def test_retrieve_tweets_by_content_between_dates_no_limit_before(self):
        self._add_claim("post0", "The Rock Running for President", "2017-02-03 00:00:00")
        self._db.commit()
        since_date = str_to_date("2017-01-03 00:00:00")
        until_date = str_to_date("2017-03-03 00:00:00")
        self.tweets_crawler._limit_start_date = False
        self.tweets_crawler._limit_end_date = True
        self.tweets_crawler._max_num_tweets = 250
        self.tweets_crawler._month_interval = 1
        tweets = self.tweets_crawler._retrieve_old_tweets(self._claims["post0"], "The Rock Running for President")

        tweets_date = [tweet.date for tweet in tweets]
        self.assertTrue(all([date < until_date for date in tweets_date]))
        self.assertEqual(250, len(tweets))

    def test_execute_retrieve_tweets_by_full_content_1_month_interval(self):
        self._add_claim("post0", "The Rock Running for President", "2017-02-03 00:00:00",
                        "The Rock Running for President, Dwayne Running for President")
        self._db.commit()
        self.tweets_crawler._limit_start_date = True
        self.tweets_crawler._limit_end_date = True
        self.tweets_crawler._max_num_tweets = 133
        self.tweets_crawler._month_interval = 1
        self.tweets_crawler._actions = ['get_old_tweets_by_claims_content']
        self.tweets_crawler.execute()

        tweets_before = self.tweets_crawler._claim_id_tweets_id_before_dict["post0"]
        tweets_after = self.tweets_crawler._claim_id_tweets_id_after_dict["post0"]
        self.assertEqual(0, len(tweets_before & tweets_after))
        tweets_retrieved = len(tweets_before) + len(tweets_after)
        self.assertGreaterEqual(133, tweets_retrieved)
        self.assertEqual(tweets_retrieved, len(self._db.get_posts()))
        self.assertEqual(tweets_retrieved, len(self._db.get_claim_tweet_connections()))
        self.assertLess(0, tweets_retrieved)

    def test_execute_retrieve_tweets_by_key_words_1_month_interval(self):
        self._add_claim("post0", "The Rock Running for President", "2017-02-03 00:00:00",
                        "The Rock Running for President,Dwayne Running for President")
        self._db.commit()
        self.tweets_crawler._limit_start_date = True
        self.tweets_crawler._limit_end_date = True
        self.tweets_crawler._max_num_tweets = 141
        self.tweets_crawler._month_interval = 1
        self.tweets_crawler._actions = ['get_old_tweets_by_claims_keywords']
        self.tweets_crawler.execute()

        tweets_before = self.tweets_crawler._claim_id_tweets_id_before_dict["post0"]
        tweets_after = self.tweets_crawler._claim_id_tweets_id_after_dict["post0"]
        self.assertEqual(0, len(tweets_before & tweets_after))
        tweets_retrieved = len(tweets_before) + len(tweets_after)
        self.assertGreaterEqual(141 * 3, tweets_retrieved)
        self.assertEqual(tweets_retrieved, len(self._db.get_posts()))
        self.assertEqual(tweets_retrieved, len(self._db.get_claim_tweet_connections()))
        self.assertLess(0, tweets_retrieved)

    def _add_author(self, author_guid):
        author = Author()
        author.author_guid = author_guid
        author.author_full_name = 'test author'
        author.author_screen_name = author_guid
        author.name = 'test'
        author.domain = 'tests'
        author.statuses_count = 0
        author.created_at = "2017-06-14 05:00:00"
        self._db.add_author(author)
        self._author = author

    def _add_post(self, post_id, content, tags, date_str, domain='Microblog'):
        post = Post()
        post.author = self._author.author_guid
        post.author_guid = self._author.author_guid
        post.content = content
        post.title = post_id
        post.domain = domain
        post.post_id = post_id
        post.guid = post.post_id
        post.date = convert_str_to_unicode_datetime(date_str)
        post.created_at = post.date
        post.tags = tags
        self._db.addPost(post)
        self._author.statuses_count += 1

    def _add_claim(self, claim_id, content, date_str, keywords="", post_type=None):
        claim = Claim()
        claim.claim_id = claim_id
        claim.verdict = post_type
        claim.title = claim_id
        claim.description = content
        claim.verdict_date = convert_str_to_unicode_datetime(date_str)
        claim.keywords = keywords
        claim.url = "claim url"
        self._db.addPost(claim)
        self._claims[claim.claim_id] = claim
