
from commons.commons import *
import random
from multiprocessing import Pool
# import unicodedata
import sys
from abc import ABCMeta, abstractmethod
from collections import defaultdict, Counter
from datetime import timedelta

from DB.schema_definition import Post, Claim_Tweet_Connection, Claim_Keywords_Connections, Term, Topic, AuthorConnection
from commons.commons import compute_post_guid, date_to_str, compute_author_guid_by_author_name
from commons.method_executor import Method_Executor
from preprocessing_tools.keywords_generator import KeywordsGenerator
from vendors.GetOldTweets.got import manager
import itertools
import datetime
import requests

__author__ = "Aviad Elyashar"


class OldTweetsCrawler(Method_Executor, metaclass=ABCMeta):
    def __init__(self, db):
        Method_Executor.__init__(self, db)
        self._lang = self._config_parser.eval(self.__class__.__name__, "language")
        self._max_num_tweets = self._config_parser.eval(self.__class__.__name__, "max_num_tweets")
        self._max_num_of_objects_without_saving = self._config_parser.eval(self.__class__.__name__,
                                                                           "max_num_of_objects_without_saving")
        self._month_interval = self._config_parser.eval(self.__class__.__name__, "month_interval")
        self._output_folder_full_path = self._config_parser.eval(self.__class__.__name__, "output_folder_full_path")
        self._limit_start_date = self._config_parser.eval(self.__class__.__name__, "limit_start_date")
        self._limit_end_date = self._config_parser.eval(self.__class__.__name__, "limit_end_date")
        self._claim_without_tweets_only = self._config_parser.eval(self.__class__.__name__, "claim_without_tweets_only")
        self._keywords_types = self._config_parser.eval(self.__class__.__name__, "keywords_types")
        self._claim_index_to_start_crawling = self._config_parser.eval(self.__class__.__name__,
                                                                       "claim_index_to_start_crawling")

        self._targeted_claim_ids_for_crawling = self._config_parser.eval(self.__class__.__name__,
                                                                         "targeted_claim_ids_for_crawling")
        self._topic_terms_dict = self._config_parser.eval(self.__class__.__name__, "topic_terms_dict")

        self._start_date_interval = self._config_parser.eval(self.__class__.__name__, "start_date_interval")
        self._start_date_interval = str_to_date(self._start_date_interval)
        self._end_date_interval = self._config_parser.eval(self.__class__.__name__, "end_date_interval")
        self._end_date_interval = str_to_date(self._end_date_interval)

        self._claim_id_tweets_id_before_dict = defaultdict(set)
        self._claim_id_tweets_id_after_dict = defaultdict(set)
        self._posts = []
        self._claim_post_connections = []
        self._retrieved = 0
        self._interval_timeline_tweets = []

    def get_old_tweets_by_claims_content(self):
        self._base_retrieve_tweets_from_claims(self._retrieve_tweets_from_claims_by_content)

    def get_old_tweets_by_claims_keywords(self):
        self._base_retrieve_tweets_from_claims(self._retrieve_tweets_from_claims_by_keywords)

    def _retrieve_tweets_from_claims_by_content(self, claims):
        for i, claim in enumerate(claims):
            tweets = self._retrieve_old_tweets(claim, claim.description)
            msg = "Processing claims {0}/{1} Retreived {2} tweets".format(str(i + 1), len(claims), self._retrieved)
            print(msg)
            self._add_tweets_and_connections_to_db(claim, tweets)
        print()

    def retrieve_tweets_by_claim_keywords(self, claim, keywords):
        tweets = self._retrieve_old_tweets(claim, keywords)
        # self._retrieve_tweets_from_claims_by_keywords([claim], reduce_keywords)
        # self._save_posts_claim_tweet_connections_to_db()
        return tweets

    def _retrieve_tweets_from_claims_by_keywords(self, claims, reduce_keywords=True):
        for keyword_type in self._keywords_types:
            # print('######## {} ########'.format(keyword_type))
            claim_id_keywords_dict = self._db.get_claim_id_keywords_dict_by_connection_type(keyword_type)
            num_of_claims = len(claims)
            for i, claim in enumerate(claims):
                try:
                    self.get_tweets_for_claim_by_keywords(claim, claim_id_keywords_dict, i, keyword_type, num_of_claims,
                                                          reduce_keywords)
                except Exception as e:
                    print(e)

    def get_tweets_for_claim_by_keywords(self, claim, claim_id_keywords_dict, i, keyword_type, num_of_claims,
                                         reduce_keywords):
        try:
            if keyword_type == 'from_claims_table':
                self.retrive_tweets_from_claim_table_keywords(claim, i, num_of_claims)
            else:
                self._domain = keyword_type
                keywords_list = claim_id_keywords_dict[claim.claim_id].lower().strip().split('||')
                total_retrieved = 0
                for keywords in keywords_list:
                    if 'tf_idf' in keyword_type:
                        word_rank_mapper = KeywordsGenerator(self._db).get_word_tf_idf_of_claims()
                    elif 'pos_tagging' in keyword_type:
                        word_rank_mapper = KeywordsGenerator(self._db).get_word_pos_tagging_dict(claim)
                    else:
                        word_rank_mapper = {word: random.random() for word in keywords.split(' ')}
                    retrieved_tweets_count = self.reduce_keywords_by_word_mapper(claim, keywords, word_rank_mapper,
                                                                                 keyword_type, reduce_keywords)

                    print('\rtweets retrieved {0}'.format(retrieved_tweets_count), end='')
                    total_retrieved += retrieved_tweets_count
                self._save_posts_claim_tweet_connections_to_db()
                msg = "\nProcessing claims {0}/{1}, Retreived {2} tweets ,type {3}".format(str(i + 1), num_of_claims,
                                                                                           total_retrieved,
                                                                                           keyword_type)
                print(msg)
            # else:
            #     self._domain = keyword_type
            #     retrieved_tweets_count = 0
            #     tweets = self._retrieve_old_tweets(claim, claim_id_keywords_dict[claim.claim_id].lower().strip())
            #     retrieved_tweets_count += len(set(tweet.id for tweet in tweets))
            #     print('\rtweets retrieved {0}'.format(retrieved_tweets_count), end='')
            #
            #     msg = "\nProcessing claims {0}/{1}, Retreived {2} tweets".format(str(i + 1), num_of_claims,
            #                                                                      retrieved_tweets_count)
            #     print(msg)
            #     self._add_tweets_and_connections_to_db(claim, tweets)
        except Exception as e:
            print(e)

    def reduce_keywords_by_word_mapper(self, claim, keywords, word_rank_mapper, keyword_type, reduce_keywords=True):
        # while True:
        retrieved_tweets_count = 0
        tweets = self._retrieve_old_tweets(claim, keywords)
        retrieved_tweets_count += len(set(tweet.id for tweet in tweets))

        # keywords_size = len(keywords.split(' '))
        # if len(tweets) > 0 or keywords_size == 1 or not reduce_keywords:
        #     break
        # keyword_to_rank = Counter({word: word_rank_mapper[word] for word in keywords.split(' ')})
        # keywords = ' '.join([word for word, rank in keyword_to_rank.most_common(keywords_size - 1)])
        # self._add_new_keywords(claim, keywords, keyword_type)
        self._add_tweets_and_connections_to_db(claim, tweets)
        return retrieved_tweets_count

    def retrive_tweets_from_claim_table_keywords(self, claim, i, num_of_claims):
        keywords_str = claim.keywords
        keywords = keywords_str.split("||")
        retrieved_tweets_count = 0
        for keyword in keywords:
            tweets = self._retrieve_old_tweets(claim, keyword.lower().strip())
            retrieved_tweets_count += len(set(tweet.id for tweet in tweets))
            self._add_tweets_and_connections_to_db(claim, tweets)
        msg = "\nProcessing claims {0}/{1}, Retreived {2} tweets".format(str(i + 1), num_of_claims,
                                                                         retrieved_tweets_count)
        print(msg)

    def _base_retrieve_tweets_from_claims(self, retrieve_tweets_function):
        # self._db.session.expire_on_commit = False
        if self._claim_without_tweets_only:
            claims = self._db.get_claims_without_tweets()
        else:
            claims = self._db.get_claims()
        claims = claims[self._claim_index_to_start_crawling:]
        self._claim_dict = {claim.claim_id: claim for claim in claims}

        if len(self._targeted_claim_ids_for_crawling) > 0:
            claims = []
            for targeted_claim_id in self._targeted_claim_ids_for_crawling:
                claim = self._claim_dict[targeted_claim_id]
                claims.append(claim)

        retrieve_tweets_function(claims)

        self._save_posts_claim_tweet_connections_to_db()
        # self._db.insert_or_update_authors_from_posts(self._domain, {}, {})

        before_dict = self._claim_id_tweets_id_before_dict
        after_dict = self._claim_id_tweets_id_after_dict
        output_folder_path = self._output_folder_full_path
        self._export_csv_files_for_statistics(output_folder_path, before_dict, after_dict, self._claim_dict)
        # self._db.session.expire_on_commit = True

    def _retrieve_old_tweets(self, claim, content):
        datetime_object = claim.verdict_date
        month_interval = timedelta(self._month_interval * 365 / 12)
        start_date = date_to_str(datetime_object - month_interval, "%Y-%m-%d")
        end_date = date_to_str(datetime_object + month_interval, "%Y-%m-%d")
        tweets = []
        try:
            tweets = self._retrieve_tweets_between_dates(claim, content, start_date, end_date)
        except:
            e = sys.exc_info()[0]
            print("tweet content: {0}, error:{1}".format(content, e))

        return tweets



    def _get_and_add_tweets_by_content_and_date(self, content, since, until):
        tweetCriteria = manager.TweetCriteria().setQuerySearch(content).setMaxTweets(self._max_num_tweets)
        if self._limit_start_date:
            tweetCriteria = tweetCriteria.setSince(since)
        if self._limit_end_date:
            tweetCriteria = tweetCriteria.setUntil(until)
        tweets = manager.TweetManager.getTweets(tweetCriteria, lang=self._lang)
        # self._add_tweets_and_connections_to_db(claim, tweets)
        return tweets

    def _retrieve_tweets_between_dates(self, claim, content, start_date, current_date):
        original_claim_id = str(claim.claim_id)
        tweets = self._get_and_add_tweets_by_content_and_date(content, start_date, current_date)
        for tweet in tweets:
            if tweet.date < claim.verdict_date:
                self._claim_id_tweets_id_before_dict[original_claim_id].add(tweet.id)
            else:
                self._claim_id_tweets_id_after_dict[original_claim_id].add(tweet.id)
        return tweets

    def _retweet_tweets_between_dates_no_claim(self, content, start_date, end_date):
        tweets = self._get_and_add_tweets_by_content_and_date(content, start_date, end_date)
        return tweets


    def _save_posts_claim_tweet_connections_to_db(self):
        self._db.add_posts_fast(self._posts)
        self._db.add_claim_tweet_connections_fast(self._claim_post_connections)
        self._db.session.commit()
        self._posts = []
        self._claim_post_connections = []

    def _add_tweets_and_connections_to_db(self, claim, tweets):
        original_claim_id = str(claim.claim_id)
        claim_verdict = claim.verdict

        posts_per_claim, claim_connections = self._convert_tweets_to_posts(tweets, original_claim_id, claim_verdict)
        self._posts += posts_per_claim
        self._claim_post_connections += claim_connections
        if len(self._posts) > self._max_num_of_objects_without_saving:
            self._save_posts_claim_tweet_connections_to_db()

    def _convert_tweets_to_posts(self, tweets, original_claim_id, post_type):
        posts = []
        claim_post_connections = []
        seen_tweets = set()
        for tweet in tweets:
            if tweet.id not in seen_tweets:
                seen_tweets.add(tweet.id)
                post = self._convert_tweet_to_post(tweet, post_type)
                claim_post_connection = self._create_claim_post_connection(original_claim_id, post.post_id)

                claim_post_connections.append(claim_post_connection)
                posts.append(post)
        self._retrieved = len(seen_tweets)
        return posts, claim_post_connections

    def _convert_tweet_to_post(self, tweet, post_type):
        post = Post()

        post.post_osn_id = str(tweet.id)
        post_creation_date = tweet.date
        created_at = str(date_to_str(post_creation_date))
        post.created_at = created_at

        post.date = post_creation_date
        post.favorite_count = tweet.favorites
        post.retweet_count = tweet.retweets
        post.content = str(tweet.text)

        author_name = str(tweet.username)
        post.author = author_name
        post.author_guid = compute_author_guid_by_author_name(author_name)
        post_url = tweet.permalink
        post.url = str(post_url)

        post_guid = compute_post_guid(post_url, author_name, created_at)
        post.guid = post_guid
        post.post_id = post_guid
        post.domain = self._domain

        post.post_type = post_type
        return post

    def _create_claim_post_connection(self, original_claim_id, post_id):
        claim_post_connection = Claim_Tweet_Connection()
        claim_post_connection.claim_id = original_claim_id
        claim_post_connection.post_id = post_id
        return claim_post_connection

    def _add_new_keywords(self, claim, keywords_str, type_name, score=None, tweet_count=None):
        claim_keywords_connections = Claim_Keywords_Connections()
        claim_keywords_connections.claim_id = claim.claim_id
        claim_keywords_connections.keywords = keywords_str
        claim_keywords_connections.type = type_name
        if score:
            claim_keywords_connections.score = score
        if tweet_count:
            claim_keywords_connections.tweet_count = tweet_count
        self._db.addPosts([claim_keywords_connections])

    def collect_tweets_by_manual_keywords(self):
        old_terms, terms_to_add, all_config_terms, term_term_id_dict = self._convert_keywords_to_terms_by_given_dict()
        topics = self._create_topics_by_given_dict(term_term_id_dict)
        self._db.addPosts(terms_to_add)
        self._db.addPosts(topics)

        today = datetime.datetime.today()
        month_interval = timedelta(self._month_interval * 365 / 12)
        before = today - month_interval

        start_date = date_to_str(before, "%Y-%m-%d")
        end_date = date_to_str(today, "%Y-%m-%d")

        for i, term in enumerate(all_config_terms):
            print("\rTerm: {0} {1}/{2}".format(term, i, len(all_config_terms)), end='')

            tweets = self._collect_tweets_by_term(term, start_date, end_date)
            print("\rTerm: {0} Num tweets retrieved: {1}".format(term, len(tweets)), end='')
            posts, claim_post_connections = self._convert_tweets_to_posts(tweets, term_term_id_dict[term], self._domain)

            self._db.addPosts(posts)
            self._db.addPosts(claim_post_connections)
            # total_posts += total_posts + posts
            # total_connections += total_connections + claim_post_connections

        # self._db.addPosts(total_posts)
        # self._db.addPosts(total_connections)

    def _collect_tweets_by_term(self, term, start_date, end_date):
        tweets = []
        try:
            tweets = self._retweet_tweets_between_dates_no_claim(term, start_date, end_date)
        except:
            e = sys.exc_info()[0]
            print("tweet content: {0}, error:{1}".format(term, e))

        return tweets



    def _convert_keywords_to_terms_by_given_dict(self):
        terms = self._db.get_terms()
        term_term_id_dict = {term.description : term.term_id for term in terms}

        old_terms = set(term_term_id_dict.keys())

        list_of_terms = list(self._topic_terms_dict.values())
        optional_terms = list(itertools.chain(*list_of_terms))

        optional_terms = set(optional_terms)

        keywords_to_add_set = optional_terms - old_terms
        keywords_to_add = list(keywords_to_add_set)

        term_ids = list(term_term_id_dict.values())
        if len(term_ids) > 0:
            max_term_id = max(term_ids)
            new_term_id = max_term_id + 1
        else:
            new_term_id = 1
        terms_to_add = []
        for keyword in keywords_to_add:
            term = Term()
            term.term_id = new_term_id

            term_term_id_dict[keyword] = new_term_id

            term.description = keyword
            terms_to_add.append(term)
            new_term_id += 1
        return terms, terms_to_add, optional_terms, term_term_id_dict

    def _create_topics_by_given_dict(self, term_term_id_dict):
        topics = self._db.get_topics()
        topic_ids = [topic.topic_id for topic in topics]

        topic_desc_topic_id_dict = {topic.description: topic.topic_id for topic in topics}

        topic_descriptions = list(topic_desc_topic_id_dict.keys())
        optional_topics = list(self._topic_terms_dict.keys())

        topics_to_add = list(set(optional_topics) - set(topic_descriptions))

        topic_index_to_assign = self._find_id_to_assign(topic_ids)

        new_topic_desc_topic_id_dict = {}
        for topic_to_add in topics_to_add:
            new_topic_desc_topic_id_dict[topic_to_add] = topic_index_to_assign
            topic_index_to_assign += 1

        new_topics = []
        for topic_description, terms in self._topic_terms_dict.items():
            if topic_description in new_topic_desc_topic_id_dict:
                for term in terms:
                    term_id = term_term_id_dict[term]

                    topic = Topic()
                    topic_id = new_topic_desc_topic_id_dict[topic_description]
                    topic.topic_id = topic_id
                    topic.term_id = term_id
                    topic.description = topic_description

                    new_topics.append(topic)
        return new_topics


    def _find_id_to_assign(self, ids):
        if len(ids) > 0:
            max_id = max(ids)
            new_id = max_id + 1
        else:
            new_id = 1

        return new_id


    def crawl_timelines_for_users_by_interval(self):
        authors = self._db.get_authors()

        self._author_osn_id_author_dict = {}
        for author in authors:
            author_osn_id = author.author_osn_id
            protected = author.protected
            if protected == False:
                self._author_osn_id_author_dict[author_osn_id] = author

        headers = self._create_headers()
        original_params = self._get_parameters()

        #headers = self._update_cookies(headers)


        author_osn_ids = list(self._author_osn_id_author_dict.keys())
        author_osn_ids = author_osn_ids[5:]

        for i, author_osn_id in enumerate(author_osn_ids):
            print("crawling tweets for author_id:{0} {1}/{2}".format(author_osn_id, i, len(author_osn_ids)))
            self._stop_searching_for_author_tweets = False
            self._orignal_request = None
            self._next_cursor_for_scrolling = None
            self._scroll_count = 0

            while self._stop_searching_for_author_tweets is not True:
                # targeted_url = "https://api.twitter.com/2/timeline/profile/114894966.json?include_profile_interstitial_type=1&include_blocking=1&include_blocked_by=1&include_followed_by=1&include_want_retweets=1&include_mute_edge=1&include_can_dm=1&include_can_media_tag=1&skip_status=1&cards_platform=Web-12&include_cards=1&include_composer_source=true&include_ext_alt_text=true&include_reply_count=1&tweet_mode=extended&include_entities=true&include_user_entities=true&include_ext_media_color=true&include_ext_media_availability=true&send_error_codes=true&simple_quoted_tweet=true&include_tweet_replies=false&userId=114894966&count=20&ext=mediaStats,highlightedLabel,cameraMoment&include_quote_count=true"
                targeted_url = "https://api.twitter.com/2/timeline/profile/{0}.json".format(author_osn_id)

                try:
                    # sending get request and saving the response as response object
                    if self._next_cursor_for_scrolling is None:
                        response = requests.get(url=targeted_url, params=original_params, headers=headers)
                        #self._orignal_request = response.url
                    else:
                        current_params = original_params.copy()
                        #next_request_url = self._orignal_request + '&cursor=' + self._next_cursor_for_scrolling
                        current_params['cursor'] = self._next_cursor_for_scrolling
                        response = requests.get(url=targeted_url, params=current_params, headers=headers)
                        #response = requests.get(url=next_request_url)
                    print("scroll count for author:{0} is {1}".format(author_osn_id, self._scroll_count))

                    time_to_wait = random.randint(10, 30)
                    count_down_time(time_to_wait)
                    if response.status_code == 200:
                        self._handle_ok_response(response, author_osn_id, i, len(author_osn_ids))
                    else:
                        self._handle_bad_response(response, author_osn_id, i, len(author_osn_ids))
                        headers = self._update_cookies(headers)
                        if self._next_cursor_for_scrolling is None:
                            response = requests.get(url=targeted_url, params=original_params, headers=headers)
                        else:
                            current_params = original_params.copy()
                            # next_request_url = self._orignal_request + '&cursor=' + self._next_cursor_for_scrolling
                            current_params['cursor'] = self._next_cursor_for_scrolling
                            response = requests.get(url=targeted_url, params=current_params, headers=headers)
                        if response.status_code == 200:
                            self._handle_ok_response(response, author_osn_id, i, len(author_osn_ids))
                        else:
                            break


                except Exception as e:
                    print(e)

        self._convert_tweet_dicts_to_posts_and_save()


    def _create_headers(self):
        headers = {
            "Host": "api.twitter.com",
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:76.0) Gecko/20100101 Firefox/76.0",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.5",
            # "Accept-Encoding": "gzip, deflate, br",
            "Accept-Encoding": "json",
            "authorization": "Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA",
            # changed:
            "x-guest-token": "1268255708786630656",
            "x-twitter-client-language": "en",
            "x-twitter-active-user": "yes",
            # changed
            "x-csrf-token": "ad665cb4005404da6e4ebca8b71746cd",
            "Origin": "https://twitter.com",
            "DNT": "1",
            "Connection": "keep-alive",
            "Referer": "https://twitter.com/",
            "Cookie": "personalization_id=\"v1_9theD6yOz6BiNOUaApASSA==\"; guest_id=v1%3A159121069867184382; gt=1268255708786630656; ct0=ad665cb4005404da6e4ebca8b71746cd",
            # "Cookie": "personalization_id=\"v1_LvDcJ7yRQd7aQjKobp+Mrg==\"; guest_id=v1%3A159093064735858785; gt=1267081088444305410; ct0=5849c8",
            "TE": "Trailers",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache"
        }

        return headers

    def _get_parameters(self):
        # defining a params dict for the parameters to be sent to the API
        parameters = {
            "include_profile_interstitial_type": "1",
            "include_blocking": "1",
            "include_blocked_by": "1",
            "include_followed_by": "1",
            "include_want_retweets": "1",
            "include_mute_edge": "1",
            "include_can_dm": "1",
            "include_can_media_tag": "1",
            "skip_status": "1",
            "cards_platform": "Web-12",
            "include_cards": "1",
            "include_composer_source": "true",
            "include_ext_alt_text": "true",
            "include_reply_count": "1",
            "tweet_mode": "extended",
            "include_entities": "true",
            "include_user_entities": "true",
            "include_ext_media_color": "true",
            "include_ext_media_availability": "true",
            "send_error_codes": "true",
            "simple_quoted_tweets": "true",
            "include_tweet_replies": "false",
            "userId": "34613288",
            "count": "20",
            "ext": "mediaStats,highlightedLabel,cameraMoment"
        }

        return parameters

    def _search_for_tweets_in_interval(self, response_dict, author_osn_id, i, authors_count):
        tweet_dates = []

        tweets_dict = response_dict['globalObjects']['tweets']
        tweets_ids = list(tweets_dict.keys())

        for tweet_id in tweets_ids:
            tweet = tweets_dict[tweet_id]
            created_at = tweet['created_at']
            creation_date_str = extract_tweet_publiction_date(created_at)
            creation_date = str_to_date(creation_date_str)

            tweet_dates.append(creation_date)

            if creation_date >= self._start_date_interval and creation_date <= self._end_date_interval:
                tweet['author_osn_id'] = author_osn_id
                self._interval_timeline_tweets.append(tweet)
        print("author_id: {0}, {1}/{2}, Num_of_tweets_in_interval: {3}, cursor={4}".format(author_osn_id, i,
                                                                                           authors_count,
                                                                                           len(self._interval_timeline_tweets),
                                                                                           self._next_cursor_for_scrolling))
        oldest_date_published_tweet = min(tweet_dates)
        return oldest_date_published_tweet

    def _convert_tweet_dicts_to_posts_and_save(self):
        posts = []
        total_author_connections = []
        for tweet_dict in self._interval_timeline_tweets:
            post = self._convert_tweet_dict_to_post(tweet_dict)
            posts.append(post)

            entities_dict = tweet_dict['entities']
            if not entities_dict and len(list(entities_dict.keys())) > 0:
                connections = self._convert_tweet_dict_to_author_connections(tweet_dict, post)
                total_author_connections += connections

        self._db.addPosts(posts)
        self._db.addPosts(total_author_connections)
        self._interval_timeline_tweets = []

    def _convert_tweet_dict_to_post(self, tweet_dict):
        post = Post()

        post_osn_id = tweet_dict['id_str']
        post.post_osn_id = post_osn_id

        author_osn_id = tweet_dict['author_osn_id']
        author = self._author_osn_id_author_dict[author_osn_id]
        author_screen_name = author.author_screen_name
        post.author = author_screen_name

        post.author_guid = compute_author_guid_by_author_name(author_screen_name)


        created_at = tweet_dict['created_at']
        post.created_at = created_at

        creation_date_str = extract_tweet_publiction_date(created_at)
        creation_date = str_to_date(creation_date_str)
        post.date = creation_date

        post.favorite_count = tweet_dict['favorite_count']
        post.retweet_count = tweet_dict['retweet_count']
        post.reply_count = tweet_dict['reply_count']
        post.content = str(tweet_dict['full_text'])
        post.domain = self._domain
        post.language = str(tweet_dict['lang'])

        post_url = "https://twitter.com/{0}/status/{1}".format(author_screen_name, post_osn_id)
        post.url = post_url

        post_guid = compute_post_guid(post_url, author_screen_name, creation_date_str)
        post.guid = post_guid
        post.post_id = post_guid

        return post

    def _convert_tweet_dict_to_author_connections(self, tweet_dict, post):
        connections = []
        entities_dict = tweet_dict['entities']

        if 'urls' in entities_dict:
            post_url_connections = self._create_post_url_connections(entities_dict, post)
            connections += post_url_connections

        if 'media' in entities_dict:
            post_media_url_connections = self._create_post_media_url_connections(entities_dict, post)
            connections += post_media_url_connections

        if 'user_mentions' in entities_dict:
            post_user_mention_connections = self._create_post_user_mention_connections(entities_dict, post)
            connections += post_user_mention_connections
        return connections

    def _create_post_url_connections(self, entities_dict, post):
        connections = []
        post_guid = post.post_id

        url_dicts = entities_dict['urls']
        for url_dict in url_dicts:
            expanded_url = url_dict['expanded_url']

            author_connection = AuthorConnection()

            author_connection.source_author_guid = post_guid
            author_connection.destination_author_guid = expanded_url
            author_connection.connection_type = "post_url_connection"
            connections.append(author_connection)
        return connections

    def _create_post_media_url_connections(self, entities_dict, post):
        connections = []
        post_guid = post.post_id

        media_dicts = entities_dict['media_url']
        for media_dict in media_dicts:
            media_url = media_dict['media_url']

            author_connection = AuthorConnection()
            author_connection.source_author_guid = post_guid
            author_connection.destination_author_guid = media_url
            author_connection.connection_type = "post_media_url_connection"

            connections.append(author_connection)
        return connections

    def _create_post_user_mention_connections(self, entities_dict, post):
        connections = []
        post_guid = post.post_id

        user_mention_dicts = entities_dict['user_mentions']
        for user_mention_dict in user_mention_dicts:
            screen_name = user_mention_dict['screen_name']

            author_connection = AuthorConnection()
            author_connection.source_author_guid = post_guid
            author_connection.destination_author_guid = screen_name
            author_connection.connection_type = "post_user_mention_connection"

            connections.append(author_connection)
        return connections

    def _extract_next_cursor_for_scrolling(self, response_dict):
        cursor_dict = response_dict['timeline']['instructions'][0]['addEntries']['entries'][-1]['content']['operation']['cursor']
        next_cursor = cursor_dict['value']
        return next_cursor



    def _handle_ok_response(self, response, author_osn_id, i, authors_count):
        response_dict = response.json()
        next_cursor = self._extract_next_cursor_for_scrolling(response_dict)

        if self._next_cursor_for_scrolling != next_cursor:
            self._next_cursor_for_scrolling = next_cursor
            self._scroll_count += 1

            oldest_date_published_tweet = self._search_for_tweets_in_interval(response_dict, author_osn_id, i, authors_count)

            if oldest_date_published_tweet <= self._start_date_interval:
                self._stop_searching_for_author_tweets = True
                self._scroll_count = 0
        else:
            self._stop_searching_for_author_tweets = True
            self._scroll_count = 0

        if len(self._interval_timeline_tweets) >= self._max_num_of_objects_without_saving:
            self._convert_tweet_dicts_to_posts_and_save()

    def _handle_bad_response(self, response, author_osn_id, i, authors_count):
        status_code = response.status_code
        text = response.text
        print("Error: author_osn_id:{0}, {1}/{2} status_code:{3}, text={4}".format(author_osn_id, i, authors_count,
                                                                                   status_code, text))

    def _update_cookies(self, headers):
        cookie_value = self._get_cookie_value()
        headers["Cookie"] = cookie_value
        return headers

    def _get_cookie_value(self):
        ordinary_twitter_url = "https://twitter.com/amit_segal"
        response = requests.get(url=ordinary_twitter_url)
        cookies = response.cookies
        cookies_tuples = []
        for cookie in cookies:
            cookie_tuple = (cookie.name, cookie.value)
            cookies_tuples.append(cookie_tuple)

        cookie_value = "personalization_id=\"{0}\"; guest_id={1}".format(cookies_tuples[1][1], cookies_tuples[0][1])
        return cookie_value

